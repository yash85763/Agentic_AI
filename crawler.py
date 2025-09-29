# site_graph_crawler.py
# Crawl → Parse → Build Graph → Export (sitemap + nodes/edges + GraphML)
# By default uses requests+BS4; enable Selenium for dynamic content with --use-selenium

import argparse
import hashlib
import json
import queue
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlparse

import networkx as nx
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from urllib import robotparser

# --- Optional Selenium (dynamic pages) ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_OK = True
except Exception:
    SELENIUM_OK = False


# -------------------- Data Models --------------------

@dataclass
class Link:
    href: str
    anchor_text: str
    position: str  # 'nav' | 'header' | 'footer' | 'body' | 'aside' | 'unknown'


@dataclass
class Section:
    section_id: str        # e.g. /docs/auth#h2-tokens
    level: int             # 1,2,3...
    heading_text: str
    heading_path: List[str]  # ["Docs","Auth","Tokens"]
    css_id: Optional[str] = None


@dataclass
class ToolSpec:
    tool_id: str
    kind: str              # 'form' | 'iframe' | 'script'
    name: Optional[str]
    details: Dict


@dataclass
class PageData:
    url: str
    status_code: int
    title: str
    text: str                  # main textual content (boilerplate-reduced)
    headings: List[Section]
    links: List[Link]
    tools: List[ToolSpec]
    last_modified: Optional[str]
    discovered_parent: Optional[str]  # first page where this link was found
    content_hash: str


# -------------------- Helpers --------------------

UA = "SiteGraphCrawler/1.0 (+research; contact: webmaster@example.com)"

def same_site(url: str, base_netloc: str, include_subdomains: bool) -> bool:
    nl = urlparse(url).netloc.lower()
    base = base_netloc.lower()
    if nl == base:
        return True
    if include_subdomains and nl.endswith("." + base):
        return True
    return False

def normalize_url(href: str, base_url: str) -> Optional[str]:
    if not href:
        return None
    absu = urljoin(base_url, href)
    absu, _frag = urldefrag(absu)
    parsed = urlparse(absu)
    if parsed.scheme not in ("http", "https"):
        return None
    # Canonicalize trivial differences (remove trailing slash for root-only)
    if parsed.path == "":
        absu = absu.replace(parsed.netloc + "", parsed.netloc + "/")
    return absu

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def position_of(a_tag) -> str:
    # Heuristic based on ancestor tag names / classes
    anc = [p.name for p in a_tag.parents if getattr(p, "name", None)]
    classes = " ".join(a_tag.get("class", []))
    if "nav" in anc or "navbar" in classes:
        return "nav"
    if "header" in anc:
        return "header"
    if "footer" in anc:
        return "footer"
    if "aside" in anc:
        return "aside"
    return "body"

def clean_text(soup: BeautifulSoup) -> str:
    # Remove obvious boilerplate
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    # Optionally drop nav/footer/aside
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    # Squeeze excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def extract_headings(soup: BeautifulSoup, page_url: str) -> List[Section]:
    heading_tags = []
    for lvl in range(1, 7):
        heading_tags.extend(soup.find_all(f"h{lvl}"))
    # Build hierarchical heading_path incrementally
    current_path: List[str] = []
    sections: List[Section] = []
    last_level = 0
    for h in heading_tags:
        level = int(h.name[1])
        text = re.sub(r"\s+", " ", h.get_text(" ", strip=True))
        if not text:
            continue
        # Adjust path by level
        if level > last_level:
            current_path.append(text)
        else:
            # pop back to parent level
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(text)
        last_level = level
        css_id = h.get("id")
        frag = css_id or f"h{level}-" + sha1(text)[:8]
        section_id = urlparse(page_url).path + "#" + frag
        sections.append(
            Section(
                section_id=section_id,
                level=level,
                heading_text=text,
                heading_path=current_path.copy(),
                css_id=css_id,
            )
        )
    return sections

TOOL_NAME_HINTS = re.compile(r"(calc|calculator|tool|widget|search|converter|estimator)", re.I)

def detect_tools(soup: BeautifulSoup, page_url: str) -> List[ToolSpec]:
    tools: List[ToolSpec] = []

    # Forms deemed interactive “tools”
    for f in soup.find_all("form"):
        attrs = " ".join([f.get("id", ""), " ".join(f.get("class", []))])
        inputs = [i.get("name") or i.get("id") for i in f.find_all(["input", "select", "textarea"])]
        if TOOL_NAME_HINTS.search(attrs) or len(inputs) >= 2:
            name = f.get("id") or f.get("name") or f.get("action") or "form"
            action = f.get("action")
            method = (f.get("method") or "GET").upper()
            tool_id = f"tool:{urlparse(page_url).path}#form-{sha1(name)[:8]}"
            tools.append(
                ToolSpec(
                    tool_id=tool_id,
                    kind="form",
                    name=name,
                    details={"action": action, "method": method, "inputs": inputs},
                )
            )

    # Iframes hosting calculators/dashboards
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src") or ""
        title = iframe.get("title") or ""
        if TOOL_NAME_HINTS.search(src) or TOOL_NAME_HINTS.search(title):
            tool_id = f"tool:{urlparse(page_url).path}#iframe-{sha1(src)[:8]}"
            tools.append(
                ToolSpec(
                    tool_id=tool_id,
                    kind="iframe",
                    name=title or "iframe",
                    details={"src": urljoin(page_url, src)},
                )
            )

    # Embedded scripts that look like tools
    for s in soup.find_all("script", src=True):
        src = s.get("src") or ""
        if TOOL_NAME_HINTS.search(src):
            tool_id = f"tool:{urlparse(page_url).path}#script-{sha1(src)[:8]}"
            tools.append(
                ToolSpec(
                    tool_id=tool_id,
                    kind="script",
                    name="script",
                    details={"src": urljoin(page_url, src)},
                )
            )

    return tools

def get_html_via_requests(url: str, timeout: float = 15.0) -> Optional[str]:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if resp.status_code >= 400:
            return None
        return resp.text
    except RequestException:
        return None

def get_html_via_selenium(url: str, driver) -> Optional[str]:
    try:
        driver.get(url)
        time.sleep(0.5)  # small wait for dynamic content
        return driver.page_source
    except Exception:
        return None


# -------------------- Crawler --------------------

class SiteCrawler:
    def __init__(
        self,
        base_url: str,
        max_pages: int = 1000,
        include_subdomains: bool = False,
        use_selenium: bool = False,
        crawl_delay: float = 0.4,
    ):
        self.base_url = base_url.rstrip("/")
        self.base_parsed = urlparse(self.base_url)
        self.base_netloc = self.base_parsed.netloc
        self.max_pages = max_pages
        self.include_subdomains = include_subdomains
        self.use_selenium = use_selenium and SELENIUM_OK
        self.crawl_delay = crawl_delay

        self.robot = robotparser.RobotFileParser()
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            self.robot.set_url(robots_url)
            self.robot.read()
        except Exception:
            pass

        self.driver = None
        if self.use_selenium:
            opts = ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-gpu")
            self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)

    def allowed_by_robots(self, url: str) -> bool:
        try:
            return self.robot.can_fetch(UA, url)
        except Exception:
            return True

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

    def crawl(self) -> Tuple[Dict[str, PageData], List[Tuple[str, str, Link]]]:
        pages: Dict[str, PageData] = {}
        link_edges: List[Tuple[str, str, Link]] = []

        seen: Set[str] = set()
        q = queue.Queue()
        q.put((self.base_url, None))  # (url, discovered_parent)

        while not q.empty() and len(pages) < self.max_pages:
            url, parent = q.get()

            if url in seen:
                continue
            seen.add(url)

            if not same_site(url, self.base_netloc, self.include_subdomains):
                continue

            if not self.allowed_by_robots(url):
                print(f"[robots] Skipping {url}", file=sys.stderr)
                continue

            html = None
            if self.use_selenium and self.driver:
                html = get_html_via_selenium(url, self.driver)
            if html is None:
                html = get_html_via_requests(url)

            if html is None:
                continue

            soup = BeautifulSoup(html, "lxml")

            # Extract title/meta
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")
            last_modified = None
            # Links
            raw_links = []
            for a in soup.find_all("a", href=True):
                href = normalize_url(a.get("href"), url)
                if not href:
                    continue
                lnk = Link(href=href, anchor_text=a.get_text(" ", strip=True)[:160], position=position_of(a))
                raw_links.append(lnk)

            # Sections & text
            headings = extract_headings(soup, url)
            text = clean_text(soup)
            content_hash = sha1(url + "|" + text[:4096])  # bounded for speed
            tools = detect_tools(soup, url)

            # Build page data
            pd = PageData(
                url=url,
                status_code=200,
                title=title,
                text=text,
                headings=headings,
                links=raw_links,
                tools=tools,
                last_modified=last_modified,
                discovered_parent=parent,
                content_hash=content_hash,
            )
            pages[url] = pd

            # Enqueue children
            for l in raw_links:
                if same_site(l.href, self.base_netloc, self.include_subdomains):
                    if l.href not in seen:
                        q.put((l.href, url))
                        link_edges.append((url, l.href, l))

            time.sleep(self.crawl_delay)

        return pages, link_edges


# -------------------- Graph Builder & Exports --------------------

class GraphBuilder:
    def __init__(self):
        self.G = nx.MultiDiGraph()  # allow parallel edges & typed relations

    def add_page(self, page: PageData):
        self.G.add_node(
            f"page:{page.url}",
            kind="page",
            url=page.url,
            title=page.title,
            content_hash=page.content_hash,
        )

        # Sections and containment edges
        for sec in page.headings:
            self.G.add_node(
                f"section:{sec.section_id}",
                kind="section",
                url=page.url,
                section_id=sec.section_id,
                level=sec.level,
                heading_text=sec.heading_text,
                heading_path=sec.heading_path,
                css_id=sec.css_id,
            )
            self.G.add_edge(f"page:{page.url}", f"section:{sec.section_id}", rel="contains")
            # parent_of between headings by level
        # parent_of for sections (simple stack by level)
        stack: List[Tuple[int, str]] = []
        for sec in page.headings:
            node = f"section:{sec.section_id}"
            while stack and stack[-1][0] >= sec.level:
                stack.pop()
            if stack:
                parent_node = stack[-1][1]
                self.G.add_edge(parent_node, node, rel="parent_of")
            stack.append((sec.level, node))

        # Tools and embed edges
        for tool in page.tools:
            self.G.add_node(
                tool.tool_id,
                kind="tool",
                tool_kind=tool.kind,
                name=tool.name,
                details=json.dumps(tool.details, ensure_ascii=False),
                page_url=page.url,
            )
            self.G.add_edge(f"page:{page.url}", tool.tool_id, rel="embeds_tool")

    def add_link(self, src_url: str, dst_url: str, link: Link):
        self.G.add_edge(
            f"page:{src_url}",
            f"page:{dst_url}",
            rel="links_to",
            anchor_text=link.anchor_text,
            position=link.position,
        )

    def export(self, outdir: Path, pages: Dict[str, PageData]):
        outdir.mkdir(parents=True, exist_ok=True)

        # Nodes/Edges JSONL
        nodes_path = outdir / "nodes.jsonl"
        edges_path = outdir / "edges.jsonl"
        with nodes_path.open("w", encoding="utf-8") as nf:
            for n, data in self.G.nodes(data=True):
                record = {"id": n}
                record.update(data)
                nf.write(json.dumps(record, ensure_ascii=False) + "\n")

        with edges_path.open("w", encoding="utf-8") as ef:
            for u, v, data in self.G.edges(data=True):
                record = {"src": u, "dst": v, "rel": data.get("rel")}
                for k, val in data.items():
                    if k != "rel":
                        record[k] = val
                ef.write(json.dumps(record, ensure_ascii=False) + "\n")

        # GraphML (nice for Gephi/Cytoscape)
        nx.write_graphml(self.G, outdir / "site.graphml")

        # Sitemap as a tree using discovered_parent (first-hit tree)
        tree = self._build_sitemap_tree(pages)
        with (outdir / "sitemap.json").open("w", encoding="utf-8") as sf:
            json.dump(tree, sf, ensure_ascii=False, indent=2)

    def _build_sitemap_tree(self, pages: Dict[str, PageData]):
        children = defaultdict(list)
        for url, pd in pages.items():
            parent = pd.discovered_parent
            children[parent].append(url)

        def build(node_url):
            entry = {
                "url": node_url,
                "title": self.G.nodes.get(f"page:{node_url}", {}).get("title", ""),
                "children": [build(c) for c in sorted(children.get(node_url, []))],
            }
            return entry

        # root is the base page (the one with discovered_parent=None)
        roots = [url for url, pd in pages.items() if pd.discovered_parent is None]
        roots = sorted(roots)
        return [build(r) for r in roots]


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Crawl a website and build a structural graph + sitemap.")
    ap.add_argument("--base-url", required=True, help="Starting URL, e.g., https://example.com")
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--include-subdomains", action="store_true", help="Allow crawling subdomains")
    ap.add_argument("--use-selenium", action="store_true", help="Render dynamic pages with Selenium (optional)")
    ap.add_argument("--outdir", default="out_site_graph", help="Output directory")
    ap.add_argument("--delay", type=float, default=0.4, help="Crawl delay (seconds)")
    args = ap.parse_args()

    if args.use_selenium and not SELENIUM_OK:
        print("[WARN] Selenium not available. Install selenium and webdriver-manager, or omit --use-selenium.", file=sys.stderr)

    crawler = SiteCrawler(
        base_url=args.base_url,
        max_pages=args.max_pages,
        include_subdomains=args.include_subdomains,
        use_selenium=args.use_selenium,
        crawl_delay=args.delay,
    )

    try:
        pages, link_edges = crawler.crawl()
    finally:
        crawler.close()

    print(f"[INFO] Crawled {len(pages)} pages; {len(link_edges)} internal link edges.")

    gb = GraphBuilder()
    for url, pd in pages.items():
        gb.add_page(pd)
    for src, dst, lnk in link_edges:
        gb.add_link(src, dst, lnk)

    outdir = Path(args.outdir)
    gb.export(outdir, pages)

    print(f"[OK] Exported graph to {outdir}/site.graphml, nodes.jsonl, edges.jsonl, and sitemap.json")

if __name__ == "__main__":
    main()