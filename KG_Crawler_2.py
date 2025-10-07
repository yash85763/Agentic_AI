"""
Knowledge Graph Crawler – hardened version
- Keeps raw HTML and per‑page structure manifests (headings, breadcrumbs, anchor windows)
- Writes cleaned text exactly as before (URL on first line, into output_dir/pages/*.txt)
- Adds checkpoint/resume (queue + visited + partial graph)
- Can rebuild/refine the graph later from cached structure (no re-crawl)

NOTE: Start-of-run terminal prompts are unchanged from your original script.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
import hashlib
from collections import deque
import time
import networkx as nx


class KnowledgeGraphCrawler:
    """
    Crawls websites and builds a knowledge graph with relationships
    (with raw HTML capture, structure manifests, checkpointing, and refine-from-structure support)
    """

    def __init__(self, start_url, main_domains, include_domains=None,
                 nav_links=None, max_depth=None, nav_container_class="page-header",
                 output_dir="kg_output"):
        """
        Initialize the crawler

        Args:
            start_url (str): Starting URL to begin crawling
            main_domains (list): Domains to crawl with configurable depth
            include_domains (list): Domains to crawl up to depth 2
            nav_links (list): Navigation link patterns for "back_to_page" relationship
            max_depth (int): Maximum crawl depth (None = unlimited)
            nav_container_class (str): CSS class containing navigation menu
            output_dir (str): Directory for output files
        """
        self.start_url = start_url
        self.main_domains = [d.lower().strip() for d in main_domains]
        self.include_domains = [d.lower().strip() for d in (include_domains or [])]
        self.nav_links = set(nav_links) if nav_links else set()
        self.max_depth = max_depth
        self.nav_container_class = nav_container_class
        self.output_dir = output_dir

        # Internal state
        self.visited = set()
        self.graph = nx.DiGraph()
        self.page_contents = {}
        self.domain_depths = {}
        self.driver = None

        # Create output directories (keeps original 'pages' for cleaned text)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pages'), exist_ok=True)          # cleaned txt (as before)
        os.makedirs(os.path.join(output_dir, 'pages_html'), exist_ok=True)     # raw html
        os.makedirs(os.path.join(output_dir, 'structure'), exist_ok=True)      # per-page manifests
        os.makedirs(os.path.join(output_dir, 'state'), exist_ok=True)          # checkpoints

    # ------------------------- WebDriver -------------------------
    def setup_driver(self):
        """Setup and configure Selenium WebDriver"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.page_load_strategy = 'normal'

        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(60)
            self.driver.implicitly_wait(10)
            print("✓ WebDriver initialized successfully\n")
            return True
        except Exception as e:
            print(f"✗ Error initializing WebDriver: {e}")
            print("\nPlease ensure:")
            print("  1. Chrome browser is installed")
            print("  2. ChromeDriver is installed (pip install selenium)")
            print("  3. ChromeDriver version matches Chrome version")
            return False

    def wait_for_page_load(self):
        """Wait for page to fully load including dynamic content"""
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            try:
                WebDriverWait(self.driver, 5).until_not(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Loading') or contains(text(), 'loading') or contains(text(),'Please wait')]"))
                )
            except Exception:
                pass
            time.sleep(2.5)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);"); time.sleep(0.4)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"); time.sleep(0.4)
            self.driver.execute_script("window.scrollTo(0, 0);"); time.sleep(0.4)
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(1.5)
        except Exception as e:
            print(f"      Wait warning: {e}")
            time.sleep(3)

    # ------------------------- URL helpers -------------------------
    def get_domain(self, url):
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def is_main_domain(self, url):
        domain = self.get_domain(url)
        return any(domain == d or domain.endswith('.' + d) for d in self.main_domains)

    def is_include_domain(self, url):
        domain = self.get_domain(url)
        return any(domain == d or domain.endswith('.' + d) for d in self.include_domains)

    def is_pdf(self, url):
        try:
            return urlparse(url).path.lower().endswith('.pdf')
        except Exception:
            return False

    def get_entity_name_from_url(self, url):
        try:
            path = urlparse(url).path.strip('/')
            if not path:
                return self.get_domain(url)
            parts = path.split('/')
            name = parts[-1] if parts else self.get_domain(url)
            name = name.split('?')[0].split('#')[0]
            return name if name else self.get_domain(url)
        except Exception:
            return "unknown"

    def url_matches_nav_link(self, url):
        for nav_link in self.nav_links:
            if url == nav_link or nav_link in url:
                return True
        return False

    # ------------------------- Link extraction (original-compatible) -------------------------
    def is_in_navigation_container(self, element):
        current = element
        for _ in range(10):
            if current is None:
                break
            classes = current.get('class', [])
            if self.nav_container_class in classes:
                return True
            current = current.parent
        return False

    def extract_page_links(self, soup, base_url):
        links = []
        seen_keys = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            absolute_url = urljoin(base_url, href)
            absolute_url = absolute_url.split('#')[0]
            link_text = a_tag.get_text(strip=True) or "Link"
            in_nav_container = self.is_in_navigation_container(a_tag)
            matches_nav_url = self.url_matches_nav_link(absolute_url)
            is_navigation = in_nav_container and matches_nav_url
            key = (absolute_url, is_navigation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            links.append({
                'url': absolute_url,
                'link_text': link_text,
                'is_navigation': is_navigation
            })
        return links

    # ------------------------- File save (preserve original .txt behavior) -------------------------
    def create_safe_filename(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        path = urlparse(url).path.strip('/')
        if path:
            name = path.replace('/', '_').replace('\\', '_')
            name = re.sub(r'[<>:"|?*]', '', name)
            name = name[:50]
        else:
            name = 'index'
        return f"{name}_{url_hash}.txt"

    def save_content(self, url, content):
        """Save cleaned page text into output_dir/pages/*.txt (keeps your original practice)"""
        filename = self.create_safe_filename(url)
        filepath = os.path.join(self.output_dir, 'pages', filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(url)
                f.write('\n\n')
                f.write(content)
            return filepath
        except Exception as e:
            print(f"      Error saving file: {e}")
            return None

    # ------------------------- New: raw HTML + structure manifest -------------------------
    def save_raw_html(self, url, html):
        fname = self._safe_base(url) + '.html'
        fpath = os.path.join(self.output_dir, 'pages_html', fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(html)
        return fpath

    def _safe_base(self, url):
        h = hashlib.md5(url.encode()).hexdigest()[:10]
        path = urlparse(url).path.strip('/').replace('/', '_') or 'index'
        path = re.sub(r'[<>:"|?*]', '', path)[:80]
        return f"{path}_{h}"

    def normalize_url(self, url, soup=None):
        parsed = urlparse(url)
        scheme = parsed.scheme or 'https'
        netloc = parsed.netloc.lower()
        path = parsed.path or '/'
        # drop fragments; drop utm_* params; sort query
        query_params = [kv for kv in parsed.query.split('&') if kv and not kv.lower().startswith('utm_')]
        qs = "&".join(sorted(query_params))
        norm = f"{scheme}://{netloc}{path}"
        if qs:
            norm += f"?{qs}"
        if soup:
            link = soup.find('link', rel=lambda v: v and 'canonical' in v)
            if link and link.get('href'):
                try:
                    canon = urljoin(norm, link['href'])
                    return canon
                except Exception:
                    pass
        return norm

    def extract_breadcrumbs(self, soup):
        bc = soup.select('ol[itemscope][itemtype*="BreadcrumbList"] li[itemprop="itemListElement"]')
        if bc:
            out = []
            for li in bc:
                name = li.select_one('[itemprop="name"]')
                if name:
                    out.append(name.get_text(" ", strip=True))
            if out:
                return out
        nav = soup.select_one('nav[aria-label*="breadcrumb" i], nav.breadcrumb, .breadcrumb, .breadcrumbs')
        if nav:
            parts = [a.get_text(" ", strip=True) for a in nav.select('a')]
            if parts:
                return parts
        return []

    def _css_path(self, el):
        parts = []
        cur = el
        while cur and getattr(cur, 'name', None) and cur.name != 'html':
            ident = cur.name
            if cur.get('id'):
                ident += f"#{cur['id']}"
            elif cur.get('class'):
                ident += "." + ".".join(cur.get('class')[:2])
            parts.append(ident)
            cur = cur.parent
        return " > ".join(reversed(parts))

    def extract_headings_and_sections(self, soup):
        hs = []
        for h in soup.find_all(re.compile(r'^h[1-6]$')):
            level = int(h.name[1])
            text = h.get_text(" ", strip=True)
            hs.append({"el": h, "level": level, "text": text, "id": h.get('id'), "css_path": self._css_path(h)})
        sections = []
        for i, h in enumerate(hs):
            end_el = None
            for j in range(i+1, len(hs)):
                if hs[j]["level"] <= h["level"]:
                    end_el = hs[j]["el"]
                    break
            contents = []
            node = h["el"].next_sibling
            while node and node is not end_el:
                if getattr(node, 'get_text', None):
                    txt = node.get_text(" ", strip=True)
                    if txt:
                        contents.append(txt)
                node = node.next_sibling
            section_text = "\n".join(contents).strip()
            # heading_path up to this heading level
            hp = []
            for k in range(0, i+1):
                if hs[k]["level"] <= h["level"]:
                    hp.append(hs[k]["text"])
            sections.append({
                "heading": h["text"], "level": h["level"],
                "heading_path": hp, "css_path": h["css_path"], "id": h["id"],
                "text": section_text
            })
        headings = [{"level": x["level"], "text": x["text"], "id": x["id"], "css_path": x["css_path"]} for x in hs]
        return headings, sections

    def extract_anchor_windows(self, soup, base_url, window=4):
        anchors = []
        for a in soup.find_all('a', href=True):
            href = urljoin(base_url, a['href'].split('#')[0])
            anchor_text = a.get_text(" ", strip=True)
            # classify navigation via ancestry + configured patterns
            in_nav = False
            cur = a
            for _ in range(10):
                if cur is None:
                    break
                classes = cur.get('class', [])
                if self.nav_container_class in classes:
                    in_nav = True
                    break
                cur = cur.parent
            in_nav = in_nav or self.url_matches_nav_link(href)
            # choose nearest block
            block = a.find_parent(['p','li','td','dd','figcaption','h1','h2','h3','h4','h5','h6']) or a.parent
            block_text = block.get_text(" ", strip=True) if block else anchor_text
            btoks = block_text.split()
            atoks = anchor_text.split() if anchor_text else []
            pos = None
            if atoks:
                for i in range(len(btoks)-len(atoks)+1):
                    if btoks[i:i+len(atoks)] == atoks:
                        pos = i
                        break
            start = max(0, (pos or 0) - window)
            end = min(len(btoks), (pos or 0) + (len(atoks) or 1) + window)
            before = " ".join(btoks[start:(pos or 0)])
            after = " ".join(btoks[(pos or 0) + (len(atoks) or 1):end])
            anchors.append({
                "href": href,
                "anchor_text": anchor_text,
                "before": before,
                "after": after,
                "is_navigation": bool(in_nav)
            })
        return anchors

    def compute_content_hash(self, soup_for_main):
        clone = BeautifulSoup(str(soup_for_main), 'html.parser')
        for t in clone(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            t.decompose()
        text = clone.get_text(" ", strip=True)
        return "sha256:" + hashlib.sha256(text.encode('utf-8', 'ignore')).hexdigest()

    def save_structure_manifest(self, url, html):
        soup_raw = BeautifulSoup(html, 'html.parser')
        canonical = self.normalize_url(url, soup_raw)
        breadcrumbs = self.extract_breadcrumbs(soup_raw)
        headings, sections = self.extract_headings_and_sections(soup_raw)
        anchors = self.extract_anchor_windows(soup_raw, canonical)
        segs = [s for s in urlparse(canonical).path.split('/') if s]
        title = (soup_raw.title.get_text(strip=True) if soup_raw.title else "")[:200]
        content_hash = self.compute_content_hash(soup_raw)
        manifest = {
            "url": url,
            "canonical_url": canonical,
            "title": title,
            "path_segments": segs,
            "breadcrumbs": breadcrumbs,
            "headings": headings,
            "sections": sections,
            "anchors": anchors,
            "content_hash": content_hash
        }
        fpath = os.path.join(self.output_dir, 'structure', self._safe_base(canonical) + ".json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return fpath

    # ------------------------- Checkpoint / Resume -------------------------
    def save_checkpoint(self, queue):
        data = {
            "queue": list(queue),  # list of [url, depth]
            "visited": list(self.visited)
        }
        with open(os.path.join(self.output_dir, 'state', 'checkpoint.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f)
        nx.write_gpickle(self.graph, os.path.join(self.output_dir, 'state', 'graph_partial.gpickle'))

    def load_checkpoint(self):
        cpath = os.path.join(self.output_dir, 'state', 'checkpoint.json')
        gpath = os.path.join(self.output_dir, 'state', 'graph_partial.gpickle')
        if not os.path.exists(cpath):
            return None
        with open(cpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if os.path.exists(gpath):
            self.graph = nx.read_gpickle(gpath)
        self.visited = set(data.get("visited", []))
        queue = deque([tuple(x) for x in data.get("queue", [])])
        print("↻ Resuming from checkpoint:", len(self.visited), "visited,", len(queue), "in queue")
        return queue

    # ------------------------- Crawl -------------------------
    def should_crawl(self, url, current_depth):
        if self.is_pdf(url):
            return False
        if self.max_depth is not None and current_depth > self.max_depth:
            return False
        if self.is_main_domain(url):
            return True
        if self.is_include_domain(url):
            if url not in self.domain_depths:
                self.domain_depths[url] = current_depth
            return self.domain_depths[url] <= 2
        return False

    def crawl(self):
        print("="*80)
        print("KNOWLEDGE GRAPH CRAWLER")
        print("="*80)
        depth_info = f"max depth {self.max_depth}" if self.max_depth is not None else "unlimited depth"
        print(f"Main domains ({depth_info}): {', '.join(self.main_domains)}")
        if self.include_domains:
            print(f"Include domains (depth ≤ 2): {', '.join(self.include_domains)}")
        print(f"Navigation container: .{self.nav_container_class}")
        print(f"Navigation links: {len(self.nav_links)} provided")
        print("="*80 + "\n")

        if not self.setup_driver():
            return False

        queue = deque([(self.start_url, 0)])
        # Try to resume if a checkpoint exists
        resumed = self.load_checkpoint()
        if resumed is not None:
            queue = resumed

        try:
            pages_since_checkpoint = 0
            while queue:
                url, depth = queue.popleft()
                if url in self.visited:
                    continue

                if not self.should_crawl(url, depth):
                    entity_name = self.get_entity_name_from_url(url)
                    self.graph.add_node(
                        url,
                        depth=depth,
                        label=entity_name,
                        is_external=True,
                        is_pdf=self.is_pdf(url)
                    )
                    continue

                domain_type = "MAIN" if self.is_main_domain(url) else "INCLUDE"
                print(f"\n{'='*80}")
                print(f"[{domain_type}] Depth {depth}: {url}")
                print(f"{'='*80}")

                self.visited.add(url)
                self.graph.add_node(
                    url,
                    depth=depth,
                    label=self.get_entity_name_from_url(url),
                    is_external=False,
                    is_pdf=False
                )

                try:
                    print("  → Loading page...")
                    self.driver.get(url)
                    self.wait_for_page_load()

                    # Basic presence check
                    try:
                        page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    except Exception:
                        page_text = ""
                    if len(page_text.strip()) < 50 or "loading" in page_text.lower()[:120]:
                        print("  → Waiting for content to fully load...")
                        time.sleep(4)

                    html = self.driver.page_source
                    # Save raw HTML early (so we have it even if later parsing fails)
                    self.save_raw_html(url, html)

                    soup = BeautifulSoup(html, 'html.parser')

                    # Build and save per-page structure manifest (headings/breadcrumbs/anchors/paths)
                    try:
                        self.save_structure_manifest(url, html)
                    except Exception as e:
                        print(f"  ! Structure manifest warning: {e}")

                    # Extract cleaned text (remove boilerplate)
                    soup_clean = BeautifulSoup(html, 'html.parser')
                    for tag in soup_clean(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                        tag.decompose()
                    text = soup_clean.get_text(separator='\n', strip=True)
                    text = re.sub(r'\n\s*\n', '\n\n', text)

                    if len(text) < 100 or "loading" in text.lower()[:200]:
                        print("  → Content appears incomplete, waiting more...")
                        time.sleep(4)
                        html = self.driver.page_source
                        soup_clean = BeautifulSoup(html, 'html.parser')
                        for tag in soup_clean(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                            tag.decompose()
                        text = soup_clean.get_text(separator='\n', strip=True)
                        text = re.sub(r'\n\s*\n', '\n\n', text)

                    filepath = self.save_content(url, text)
                    if filepath:
                        self.page_contents[url] = filepath
                        print(f"  ✓ Content saved ({len(text)} chars)")

                    print("  → Extracting links...")
                    links = self.extract_page_links(BeautifulSoup(html, 'html.parser'), url)
                    nav_count = sum(1 for l in links if l['is_navigation'])
                    content_count = len(links) - nav_count
                    print(f"  ✓ Found {len(links)} links ({nav_count} navigation, {content_count} content)")

                    queued = 0
                    pdf_count = 0
                    external_count = 0

                    for link in links:
                        target_url = link['url']
                        link_text = link['link_text']
                        is_nav = link['is_navigation']
                        relationship = "back_to_page" if is_nav else link_text
                        self.graph.add_edge(url, target_url, relationship=relationship)

                        if self.is_pdf(target_url):
                            entity_name = self.get_entity_name_from_url(target_url)
                            self.graph.add_node(
                                target_url,
                                label=entity_name,
                                is_external=True,
                                is_pdf=True
                            )
                            pdf_count += 1
                        elif self.is_main_domain(target_url):
                            if target_url not in self.visited:
                                if self.max_depth is None or depth < self.max_depth:
                                    queue.append((target_url, depth + 1))
                                    queued += 1
                                else:
                                    entity_name = self.get_entity_name_from_url(target_url)
                                    self.graph.add_node(
                                        target_url,
                                        label=entity_name,
                                        is_external=True,
                                        is_pdf=False
                                    )
                                    external_count += 1
                        elif self.is_include_domain(target_url):
                            if depth < 2 and target_url not in self.visited:
                                queue.append((target_url, depth + 1))
                                queued += 1
                            else:
                                entity_name = self.get_entity_name_from_url(target_url)
                                self.graph.add_node(
                                    target_url,
                                    label=entity_name,
                                    is_external=True,
                                    is_pdf=False
                                )
                                external_count += 1
                        else:
                            entity_name = self.get_entity_name_from_url(target_url)
                            self.graph.add_node(
                                target_url,
                                label=entity_name,
                                is_external=True,
                                is_pdf=False
                            )
                            external_count += 1

                    print(f"  ✓ Queued: {queued} new URLs to crawl")
                    print(f"  ✓ External: {external_count} entities, {pdf_count} PDFs")
                    print(f"  → Progress: Queue={len(queue)}, Visited={len(self.visited)}")

                    pages_since_checkpoint += 1
                    if pages_since_checkpoint >= 25:
                        self.save_checkpoint(queue)
                        pages_since_checkpoint = 0

                except Exception as e:
                    print(f"  ✗ Page error: {e}")
                    # Save checkpoint and continue
                    self.save_checkpoint(queue)
                    time.sleep(1)
                    continue

                time.sleep(0.8)  # politeness

        finally:
            if self.driver:
                self.driver.quit()
                print("\n✓ Browser closed")
            # final checkpoint
            try:
                self.save_checkpoint(deque())
            except Exception:
                pass

        print(f"\n{'='*80}")
        print("CRAWL COMPLETE")
        print(f"{'='*80}")
        print(f"Pages crawled: {len(self.visited)}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        pdf_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('is_pdf', False)]
        external_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('is_external', False)]
        print(f"PDF documents: {len(pdf_nodes)}")
        print(f"External entities: {len(external_nodes)}")
        print(f"{'='*80}\n")
        return True

    # ------------------------- Save graph JSON & visualization -------------------------
    def save_graph_json(self):
        data = {
            'metadata': {
                'start_url': self.start_url,
                'main_domains': self.main_domains,
                'include_domains': self.include_domains,
                'max_depth': self.max_depth,
                'nav_container_class': self.nav_container_class,
                'nav_links_count': len(self.nav_links),
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'crawled_pages': len(self.visited)
            },
            'nodes': [],
            'edges': []
        }
        for node, attrs in self.graph.nodes(data=True):
            data['nodes'].append({
                'id': node,
                'url': node,
                'label': attrs.get('label', self.get_entity_name_from_url(node)),
                'depth': attrs.get('depth', 0),
                'is_external': attrs.get('is_external', False),
                'is_pdf': attrs.get('is_pdf', False)
            })
        for src, tgt, attrs in self.graph.edges(data=True):
            data['edges'].append({
                'from': src,
                'to': tgt,
                'label': attrs.get('relationship', '')
            })
        filepath = os.path.join(self.output_dir, 'graph.json')
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            file_size = os.path.getsize(filepath)
            print(f"✓ Graph JSON: {filepath}")
            print(f"  File size: {file_size:,} bytes")
            return data
        except Exception as e:
            print(f"✗ Error saving JSON: {e}")
            return None

    def create_interactive_graph(self, graph_data):
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            print("✗ Error: Invalid graph data for visualization")
            return False
        print(f"  → Creating interactive graph...")
        print(f"    Nodes: {len(graph_data['nodes'])}")
        print(f"    Edges: {len(graph_data['edges'])}")
        try:
            json_data = json.dumps(graph_data, ensure_ascii=False)
        except Exception as e:
            print(f"✗ Error serializing data: {e}")
            return False

        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph Visualization</title>
    <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: white; overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
        .header h1 { margin: 0; font-size: 28px; font-weight: 600; }
        .stats { background: #16213e; padding: 15px 20px; display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .stat { text-align: center; }
        .stat .value { font-size: 28px; font-weight: bold; color: #667eea; }
        .stat .label { font-size: 12px; color: #aaa; margin-top: 3px; }
        .controls { background: #16213e; padding: 15px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; border-bottom: 1px solid #0f3460; }
        .controls input, .controls select, .controls button { padding: 10px 15px; border-radius: 5px; border: none; font-size: 14px; }
        .controls input { flex: 1; max-width: 300px; background: white; }
        .controls button { background: #667eea; color: white; cursor: pointer; font-weight: 600; transition: all 0.3s; }
        .controls button:hover { background: #5568d3; transform: translateY(-1px); }
        #network { height: calc(100vh - 220px); background: #0f3460; }
        .info { position: fixed; top: 200px; right: 20px; background: rgba(22,33,62,0.98); padding: 20px; border-radius: 10px; max-width: 350px; display: none; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-height: 60vh; overflow-y: auto; z-index: 1000; border: 1px solid #667eea; }
        .info h3 { color: #667eea; margin-bottom: 10px; font-size: 18px; font-weight: 600; }
        .info p { margin: 8px 0; font-size: 13px; line-height: 1.6; }
        .info a { color: #667eea; text-decoration: none; word-break: break-all; }
        .info a:hover { text-decoration: underline; }
        .close { float: right; cursor: pointer; font-size: 20px; color: #aaa; line-height: 1; font-weight: bold; }
        .close:hover { color: white; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; margin: 3px 3px 3px 0; font-weight: 600; }
        .badge-nav { background: #3498db; color: white; }
        .badge-external { background: #e74c3c; color: white; }
        .badge-pdf { background: #e67e22; color: white; }
        .badge-crawled { background: #27ae60; color: white; }
        .legend { position: fixed; bottom: 20px; left: 20px; background: rgba(22,33,62,0.98); padding: 15px; border-radius: 10px; font-size: 12px; z-index: 1000; border: 1px solid #667eea; }
        .legend strong { display: block; margin-bottom: 10px; color: #667eea; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .legend-line { width: 30px; height: 2px; margin-right: 10px; }
        .loading { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(22,33,62,0.98); padding: 30px 50px; border-radius: 10px; display: none; z-index: 2000; font-size: 18px; border: 2px solid #667eea; }
    </style>
</head>
<body>
    <div class="loading" id="loading">⏳ Building graph...</div>

    <div class="header">
        <h1>🕸️ Knowledge Graph Visualization</h1>
    </div>

    <div class="stats">
        <div class="stat"><div class="value" id="nodes">0</div><div class="label">Total Nodes</div></div>
        <div class="stat"><div class="value" id="edges">0</div><div class="label">Total Edges</div></div>
        <div class="stat"><div class="value" id="crawled">0</div><div class="label">Pages Crawled</div></div>
        <div class="stat"><div class="value" id="pdfs">0</div><div class="label">PDF Documents</div></div>
        <div class="stat"><div class="value" id="external">0</div><div class="label">External Entities</div></div>
    </div>

    <div class="controls">
        <input type="text" id="search" placeholder="🔍 Search nodes by name or URL..." />
        <select id="filter">
            <option value="">All Nodes</option>
            <option value="crawled">Crawled Pages Only</option>
            <option value="external">External Only</option>
            <option value="pdf">PDFs Only</option>
        </select>
        <button onclick="resetView()">Reset View</button>
        <button onclick="fitNetwork()">Fit to Screen</button>
    </div>

    <div id="network"></div>

    <div class="info" id="info">
        <span class="close" onclick="closeInfo()">×</span>
        <div id="info-content"></div>
    </div>

    <div class="legend">
        <strong>Legend</strong>
        <div class="legend-item"><div class="legend-color" style="background: #667eea;"></div>Crawled Page</div>
        <div class="legend-item"><div class="legend-color" style="background: #e67e22;"></div>PDF Document</div>
        <div class="legend-item"><div class="legend-color" style="background: #e74c3c;"></div>External Link</div>
        <div class="legend-item"><div class="legend-line" style="background: #3498db; height: 3px; border-top: 2px dashed #3498db;"></div>Navigation</div>
        <div class="legend-item"><div class="legend-line" style="background: #666;"></div>Content Link</div>
    </div>

    <script>
        document.getElementById('loading').style.display = 'block';
        try {
            const data = __GRAPH_DATA__;
            if (!data || !data.nodes || !data.edges) {
                throw new Error('Invalid graph data structure');
            }
            document.getElementById('nodes').textContent = data.nodes.length;
            document.getElementById('edges').textContent = data.edges.length;
            document.getElementById('crawled').textContent = data.metadata.crawled_pages;
            const pdfCount = data.nodes.filter(n => n.is_pdf).length;
            const extCount = data.nodes.filter(n => n.is_external && !n.is_pdf).length;
            document.getElementById('pdfs').textContent = pdfCount;
            document.getElementById('external').textContent = extCount;

            const nodes = data.nodes.map(n => ({
                id: n.id,
                label: n.label || 'Unknown',
                title: n.url,
                color: n.is_pdf ? '#e67e22' : (n.is_external ? '#e74c3c' : '#667eea'),
                shape: n.is_external ? 'box' : 'dot',
                size: n.is_external ? 15 : 25,
                font: { color: 'white', size: 12, face: 'Arial' },
                ...n
            }));
            const edges = data.edges.map(e => ({
                from: e.from,
                to: e.to,
                label: e.label || '',
                arrows: 'to',
                color: { color: e.label === 'back_to_page' ? '#3498db' : '#666' },
                font: { size: 9, color: '#aaa', strokeWidth: 0 },
                dashes: e.label === 'back_to_page',
                width: e.label === 'back_to_page' ? 2 : 1
            }));

            const container = document.getElementById('network');
            const network = new vis.Network(
                container,
                { nodes: nodes, edges: edges },
                {
                    physics: {
                        enabled: true,
                        barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3, springLength: 150, springConstant: 0.04 },
                        stabilization: { iterations: 200 }
                    },
                    interaction: { hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }
                }
            );

            network.once('stabilizationIterationsDone', function() {
                document.getElementById('loading').style.display = 'none';
            });

            network.on('click', params => {
                if (params.nodes.length > 0) {
                    const node = data.nodes.find(n => n.id === params.nodes[0]);
                    if (node) showNodeInfo(node);
                }
            });

            function showNodeInfo(node) {
                const outEdges = data.edges.filter(e => e.from === node.id);
                const inEdges = data.edges.filter(e => e.to === node.id);
                const navEdges = outEdges.filter(e => e.label === 'back_to_page');
                let badges = '';
                if (!node.is_external) badges += '<span class="badge badge-crawled">CRAWLED</span>';
                if (node.is_external) badges += '<span class="badge badge-external">EXTERNAL</span>';
                if (node.is_pdf) badges += '<span class="badge badge-pdf">PDF</span>';
                if (navEdges.length > 0) badges += '<span class="badge badge-nav">' + navEdges.length + ' NAV</span>';
                let html = `
                    <h3>${node.label}</h3>
                    ${badges}
                    <p><strong>URL:</strong><br><a href="${node.url}" target="_blank">${node.url}</a></p>
                    <p><strong>Depth:</strong> ${node.depth}</p>
                    <p><strong>Outgoing:</strong> ${outEdges.length} (${navEdges.length} nav)</p>
                    <p><strong>Incoming:</strong> ${inEdges.length}</p>
                `;
                if (outEdges.length > 0 && outEdges.length <= 10) {
                    html += '<p><strong>Links:</strong></p><ul style="margin-left: 20px; font-size: 12px;">';
                    outEdges.forEach(e => {
                        const target = data.nodes.find(n => n.id === e.to);
                        const label = e.label === 'back_to_page' ? '[NAV]' : (e.label || '').substring(0, 40);
                        html += `<li>${target ? target.label : '?'} (${label})</li>`;
                    });
                    html += '</ul>';
                }
                document.getElementById('info-content').innerHTML = html;
                document.getElementById('info').style.display = 'block';
            }

            function closeInfo() { document.getElementById('info').style.display = 'none'; }
            document.getElementById('search').addEventListener('input', e => {
                const term = e.target.value.toLowerCase();
                if (!term) { network.setData({ nodes, edges }); return; }
                const filtered = nodes.filter(n => n.label.toLowerCase().includes(term) || n.url.toLowerCase().includes(term));
                const ids = filtered.map(n => n.id);
                const filteredEdges = edges.filter(e => ids.includes(e.from) && ids.includes(e.to));
                network.setData({ nodes: filtered, edges: filteredEdges });
            });
            document.getElementById('filter').addEventListener('change', e => {
                let filtered = nodes;
                if (e.target.value === 'crawled') filtered = nodes.filter(n => !n.is_external);
                if (e.target.value === 'external') filtered = nodes.filter(n => n.is_external && !n.is_pdf);
                if (e.target.value === 'pdf') filtered = nodes.filter(n => n.is_pdf);
                const ids = filtered.map(n => n.id);
                const filteredEdges = edges.filter(e => ids.includes(e.from) && ids.includes(e.to));
                network.setData({ nodes: filtered, edges: filteredEdges });
            });
            function resetView() { network.setData({ nodes, edges }); document.getElementById('search').value=''; document.getElementById('filter').value=''; closeInfo(); }
            function fitNetwork() { network.fit(); }
            window.resetView = resetView; window.fitNetwork = fitNetwork; window.closeInfo = closeInfo;
        } catch (error) {
            document.getElementById('loading').innerHTML = '❌ Error: ' + error.message;
            console.error('Error:', error);
        }
    </script>
</body>
</html>"""
        html = html_template.replace('__GRAPH_DATA__', json_data)
        filepath = os.path.join(self.output_dir, 'interactive_graph.html')
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            file_size = os.path.getsize(filepath)
            if file_size > 1000:
                print(f"✓ Interactive graph: {filepath}")
                print(f"  File size: {file_size:,} bytes")
                print(f"  → Open this file in your browser!")
                return True
            else:
                print(f"✗ Warning: File size is only {file_size} bytes")
                return False
        except Exception as e:
            print(f"✗ Error saving HTML: {e}")
            return False

    # ------------------------- Rebuild/Refine graph from structure (no crawl) -------------------------
    def rebuild_graph_from_structure(self):
        """Rebuild self.graph only from structure/*.json (fast refinement)."""
        struct_dir = os.path.join(self.output_dir, 'structure')
        if not os.path.isdir(struct_dir):
            print("✗ No structure directory found to rebuild from.")
            return False
        files = [os.path.join(struct_dir, f) for f in os.listdir(struct_dir) if f.endswith('.json')]
        if not files:
            print("✗ No structure manifests found.")
            return False

        G = nx.DiGraph()
        # First pass: nodes
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                m = json.load(f)
            url = m.get("canonical_url") or m.get("url")
            if not url:
                continue
            depth = len(m.get("path_segments", []))
            label = (m.get("path_segments") or [None])[-1]
            if not label:
                parsed = urlparse(url)
                label = parsed.path.strip('/') or parsed.netloc
            G.add_node(url,
                       depth=depth,
                       label=label,
                       is_external=False,
                       is_pdf=urlparse(url).path.lower().endswith('.pdf'))
        # Second pass: edges from anchors
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                m = json.load(f)
            src = m.get("canonical_url") or m.get("url")
            if not src:
                continue
            for a in m.get("anchors", []):
                tgt = self.normalize_url(a.get("href", ""))
                if not tgt:
                    continue
                rel = "back_to_page" if a.get("is_navigation") else (a.get("anchor_text") or "")
                rel_text = (a.get("before", "") + " " + (a.get("anchor_text") or "") + " " + a.get("after", "")).strip()
                G.add_edge(src, tgt, relationship=rel or rel_text)
        self.graph = G
        print("✓ Rebuilt graph from structure:", self.graph.number_of_nodes(), "nodes,", self.graph.number_of_edges(), "edges")
        return True


# ------------------------- CLI main (unchanged prompts) -------------------------
def main():
    """Main function to run the crawler (prompts unchanged)."""

    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH CRAWLER")
    print("="*80)
    print("Complete implementation with:")
    print("  • Configurable max depth")
    print("  • Explicit navigation control")
    print("  • PDF detection by extension")
    print("  • Interactive visualization")
    print("  • Raw HTML + structure manifests • Checkpoint/Resume • Fast refine")
    print("="*80 + "\n")

    # Get starting URL
    start_url = input("Enter starting URL: ").strip()
    if not start_url:
        print("❌ Error: Starting URL required")
        return

    # Get main domains
    print("\nMain domains to crawl (comma-separated):")
    print("  Example: example.com,www.example.com")
    main_domains = input("Main domains: ").strip()
    main_domains = [d.strip() for d in main_domains.split(',') if d.strip()]

    if not main_domains:
        print("❌ Error: At least one main domain required")
        return

    # Get include domains
    print("\nInclude domains to crawl up to depth 2 (optional, comma-separated):")
    include_domains = input("Include domains: ").strip()
    include_domains = [d.strip() for d in include_domains.split(',') if d.strip()] if include_domains else []

    # Get max depth
    print("\nMaximum crawl depth (leave empty for unlimited):")
    print("  Example: 3, 4, 5")
    max_depth_input = input("Max depth: ").strip()
    max_depth = int(max_depth_input) if max_depth_input else None

    # Get navigation configuration
    print("\nNavigation container CSS class:")
    print("  This is the CSS class that contains your navigation menu")
    print("  Example: page-header, navbar, navigation")
    nav_container = input("Nav container class (default: page-header): ").strip()
    nav_container = nav_container if nav_container else "page-header"

    print("\nNavigation links (comma-separated URLs or paths):")
    print("  Example: /home,/about,/products,/contact")
    print("  These will use 'back_to_page' relationship")
    nav_links_input = input("Nav links: ").strip()
    nav_links = [link.strip() for link in nav_links_input.split(',') if link.strip()] if nav_links_input else []

    # Show configuration summary
    print(f"\n{'='*80}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"Start URL: {start_url}")
    print(f"Max depth: {max_depth if max_depth else 'unlimited'}")
    print(f"Main domains ({len(main_domains)}): {', '.join(main_domains)}")
    if include_domains:
        print(f"Include domains ({len(include_domains)}): {', '.join(include_domains)}")
    print(f"Nav container: .{nav_container}")
    if nav_links:
        print(f"Nav links ({len(nav_links)}):")
        for link in nav_links[:5]:
            print(f"  • {link}")
        if len(nav_links) > 5:
            print(f"  ... and {len(nav_links) - 5} more")
    else:
        print("Nav links: None")
    print(f"{'='*80}\n")

    # Confirm
    proceed = input("Start crawling? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Cancelled")
        return

    # Create crawler
    print("\nInitializing crawler...\n")
    crawler = KnowledgeGraphCrawler(
        start_url=start_url,
        main_domains=main_domains,
        include_domains=include_domains,
        nav_links=nav_links,
        max_depth=max_depth,
        nav_container_class=nav_container
    )

    # Run crawler
    success = crawler.crawl()
    if not success:
        print("❌ Crawling failed")
        return

    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80 + "\n")

    graph_data = crawler.save_graph_json()
    if graph_data:
        crawler.create_interactive_graph(graph_data)

    # Final summary
    print("\n" + "="*80)
    print("✅ ALL COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {crawler.output_dir}/")
    print("\nFiles created:")
    print(f"  📊 interactive_graph.html  ← Open in browser!")
    print(f"  📄 graph.json              ← Complete graph data")
    print(f"  📁 pages/*.txt             ← Page contents (with URL on first line)")
    print(f"  📁 pages_html/*.html       ← Raw HTML (for structure)")
    print(f"  📁 structure/*.json        ← Per-page manifests (headings/breadcrumbs/anchors)")
    print(f"  📁 state/*                 ← Checkpoints (resume supported)")
    print("\nStatistics:")
    print(f"  • Pages crawled: {len(crawler.visited)}")
    print(f"  • Total nodes: {crawler.graph.number_of_nodes()}")
    print(f"  • Total edges: {crawler.graph.number_of_edges()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
