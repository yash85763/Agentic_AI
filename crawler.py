# # build_site_pdf_graph.py
# # Create a graph-lite knowledge graph combining Website (Pages/Sections) and PDFs (Doc/Page)
# # Outputs: nodes.jsonl, edges.jsonl, chunks.jsonl, nodes.csv, edges.csv, site.graphml
# # Usage:
# #   pip install requests beautifulsoup4 lxml networkx pypdf
# #   python build_site_pdf_graph.py --base-url https://example.com --pdf-dir ./pdfs --max-pages 1000 --outdir ./out_graph

# import argparse
# import hashlib
# import json
# import queue
# import re
# import sys
# import time
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Dict, List, Optional, Set, Tuple
# from urllib.parse import urljoin, urldefrag, urlparse

# import networkx as nx
# import requests
# from bs4 import BeautifulSoup, Tag
# from requests.exceptions import RequestException
# from urllib import robotparser
# from pypdf import PdfReader

# UA = "GraphLiteCrawler/1.0 (+contact:webmaster@example.com)"

# # ===================== Helpers =====================

# def sha1(text: str) -> str:
#     return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# def normalize_url(href: str, base_url: str) -> Optional[str]:
#     if not href:
#         return None
#     absu = urljoin(base_url, href)
#     absu, _ = urldefrag(absu)
#     p = urlparse(absu)
#     if p.scheme not in ("http", "https"):
#         return None
#     return absu


# def same_site(url: str, base_netloc: str, include_subdomains: bool) -> bool:
#     nl = urlparse(url).netloc.lower()
#     base = base_netloc.lower()
#     if nl == base:
#         return True
#     if include_subdomains and nl.endswith("." + base):
#         return True
#     return False


# def position_of(a_tag: Tag) -> str:
#     anc_names = [p.name for p in a_tag.parents if isinstance(p, Tag)]
#     classes = " ".join(a_tag.get("class", []))
#     if "nav" in anc_names or "navbar" in classes:
#         return "nav"
#     if "header" in anc_names:
#         return "header"
#     if "footer" in anc_names:
#         return "footer"
#     if "aside" in anc_names:
#         return "aside"
#     return "body"


# def clean_text(soup: BeautifulSoup) -> str:
#     for tag in soup(["script", "style", "noscript", "template"]):
#         tag.decompose()
#     text = soup.get_text("\n", strip=True)
#     text = re.sub(r"\n{2,}", "\n", text)
#     return text


# # ===================== Data Classes =====================

# @dataclass
# class LinkEdge:
#     href: str
#     anchor_text: str
#     position: str  # nav|header|footer|body|aside


# @dataclass
# class Section:
#     section_id: str
#     level: int
#     heading_text: str
#     heading_path: List[str]
#     css_id: Optional[str]
#     text: str  # aggregated text under this section


# @dataclass
# class PageData:
#     url: str
#     title: str
#     text: str
#     headings: List[Section]
#     links: List[LinkEdge]
#     discovered_parent: Optional[str]
#     content_hash: str


# # ===================== HTML Parsing =====================

# HEADING_RE = re.compile(r"^h([1-6])$", re.I)


# def extract_sections_with_text(soup: BeautifulSoup, page_url: str) -> List[Section]:
#     # Collect headings in DOM order
#     headings_nodes: List[Tuple[int, Tag]] = []
#     for tag in soup.find_all(HEADING_RE):
#         try:
#             lvl = int(tag.name[1])
#         except Exception:
#             continue
#         headings_nodes.append((lvl, tag))

#     sections: List[Section] = []
#     path: List[str] = []
#     last_level = 0

#     for i, (level, htag) in enumerate(headings_nodes):
#         text = re.sub(r"\s+", " ", htag.get_text(" ", strip=True))
#         if not text:
#             text = f"Heading {level}"
#         if level > last_level:
#             path.append(text)
#         else:
#             while len(path) >= level:
#                 if path:
#                     path.pop()
#             path.append(text)
#         last_level = level

#         # determine section content until the next heading of same or higher level
#         content_parts: List[str] = []
#         node = htag.next_sibling
#         while node:
#             if isinstance(node, Tag) and HEADING_RE.match(node.name or ""):
#                 # next heading: stop
#                 nxt_level = int(node.name[1])
#                 if nxt_level <= level:
#                     break
#             if isinstance(node, Tag):
#                 # skip scripts/styles
#                 if node.name in ("script", "style", "noscript", "template"):
#                     node = node.next_sibling
#                     continue
#                 content_parts.append(node.get_text(" ", strip=True))
#             else:
#                 content_parts.append(str(node).strip())
#             node = node.next_sibling

#         css_id = htag.get("id")
#         frag = css_id or f"h{level}-" + sha1(text)[:8]
#         section_id = urlparse(page_url).path + "#" + frag
#         section_text = re.sub(r"\s+", " ", " ".join([p for p in content_parts if p]))
#         sections.append(
#             Section(
#                 section_id=section_id,
#                 level=level,
#                 heading_text=text,
#                 heading_path=path.copy(),
#                 css_id=css_id,
#                 text=section_text,
#             )
#         )

#     return sections


# # ===================== Crawler =====================

# class SiteCrawler:
#     def __init__(self, base_url: str, max_pages: int = 1000, include_subdomains: bool = False, delay: float = 0.3):
#         self.base_url = base_url.rstrip('/')
#         self.max_pages = max_pages
#         self.include_subdomains = include_subdomains
#         self.delay = delay
#         self.base_netloc = urlparse(self.base_url).netloc
#         self.robot = robotparser.RobotFileParser()
#         try:
#             self.robot.set_url(urljoin(self.base_url, "/robots.txt"))
#             self.robot.read()
#         except Exception:
#             pass

#     def allowed(self, url: str) -> bool:
#         try:
#             return self.robot.can_fetch(UA, url)
#         except Exception:
#             return True

#     def fetch_html(self, url: str) -> Optional[str]:
#         try:
#             r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
#             if r.status_code >= 400:
#                 return None
#             return r.text
#         except RequestException:
#             return None

#     def crawl(self) -> Tuple[Dict[str, PageData], List[Tuple[str, str, LinkEdge]]]:
#         pages: Dict[str, PageData] = {}
#         link_edges: List[Tuple[str, str, LinkEdge]] = []
#         seen: Set[str] = set()
#         q = queue.Queue()
#         q.put((self.base_url, None))

#         while not q.empty() and len(pages) < self.max_pages:
#             url, parent = q.get()
#             if url in seen:
#                 continue
#             seen.add(url)
#             if not same_site(url, self.base_netloc, self.include_subdomains):
#                 continue
#             if not self.allowed(url):
#                 print(f"[robots] skip {url}", file=sys.stderr)
#                 continue

#             html = self.fetch_html(url)
#             if html is None:
#                 continue
#             soup = BeautifulSoup(html, "lxml")

#             title = soup.title.string.strip() if soup.title and soup.title.string else ""
#             sections = extract_sections_with_text(soup, url)
#             text = clean_text(soup)
#             chash = sha1(url + "|" + text[:4096])

#             # links
#             raw_links: List[LinkEdge] = []
#             for a in soup.find_all("a", href=True):
#                 href = normalize_url(a.get("href"), url)
#                 if not href:
#                     continue
#                 raw_links.append(
#                     LinkEdge(
#                         href=href,
#                         anchor_text=a.get_text(" ", strip=True)[:160],
#                         position=position_of(a),
#                     )
#                 )

#             pages[url] = PageData(
#                 url=url,
#                 title=title,
#                 text=text,
#                 headings=sections,
#                 links=raw_links,
#                 discovered_parent=parent,
#                 content_hash=chash,
#             )

#             for lnk in raw_links:
#                 if same_site(lnk.href, self.base_netloc, self.include_subdomains) and lnk.href not in seen:
#                     q.put((lnk.href, url))
#                     link_edges.append((url, lnk.href, lnk))

#             time.sleep(self.delay)

#         return pages, link_edges


# # ===================== PDF Ingestion (graph-lite) =====================

# @dataclass
# class PdfPageInfo:
#     file_id: str
#     page_no: int
#     text: str


# def ingest_pdfs(pdf_dir: Path) -> Tuple[Dict[str, str], Dict[str, List[PdfPageInfo]]]:
#     """
#     Returns:
#       pdf_docs: {file_id -> filename}
#       pdf_pages: {file_id -> [PdfPageInfo,...]}
#     """
#     pdf_docs: Dict[str, str] = {}
#     pdf_pages: Dict[str, List[PdfPageInfo]] = {}
#     for pdf_path in pdf_dir.glob("**/*.pdf"):
#         try:
#             reader = PdfReader(str(pdf_path))
#         except Exception:
#             continue
#         file_id = sha1(str(pdf_path.resolve()))
#         pdf_docs[file_id] = pdf_path.name
#         pages_list: List[PdfPageInfo] = []
#         for i, page in enumerate(reader.pages, start=1):
#             try:
#                 txt = page.extract_text() or ""
#                 txt = re.sub(r"\s+", " ", txt)
#             except Exception:
#                 txt = ""
#             pages_list.append(PdfPageInfo(file_id=file_id, page_no=i, text=txt))
#         pdf_pages[file_id] = pages_list
#     return pdf_docs, pdf_pages


# # ===================== Graph Builder =====================

# class GraphLiteBuilder:
#     def __init__(self):
#         self.G = nx.MultiDiGraph()
#         self.chunks: List[dict] = []  # vector DB ingestion records

#     def add_page(self, pd: PageData):
#         pid = f"page:{pd.url}"
#         self.G.add_node(pid, kind="page", url=pd.url, title=pd.title, content_hash=pd.content_hash)
#         # Sections
#         stack: List[Tuple[int, str]] = []  # (level, node_id)
#         for sec in pd.headings:
#             sid = f"section:{sec.section_id}"
#             self.G.add_node(
#                 sid,
#                 kind="section",
#                 url=pd.url,
#                 section_id=sec.section_id,
#                 level=sec.level,
#                 heading_text=sec.heading_text,
#                 heading_path="|".join(sec.heading_path),
#             )
#             self.G.add_edge(pid, sid, rel="contains")

#             # parent_of
#             while stack and stack[-1][0] >= sec.level:
#                 stack.pop()
#             if stack:
#                 self.G.add_edge(stack[-1][1], sid, rel="parent_of")
#             stack.append((sec.level, sid))

#             # Create a chunk for this section (graph-lite)
#             if sec.text:
#                 chk_id = f"web:{sha1(pd.url + '|' + sec.section_id)}"
#                 self.chunks.append(
#                     {
#                         "chunk_id": chk_id,
#                         "source_type": "html",
#                         "url": pd.url,
#                         "section_id": sec.section_id,
#                         "heading_path": sec.heading_path,
#                         "text": sec.text,
#                     }
#                 )

#     def add_link(self, src_url: str, dst_url: str, lnk: LinkEdge):
#         self.G.add_edge(f"page:{src_url}", f"page:{dst_url}", rel="links_to", anchor_text=lnk.anchor_text, position=lnk.position)

#     def add_pdf_doc(self, file_id: str, filename: str):
#         did = f"pdfdoc:{file_id}"
#         self.G.add_node(did, kind="pdfdoc", file_id=file_id, filename=filename)
#         return did

#     def add_pdf_page(self, file_id: str, page_no: int, text: str):
#         did = f"pdfdoc:{file_id}"
#         pid = f"pdfpage:{file_id}:{page_no}"
#         self.G.add_node(pid, kind="pdfpage", file_id=file_id, page_no=page_no)
#         self.G.add_edge(did, pid, rel="contains")
#         # Create a page-level chunk
#         if text:
#             chk_id = f"pdf:{sha1(file_id + ':' + str(page_no))}"
#             self.chunks.append(
#                 {
#                     "chunk_id": chk_id,
#                     "source_type": "pdf",
#                     "pdf": {"file_id": file_id, "page": page_no},
#                     "text": text,
#                 }
#             )
#         return pid

#     def add_section_pdf_reference_edges(self, pages: Dict[str, PageData], pdf_url_map: Dict[str, Tuple[str, Optional[int]]]):
#         """
#         pdf_url_map: maps absolute PDF URLs to (file_id, page_opt)
#         Adds references(section -> pdfpage) edges when a section's page links to a PDF.
#         """
#         for url, pd in pages.items():
#             # For each link in this page, if it's a PDF, connect from all sections on this page to that PDF page/doc
#             for lnk in pd.links:
#                 if lnk.href in pdf_url_map:
#                     file_id, pno = pdf_url_map[lnk.href]
#                     if pno is None:
#                         # if page unknown, reference the first page node as entry point
#                         target = f"pdfpage:{file_id}:1"
#                     else:
#                         target = f"pdfpage:{file_id}:{pno}"
#                     for sec in pd.headings:
#                         self.G.add_edge(f"section:{sec.section_id}", target, rel="references", anchor_text=lnk.anchor_text)

#     def export(self, outdir: Path):
#         outdir.mkdir(parents=True, exist_ok=True)
#         # JSONL nodes/edges
#         with (outdir / "nodes.jsonl").open("w", encoding="utf-8") as nf:
#             for n, data in self.G.nodes(data=True):
#                 rec = {"id": n}
#                 rec.update(data)
#                 nf.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         with (outdir / "edges.jsonl").open("w", encoding="utf-8") as ef:
#             for u, v, data in self.G.edges(data=True):
#                 rec = {"src": u, "dst": v, "rel": data.get("rel")}
#                 for k, val in data.items():
#                     if k != "rel":
#                         rec[k] = val
#                 ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         # CSV (Neo4j-friendly generic)
#         import csv
#         with (outdir / "nodes.csv").open("w", newline="", encoding="utf-8") as cf:
#             w = csv.writer(cf)
#             w.writerow(["id","kind","url","title","section_id","level","heading_text","heading_path","file_id","filename","page_no"])
#             for n, d in self.G.nodes(data=True):
#                 w.writerow([
#                     n,
#                     d.get("kind",""),
#                     d.get("url",""),
#                     d.get("title",""),
#                     d.get("section_id",""),
#                     d.get("level",""),
#                     d.get("heading_text",""),
#                     d.get("heading_path",""),
#                     d.get("file_id",""),
#                     d.get("filename",""),
#                     d.get("page_no",""),
#                 ])
#         with (outdir / "edges.csv").open("w", newline="", encoding="utf-8") as cf:
#             w = csv.writer(cf)
#             w.writerow(["src","dst","rel","anchor_text","position"]) 
#             for u, v, d in self.G.edges(data=True):
#                 w.writerow([u, v, d.get("rel",""), d.get("anchor_text",""), d.get("position","")])
#         # Chunks for vector DB
#         with (outdir / "chunks.jsonl").open("w", encoding="utf-8") as ch:
#             for c in self.chunks:
#                 ch.write(json.dumps(c, ensure_ascii=False) + "\n")
#         # GraphML for viz
#         nx.write_graphml(self.G, outdir / "site.graphml")


# # ===================== Main =====================

# def main():
#     ap = argparse.ArgumentParser(description="Build graph-lite for website + PDFs and emit graph + chunk files")
#     ap.add_argument("--base-url", required=True)
#     ap.add_argument("--pdf-dir", default=None, help="Directory containing PDFs to ingest")
#     ap.add_argument("--include-subdomains", action="store_true")
#     ap.add_argument("--max-pages", type=int, default=1000)
#     ap.add_argument("--outdir", default="out_graph")
#     ap.add_argument("--delay", type=float, default=0.3)
#     args = ap.parse_args()

#     crawler = SiteCrawler(args.base_url, max_pages=args.max_pages, include_subdomains=args.include_subdomains, delay=args.delay)
#     pages, link_edges = crawler.crawl()
#     print(f"[INFO] Crawled {len(pages)} pages, {len(link_edges)} internal links")

#     gb = GraphLiteBuilder()

#     # Add website nodes/edges
#     for url, pd in pages.items():
#         gb.add_page(pd)
#     for src, dst, lnk in link_edges:
#         gb.add_link(src, dst, lnk)

#     # PDFs
#     pdf_url_map: Dict[str, Tuple[str, Optional[int]]] = {}
#     if args.pdf_dir:
#         pdf_dir = Path(args.pdf_dir)
#         pdf_docs, pdf_pages = ingest_pdfs(pdf_dir)
#         # Add pdfdoc and pdfpage nodes
#         for fid, fname in pdf_docs.items():
#             gb.add_pdf_doc(fid, fname)
#             for pinfo in pdf_pages[fid]:
#                 gb.add_pdf_page(fid, pinfo.page_no, pinfo.text)
#         # Map absolute URLs that point to local PDFs (best effort)
#         # Heuristic: if an anchor href ends with a file in pdf_dir, map it; also parse #page=N
#         name_to_id = {fname: fid for fid, fname in pdf_docs.items()}
#         for url, pd in pages.items():
#             for lnk in pd.links:
#                 href = lnk.href
#                 if href.lower().endswith('.pdf'):
#                     fname = Path(urlparse(href).path).name
#                     fid = name_to_id.get(fname)
#                     if fid:
#                         pdf_url_map[href] = (fid, None)
#                 # detect #page=N or ?page=N
#                 if '.pdf' in href.lower():
#                     try:
#                         page_no = None
#                         u = urlparse(href)
#                         # query param page=N
#                         qs = u.query
#                         m = re.search(r"page=(\d+)", qs)
#                         if m:
#                             page_no = int(m.group(1))
#                         # fragment page=N
#                         m2 = re.search(r"page=(\d+)", u.fragment)
#                         if m2:
#                             page_no = int(m2.group(1))
#                         if page_no is not None:
#                             fname = Path(u.path).name
#                             fid = name_to_id.get(fname)
#                             if fid:
#                                 pdf_url_map[href] = (fid, page_no)
#                     except Exception:
#                         pass
#         # Create references from sections to PDF pages
#         gb.add_section_pdf_reference_edges(pages, pdf_url_map)

#     outdir = Path(args.outdir)
#     gb.export(outdir)
#     print(f"[OK] Exported graph + chunks to {outdir}")


# if __name__ == "__main__":
#     main()



from openai import OpenAI
import os
from typing import List, Optional
import time
from dotenv import load_dotenv
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI Embedding Generator
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env variable)
            model: Embedding model to use (default: text-embedding-3-small)
                   Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        print(f"✓ Embedding Generator initialized with model: {model}")
    
    def generate_embedding(self, text: str, max_tokens: int = 8000) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            max_tokens: Maximum tokens to process (truncate if longer)
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        if not text or not text.strip():
            print("  ⚠ Empty text, skipping embedding")
            return None
        
        try:
            # Truncate text if too long (rough estimation: 1 token ≈ 4 characters)
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]
                print(f"  ⚠ Text truncated to {max_chars} characters")
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            print(f"  ✗ Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], max_tokens: int = 8000, 
                                  delay: float = 0.1) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts with rate limiting
        
        Args:
            texts: List of texts to embed
            max_tokens: Maximum tokens per text
            delay: Delay between requests in seconds
            
        Returns:
            List of embeddings (same order as input texts)
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"  Generating embedding {i+1}/{len(texts)}...")
            embedding = self.generate_embedding(text, max_tokens)
            embeddings.append(embedding)
            
            # Rate limiting
            if i < len(texts) - 1:  # Don't delay after last request
                time.sleep(delay)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)
    


if __name__ == "__main__":
    # Example usage
    api_key = os.getenv('OPENAI_API_KEY')
    generator = EmbeddingGenerator(api_key=api_key, model="text-embedding-3-small")
    texts = [
        "Hello world!",
        "This is a test of the OpenAI embedding API.",
        "Embeddings are useful for many NLP tasks."
    ]
    embeddings = generator.generate_embeddings_batch(texts, delay=0.5)
    for i, emb in enumerate(embeddings):
        if emb:
            print(f"Embedding {i+1} (dim {len(emb)}): {emb[:5]}...")  # Print first 5 values
        else:
            print(f"Embedding {i+1}: Failed to generate")
                
