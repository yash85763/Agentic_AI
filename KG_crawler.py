from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
import hashlib
from collections import deque
import time
import networkx as nx
from embeddings import EmbeddingGenerator

class KnowledgeGraphCrawler:
    def __init__(self, start_url, main_domains, max_depth=None, output_dir="kg_output", proxy=None, max_pages=None, openai_api_key=None, 
             embedding_model="text-embedding-3-small", generate_embeddings=True):
        """
        Knowledge Graph Crawler
        
        Args:
            start_url: Starting URL (homepage)
            main_domains: List of domains to crawl completely
            max_depth: Maximum crawl depth (None = unlimited for main domains)
            output_dir: Output directory
            proxy: Proxy server (e.g., "http://proxy.example.com:8080")
        """
        self.start_url = start_url
        self.main_domains = [d.lower() for d in main_domains]
        self.max_depth = max_depth
        self.output_dir = output_dir
        self.proxy = proxy
        self.max_pages = max_pages
        
        self.visited = set()
        self.queued = set()
        self.graph = nx.DiGraph()
        self.page_contents = {}
        self.is_first_page = True
        self.homepage_nav_links = set()

        self.homepage_nav_links = set()

        # Embedding configuration
        self.generate_embeddings = generate_embeddings
        self.embedding_generator = None
        if self.generate_embeddings:
            try:
                self.embedding_generator = EmbeddingGenerator(
                    api_key=openai_api_key,
                    model=embedding_model
                )
            except Exception as e:
                print(f"⚠ Warning: Could not initialize embedding generator: {e}")
                print("  Continuing without embeddings...")
                self.generate_embeddings = False
        
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/pages", exist_ok=True)
    
    def setup_browser(self):
        """Setup Playwright Browser"""
        try:
            self.playwright = sync_playwright().start()
            
            # Browser launch options
            launch_options = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            }
            
            # Add proxy if provided
            if self.proxy:
                print(f"Configuring proxy: {self.proxy}")
                proxy_config = {'server': self.proxy}
                launch_options['proxy'] = proxy_config
            
            # Launch browser
            self.browser = self.playwright.chromium.launch(**launch_options)
            
            # Create context with user agent
            context_options = {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'viewport': {'width': 1920, 'height': 1080}
            }
            
            self.context = self.browser.new_context(**context_options)
            self.context.set_default_timeout(60000)  # 60 seconds
            
            # Create page
            self.page = self.context.new_page()
            
            print("✓ Playwright Browser initialized\n")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("\nPlease ensure:")
            print("  1. Playwright is installed: pip install playwright")
            print("  2. Browsers are installed: playwright install chromium")
            return False
    
    def wait_for_page_load(self):
        """Wait for page to fully load including dynamic content"""
        try:
            # Wait for network to be idle
            self.page.wait_for_load_state('networkidle', timeout=10000)
            
            # Wait for "Loading..." text to disappear
            try:
                self.page.wait_for_function(
                    "() => !document.body.textContent.includes('Loading')",
                    timeout=5000
                )
            except:
                pass
            
            # Additional wait for JavaScript
            time.sleep(2)
            
            # Scroll to trigger lazy loading
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            time.sleep(0.5)
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.5)
            self.page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)
            
            # Final wait
            time.sleep(1)
            
        except Exception as e:
            print(f"    Wait warning: {e}")
            time.sleep(5)
    
    def get_domain(self, url):
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""
    
    def is_main_domain(self, url):
        """Check if URL is from main domains"""
        domain = self.get_domain(url)
        return any(domain == d or domain.endswith('.' + d) for d in self.main_domains)
    
    def is_pdf(self, url):
        """Check if URL points to a PDF file (by .pdf extension)"""
        try:
            path = urlparse(url).path.lower()
            return path.endswith('.pdf')
        except:
            return False
    
    def get_entity_name_from_url(self, url):
        """Extract entity name from URL (last part after /)"""
        try:
            path = urlparse(url).path.strip('/')
            if not path:
                return self.get_domain(url)
            parts = path.split('/')
            name = parts[-1] if parts else self.get_domain(url)
            # Remove query params and fragments
            name = name.split('?')[0].split('#')[0]
            return name if name else self.get_domain(url)
        except:
            return "external_link"
    
    def extract_page_body_content(self, soup, include_header=False):
        """Extract content from page-body class, optionally including page-header"""
        
        if include_header:
            # For homepage: parse everything
            body = soup.find('body')
            if not body:
                return ""
            
            # Remove unwanted elements but keep page-header
            for tag in body.find_all(["script", "style", "iframe", "noscript"]):
                tag.decompose()
            
            text = body.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text
        else:
            # For other pages: only page-body, exclude page-header
            page_body = soup.find(class_='page-body')
            
            if not page_body:
                print("    ⚠ Warning: No 'page-body' class found")
                return ""
            
            # Remove page-header and page-footer elements from page-body (if any nested)
            for header in page_body.find_all(class_='page-header'):
                header.decompose()
            for footer in page_body.find_all(class_='page-footer'):
                footer.decompose()

            # Remove unwanted elements
            for tag in page_body.find_all(["script", "style", "nav", "footer", "iframe", "noscript"]):
                tag.decompose()
            
            # Extract text
            text = page_body.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text
    
    def extract_page_title(self, soup):
        """Extract page title from h1, h2, or h3 tags (priority order)"""
        # Try to find title from page-body first
        page_body = soup.find(class_='page-body')
        search_area = page_body if page_body else soup
        
        # Try h1 first
        h1 = search_area.find('h1')
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        
        # Try h2 if no h1
        h2 = search_area.find('h2')
        if h2 and h2.get_text(strip=True):
            return h2.get_text(strip=True)
        
        # Try h3 if no h2
        h3 = search_area.find('h3')
        if h3 and h3.get_text(strip=True):
            return h3.get_text(strip=True)
        
        # Fallback to URL-based name
        return None
    
    def extract_nav_links_from_header(self, soup, base_url):
        """Extract navigation links from page-header (only called once for homepage)"""
        nav_links = []
        
        page_header = soup.find(class_='page-header')
        if not page_header:
            print("  ⚠ No page-header found on homepage")
            return nav_links
        
        for a_tag in page_header.find_all('a', href=True):
            href = a_tag.get('href')
            
            # Skip invalid links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Remove fragment
            absolute_url = absolute_url.split('#')[0]
            
            # Only keep links from main domain
            if self.is_main_domain(absolute_url):
                nav_links.append(absolute_url)
        
        return nav_links
    
    def extract_page_links(self, soup, base_url):
        """Extract all links from page"""
        links = []
        seen_urls = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            
            # Skip invalid links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Remove fragment
            absolute_url = absolute_url.split('#')[0]
            
            # Skip duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)
            
            # Get link text
            link_text = a_tag.get_text(strip=True) or "Link"
            
            links.append({
                'url': absolute_url,
                'link_text': link_text
            })
        
        return links
    
    def create_safe_filename(self, url):
        """Create safe filename from URL"""
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
        """Save page content to file"""
        filename = self.create_safe_filename(url)
        filepath = os.path.join(self.output_dir, 'pages', filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(url)
                f.write('\n\n')
                f.write(content)
            return filepath
        except Exception as e:
            print(f"    Error saving: {e}")
            return None
    
    def should_crawl(self, url, current_depth):
        """Determine if URL should be crawled"""
        # PDFs are not crawled
        if self.is_pdf(url):
            return False
        
        # Check max depth limit (if set)
        if self.max_depth is not None and current_depth > self.max_depth:
            return False
        
        # Only crawl main domain
        if self.is_main_domain(url):
            return True
        
        # Everything else - don't crawl
        return False
    
    def crawl(self):
        """Main crawling function"""
        print("="*80)
        print("KNOWLEDGE GRAPH CRAWLER - PLAYWRIGHT VERSION")
        print("="*80)
        depth_info = f"max depth {self.max_depth}" if self.max_depth is not None else "unlimited depth"
        pages_info = f"max {self.max_pages} pages" if self.max_pages is not None else "unlimited pages"
        print(f"Main domains ({depth_info}, {pages_info}): {', '.join(self.main_domains)}")
        print(f"Content parsing: 'page-body' only (excluding page-header and page-footer)")
        print(f"Navbar links: Detected but not included in graph relationships")
        print(f"Node naming: h1 > h2 > h3 from page-body")
        print("="*80 + "\n")
        
        if not self.setup_browser():
            return
        
        queue = deque([(self.start_url, 0)])
        self.queued.add(self.start_url)  # Track the start URL as queued
        
        try:
            while queue:
                # Check if page limit reached
                if self.max_pages is not None and len(self.visited) >= self.max_pages:
                    print(f"\n⚠ Page limit reached ({self.max_pages} pages). Stopping crawl.")
                    print(f"  Remaining URLs in queue: {len(queue)}")
                    break

                url, depth = queue.popleft()
                
                # Skip if already visited
                if url in self.visited:
                    continue
                
                # Check if should crawl
                if not self.should_crawl(url, depth):
                    # Add as external entity
                    entity_name = self.get_entity_name_from_url(url)
                    self.graph.add_node(url, 
                                       depth=depth, 
                                       label=entity_name, 
                                       is_external=True,
                                       is_pdf=self.is_pdf(url))
                    continue
                
                page_type = "HOMEPAGE" if self.is_first_page else "PAGE"
                print(f"\n{'='*80}")
                print(f"[{page_type}] Depth {depth}: {url}")
                print(f"{'='*80}")
                
                self.visited.add(url)
                
                # Add node to graph (will update label after extracting content)
                self.graph.add_node(url, 
                                   depth=depth, 
                                   label=self.get_entity_name_from_url(url),  # Temporary label
                                   is_external=False, 
                                   is_pdf=False)
                
                # Load page
                print("  → Loading page...")
                try:
                    self.page.goto(url, wait_until='domcontentloaded', timeout=60000)
                    self.wait_for_page_load()
                except PlaywrightTimeoutError:
                    print(f"  ⚠ Timeout loading page, continuing anyway...")
                except Exception as e:
                    print(f"  ✗ Error loading page: {e}")
                    continue
                
                # Get HTML
                html = self.page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Special handling for homepage (first page)
                if self.is_first_page:
                    print("  → HOMEPAGE: Parsing complete page including navbar")
                    
                    # Extract navigation links from page-header
                    nav_links = self.extract_nav_links_from_header(soup, url)
                    self.homepage_nav_links = set(nav_links)
                    print(f"  ✓ Extracted {len(self.homepage_nav_links)} navigation links from navbar")
                    if self.homepage_nav_links:
                        for nav_link in list(self.homepage_nav_links)[:5]:
                            print(f"    • {nav_link}")
                        if len(self.homepage_nav_links) > 5:
                            print(f"    ... and {len(self.homepage_nav_links) - 5} more")
                    
                    # Extract text from entire page (including header for homepage)
                    text = self.extract_page_body_content(soup, include_header=True)
                    
                    # Mark that we've processed the first page
                    self.is_first_page = False
                else:
                    print("  → Parsing page-body only (ignoring navbar)")
                    # Extract text from page-body only (excluding page-header)
                    text = self.extract_page_body_content(soup, include_header=False)
                
                # Check content quality
                if len(text) < 100:
                    print(f"  ⚠ Content seems incomplete ({len(text)} chars), waiting more...")
                    time.sleep(3)
                    html = self.page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    if self.is_first_page:
                        text = self.extract_page_body_content(soup, include_header=True)
                    else:
                        text = self.extract_page_body_content(soup, include_header=False)
                
                # Extract page title from h1/h2/h3 and update node label
                page_title = self.extract_page_title(soup)
                if page_title:
                    self.graph.nodes[url]['label'] = page_title
                    print(f"  ✓ Page title: {page_title}")
                else:
                    print(f"  ⚠ No h1/h2/h3 found, using URL-based name")

                # Save content
                filepath = self.save_content(url, text)
                if filepath:
                    self.page_contents[url] = filepath
                    print(f"  ✓ Content saved ({len(text)} chars)")

                # Store text content in node
                self.graph.nodes[url]['text'] = text

                # Generate and store embeddings
                if self.generate_embeddings and self.embedding_generator:
                    print(f"  → Generating embeddings...")
                    embedding = self.embedding_generator.generate_embedding(text)
                    if embedding:
                        self.graph.nodes[url]['embeddings'] = embedding
                        print(f"  ✓ Embeddings generated (dim: {len(embedding)})")
                    else:
                        self.graph.nodes[url]['embeddings'] = None
                        print(f"  ✗ Failed to generate embeddings")
                else:
                    self.graph.nodes[url]['embeddings'] = None
                
                # Extract links
                print(f"  → Extracting links...")
                links = self.extract_page_links(soup, url)
                
                # Count navigation vs content links
                nav_count = sum(1 for l in links if l['url'] in self.homepage_nav_links)
                content_count = len(links) - nav_count
                print(f"  ✓ Found {len(links)} links ({nav_count} navigation [ignored], {content_count} content)")
                
                # Process links
                queued = 0
                pdf_count = 0
                external_count = 0
                
                for link in links:
                    target_url = link['url']
                    link_text = link['link_text']
                    
                    # Check if this link is from the homepage navbar
                    is_nav = target_url in self.homepage_nav_links
                    
                    # Skip navigation links (back_to_page relationships)
                    if is_nav:
                        # Still need to queue the URL for crawling, but don't add edge
                        pass
                    else:
                        # Add edge to graph only for content links
                        self.graph.add_edge(url, target_url, relationship=link_text)
                    
                    # Handle different URL types
                    if self.is_pdf(target_url):
                        # PDF - add as external entity with PDF name
                        entity_name = self.get_entity_name_from_url(target_url)
                        self.graph.add_node(target_url, 
                                           label=entity_name, 
                                           is_external=True, 
                                           is_pdf=True)
                        pdf_count += 1
                        
                    elif self.is_main_domain(target_url):
                        # Main domain - queue if not visited AND not already queued
                        if target_url not in self.visited and target_url not in self.queued:
                            if self.max_depth is None or depth < self.max_depth:
                                queue.append((target_url, depth + 1))
                                self.queued.add(target_url)  # Mark as queued
                                queued += 1
                            else:
                                # Max depth reached - add as external entity
                                entity_name = self.get_entity_name_from_url(target_url)
                                self.graph.add_node(target_url, 
                                                label=entity_name, 
                                                is_external=True, 
                                                is_pdf=False)
                                external_count += 1
                    else:
                        # External domain - add as external entity
                        entity_name = self.get_entity_name_from_url(target_url)
                        self.graph.add_node(target_url, 
                                           label=entity_name, 
                                           is_external=True, 
                                           is_pdf=False)
                        external_count += 1
                
                print(f"  ✓ Queued: {queued} new URLs")
                print(f"  ✓ External: {external_count} links, {pdf_count} PDFs")
                print(f"  → Progress: Queue={len(queue)}, Visited={len(self.visited)}, Total Discovered={len(self.queued)}")
                
                # Be polite
                time.sleep(1)
        
        finally:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("\n✓ Browser closed")
        
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
        print(f"Navbar links detected: {len(self.homepage_nav_links)}")
        print(f"{'='*80}\n")
    
    def save_json(self):
        """Save graph to JSON"""
        data = {
            'metadata': {
            'start_url': self.start_url,
            'main_domains': self.main_domains,
            'max_depth': self.max_depth,
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'crawled_pages': len(self.visited),
            'navbar_links': len(self.homepage_nav_links),
            'content_source': 'page-body class (navbar parsed once from homepage)',
            'node_naming': 'h1 > h2 > h3 from page-body',
            'embeddings_enabled': self.generate_embeddings,
            'embedding_model': self.embedding_generator.model if self.embedding_generator else None,
            'embedding_dimension': self.embedding_generator.get_embedding_dimension() if self.embedding_generator else None
        },
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            node_data = {
                'id': node,
                'url': node,
                'label': attrs.get('label', self.get_entity_name_from_url(node)),
                'depth': attrs.get('depth', 0),
                'is_external': attrs.get('is_external', False),
                'is_pdf': attrs.get('is_pdf', False),
                'text': attrs.get('text', ''),
                'embeddings': attrs.get('embeddings', None)
            }
            data['nodes'].append(node_data)
        
        for src, tgt, attrs in self.graph.edges(data=True):
            data['edges'].append({
                'from': src,
                'to': tgt,
                'label': attrs.get('relationship', '')
            })
        
        filepath = os.path.join(self.output_dir, 'graph.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Graph JSON: {filepath}")
        return data
    
    def create_interactive_graph(self, graph_data):
        """Create interactive HTML graph visualization"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: white; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; }
        .stats { background: #16213e; padding: 15px 20px; display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .stat { text-align: center; }
        .stat .value { font-size: 28px; font-weight: bold; color: #667eea; }
        .stat .label { font-size: 12px; color: #aaa; margin-top: 3px; }
        .controls { background: #16213e; padding: 15px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .controls input, .controls select, .controls button { padding: 10px 15px; border-radius: 5px; border: none; font-size: 14px; }
        .controls input { flex: 1; max-width: 300px; }
        .controls button { background: #667eea; color: white; cursor: pointer; font-weight: bold; transition: background 0.3s; }
        .controls button:hover { background: #5568d3; }
        #network { height: calc(100vh - 220px); background: #0f3460; }
        .info { position: fixed; top: 200px; right: 20px; background: rgba(22,33,62,0.95); padding: 20px; border-radius: 10px; max-width: 350px; display: none; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-height: 60vh; overflow-y: auto; }
        .info h3 { color: #667eea; margin-bottom: 10px; font-size: 18px; }
        .info p { margin: 8px 0; font-size: 13px; line-height: 1.5; }
        .info a { color: #667eea; text-decoration: none; word-break: break-all; }
        .info a:hover { text-decoration: underline; }
        .close { float: right; cursor: pointer; font-size: 20px; color: #aaa; line-height: 1; }
        .close:hover { color: white; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; margin: 3px 3px 3px 0; font-weight: bold; }
        .badge-nav { background: #3498db; color: white; }
        .badge-external { background: #e74c3c; color: white; }
        .badge-pdf { background: #e67e22; color: white; }
        .badge-crawled { background: #27ae60; color: white; }
        .legend { position: fixed; bottom: 20px; left: 20px; background: rgba(22,33,62,0.95); padding: 15px; border-radius: 10px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .legend-line { width: 30px; height: 2px; margin-right: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🕸️ Knowledge Graph Visualization</h1>
    </div>
    
    <div class="stats">
        <div class="stat"><div class="value" id="nodes">0</div><div class="label">Total Nodes</div></div>
        <div class="stat"><div class="value" id="edges">0</div><div class="label">Total Edges</div></div>
        <div class="stat"><div class="value" id="crawled">0</div><div class="label">Pages Crawled</div></div>
        <div class="stat"><div class="value" id="navbar">0</div><div class="label">Navbar Links</div></div>
        <div class="stat"><div class="value" id="external">0</div><div class="label">External Links</div></div>
    </div>
    
    <div class="controls">
        <input type="text" id="search" placeholder="🔍 Search nodes by name or URL..." />
        <select id="filter">
            <option value="">All Nodes</option>
            <option value="crawled">Crawled Pages</option>
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
        <div class="legend-item"><div class="legend-line" style="background: #666;"></div>Content Link</div>
    </div>
    
    <script>
        const data = GRAPH_DATA;
        
        // Update statistics
        document.getElementById('nodes').textContent = data.nodes.length;
        document.getElementById('edges').textContent = data.edges.length;
        document.getElementById('crawled').textContent = data.metadata.crawled_pages;
        document.getElementById('navbar').textContent = data.metadata.navbar_links;
        const pdfCount = data.nodes.filter(n => n.is_pdf).length;
        const extCount = data.nodes.filter(n => n.is_external && !n.is_pdf).length;
        document.getElementById('external').textContent = extCount;
        
        // Prepare nodes for vis.js
        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            title: n.url,
            color: n.is_pdf ? '#e67e22' : (n.is_external ? '#e74c3c' : '#667eea'),
            shape: n.is_external ? 'box' : 'dot',
            size: n.is_external ? 15 : 25,
            font: { color: 'white', size: 12 },
            ...n
        }));
        
        // Prepare edges for vis.js (no back_to_page edges anymore)
        const edges = data.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            arrows: 'to',
            color: { color: '#666' },
            font: { size: 9, color: '#aaa', strokeWidth: 0 },
            width: 1
        }));
        
        // Create network
        const network = new vis.Network(
            document.getElementById('network'),
            { nodes, edges },
            {
                physics: {
                    enabled: true,
                    barnesHut: {
                        gravitationalConstant: -8000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04
                    },
                    stabilization: { iterations: 200 }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 100,
                    navigationButtons: true,
                    keyboard: true
                }
            }
        );
        
        // Node click handler
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
            if (navEdges.length > 0) badges += '<span class="badge badge-nav">' + navEdges.length + ' NAV LINKS</span>';
            
            let contentHtml = `
                <h3>${node.label}</h3>
                ${badges}
                <p><strong>URL:</strong><br><a href="${node.url}" target="_blank">${node.url}</a></p>
                <p><strong>Depth:</strong> ${node.depth}</p>
                <p><strong>Outgoing Links:</strong> ${outEdges.length} (${navEdges.length} navigation)</p>
                <p><strong>Incoming Links:</strong> ${inEdges.length}</p>
            `;
            
            if (outEdges.length > 0 && outEdges.length <= 10) {
                contentHtml += '<p><strong>Links to:</strong></p><ul style="margin-left: 20px; font-size: 12px;">';
                outEdges.forEach(e => {
                    const target = data.nodes.find(n => n.id === e.to);
                    contentHtml += `<li>${target ? target.label : 'Unknown'} (${e.label})</li>`;
                });
                contentHtml += '</ul>';
            }
            
            document.getElementById('info-content').innerHTML = contentHtml;
            document.getElementById('info').style.display = 'block';
        }
        
        function closeInfo() {
            document.getElementById('info').style.display = 'none';
        }
        
        // Search functionality
        document.getElementById('search').oninput = e => {
            const term = e.target.value.toLowerCase();
            if (!term) {
                network.setData({ nodes, edges });
                return;
            }
            const filtered = nodes.filter(n => 
                n.label.toLowerCase().includes(term) || n.url.toLowerCase().includes(term)
            );
            const nodeIds = filtered.map(n => n.id);
            const filteredEdges = edges.filter(e => nodeIds.includes(e.from) && nodeIds.includes(e.to));
            network.setData({ nodes: filtered, edges: filteredEdges });
        };
        
        // Filter functionality
        document.getElementById('filter').onchange = e => {
            let filtered = nodes;
            if (e.target.value === 'crawled') filtered = nodes.filter(n => !n.is_external);
            if (e.target.value === 'external') filtered = nodes.filter(n => n.is_external && !n.is_pdf);
            if (e.target.value === 'pdf') filtered = nodes.filter(n => n.is_pdf);
            const nodeIds = filtered.map(n => n.id);
            const filteredEdges = edges.filter(e => nodeIds.includes(e.from) && nodeIds.includes(e.to));
            network.setData({ nodes: filtered, edges: filteredEdges });
        };
        
        function resetView() {
            network.setData({ nodes, edges });
            document.getElementById('search').value = '';
            document.getElementById('filter').value = '';
            closeInfo();
        }
        
        function fitNetwork() {
            network.fit();
        }
    </script>
</body>
</html>""".replace('GRAPH_DATA', json.dumps(graph_data))
        
        filepath = os.path.join(self.output_dir, 'interactive_graph.html')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Interactive graph: {filepath}")
        print(f"  → Open this file in your browser to view the graph!")


def main():
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH CRAWLER - FINAL VERSION")
    print("="*80)
    print("Features:")
    print("  • Playwright browser automation")
    print("  • Navbar parsed once from homepage")
    print("  • All other pages: parse page-body only")
    print("  • Node naming: h1 > h2 > h3 from page-body")
    print("  • Only crawls specified main domain(s)")
    print("  • External links saved but not crawled")
    print("  • Proxy support")
    print("  • Interactive graph visualization")
    print("="*80 + "\n")
    
    # Get starting URL
    start_url = input("Enter starting URL (homepage): ").strip()
    if not start_url:
        print("Error: Starting URL required")
        return
    
    # Get main domains
    print("\nMain domain(s) to crawl (comma-separated):")
    print("  Example: example.com")
    print("  Or: example.com,www.example.com")
    main_domains = input("Main domains: ").strip()
    main_domains = [d.strip() for d in main_domains.split(',') if d.strip()]
    
    if not main_domains:
        print("Error: At least one main domain required")
        return
    
    # Get max depth
    print("\nMaximum crawl depth (leave empty for unlimited):")
    print("  Example: 3, 4, 5")
    print("  Empty = crawl until no more pages")
    max_depth_input = input("Max depth: ").strip()
    max_depth = int(max_depth_input) if max_depth_input else None

    # Get max pages
    print("\nMaximum pages to crawl (leave empty for unlimited):")
    print("  Example: 50, 100, 200")
    print("  Empty = crawl all pages up to max depth")
    max_pages_input = input("Max pages: ").strip()
    max_pages = int(max_pages_input) if max_pages_input else None
    
    # Get proxy configuration (optional)
    print("\nProxy server (optional, leave empty to skip):")
    print("  Examples:")
    print("    - HTTP: http://proxy.example.com:8080")
    print("    - HTTPS: https://proxy.example.com:8080")
    print("    - SOCKS5: socks5://127.0.0.1:1080")
    proxy = input("Proxy server: ").strip() or None

    # Get OpenAI configuration
    print("\nOpenAI Embeddings (optional, leave empty to skip):")
    print("  You can set OPENAI_API_KEY environment variable")
    print("  Or provide it here")
    generate_embeddings_input = input("Generate embeddings? (y/n, default: n): ").strip().lower()
    generate_embeddings = generate_embeddings_input == 'y'

    openai_api_key = None
    embedding_model = "text-embedding-3-small"
    if generate_embeddings:
        api_key_input = input("OpenAI API Key (or press Enter to use env variable): ").strip()
        openai_api_key = api_key_input if api_key_input else None
        
        print("\nEmbedding model:")
        print("  1. text-embedding-3-small (default, 1536 dim)")
        print("  2. text-embedding-3-large (3072 dim)")
        print("  3. text-embedding-ada-002 (1536 dim)")
        model_choice = input("Choose model (1/2/3, default: 1): ").strip()
        
        model_map = {
            "1": "text-embedding-3-small",
            "2": "text-embedding-3-large",
            "3": "text-embedding-ada-002"
        }
        embedding_model = model_map.get(model_choice, "text-embedding-3-small")
    
    # Show configuration
    print(f"\n{'='*80}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"Start URL: {start_url}")
    depth_display = f"{max_depth}" if max_depth is not None else "unlimited"
    print(f"Max depth: {depth_display}")
    pages_display = f"{max_pages}" if max_pages is not None else "unlimited"
    print(f"Max pages: {pages_display}")
    print(f"Main domains ({len(main_domains)}): {', '.join(main_domains)}")
    if proxy:
        print(f"Proxy: {proxy}")
    if generate_embeddings:
        print(f"Embeddings: Enabled (model: {embedding_model})")
    else:
        print(f"Embeddings: Disabled")
    print(f"Parsing strategy:")
    print(f"  - Homepage: Complete page including navbar")
    print(f"  - Other pages: page-body only (navbar ignored)")
    print(f"Node naming: h1 > h2 > h3 from page-body")
    print(f"{'='*80}\n")
    
    # Confirm
    proceed = input("Start crawling? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Cancelled")
        return
    
    # Create and run crawler
    print("\nInitializing crawler...\n")
    crawler = KnowledgeGraphCrawler(
        start_url=start_url,
        main_domains=main_domains,
        max_depth=max_depth,
        proxy=proxy, 
        max_pages=max_pages, 
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        generate_embeddings=generate_embeddings
    )
    
    # Start crawling
    crawler.crawl()
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80 + "\n")
    
    graph_data = crawler.save_json()
    crawler.create_interactive_graph(graph_data)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ ALL COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {crawler.output_dir}/")
    print("\nFiles created:")
    print(f"  📊 interactive_graph.html  ← OPEN THIS in your browser!")
    print(f"  📄 graph.json              ← Complete graph data")
    print(f"  📁 pages/*.txt             ← Crawled page contents")
    print("\nGraph Statistics:")
    print(f"  • Pages crawled: {len(crawler.visited)}")
    print(f"  • Navbar links detected: {len(crawler.homepage_nav_links)}")
    print(f"  • Total nodes: {crawler.graph.number_of_nodes()}")
    print(f"  • Total edges: {crawler.graph.number_of_edges()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
