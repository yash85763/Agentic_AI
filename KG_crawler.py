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
    def __init__(self, start_url, main_domains, include_domains, nav_links, 
                 max_depth=None, nav_container_class="page-header", output_dir="kg_output"):
        """
        Knowledge Graph Crawler
        
        Args:
            start_url: Starting URL
            main_domains: List of domains to crawl completely
            include_domains: List of domains to crawl up to depth 2
            nav_links: List of navigation URLs/paths (for "back_to_page" relationship)
            max_depth: Maximum crawl depth (None = unlimited for main domains)
            nav_container_class: CSS class containing navigation (default: "page-header")
            output_dir: Output directory
        """
        self.start_url = start_url
        self.main_domains = [d.lower() for d in main_domains]
        self.include_domains = [d.lower() for d in include_domains]
        self.nav_links = set(nav_links) if nav_links else set()
        self.max_depth = max_depth
        self.nav_container_class = nav_container_class
        self.output_dir = output_dir
        
        self.visited = set()
        self.graph = nx.DiGraph()
        self.page_contents = {}
        self.domain_depths = {}
        
        self.driver = None
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/pages", exist_ok=True)
    
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        options.page_load_strategy = 'normal'
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(60)
            self.driver.implicitly_wait(10)
            print("✓ WebDriver initialized\n")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            print("\nPlease ensure:")
            print("  1. Chrome browser is installed")
            print("  2. ChromeDriver is installed: pip install selenium")
            return False
    
    def wait_for_page_load(self):
        """Wait for page to fully load including dynamic content"""
        try:
            # Wait for body element
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for "Loading..." text to disappear
            try:
                WebDriverWait(self.driver, 5).until_not(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Loading')]"))
                )
            except:
                pass
            
            # Wait for JavaScript execution
            time.sleep(3)
            
            # Scroll to trigger lazy loading
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            
            # Wait for document ready state
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            
            # Final wait for AJAX
            time.sleep(2)
            
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
    
    def is_include_domain(self, url):
        """Check if URL is from include domains"""
        domain = self.get_domain(url)
        return any(domain == d or domain.endswith('.' + d) for d in self.include_domains)
    
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
    
    def is_in_navigation_container(self, element):
        """Check if element is within the navigation container"""
        current = element
        for _ in range(10):  # Check up to 10 levels up
            if current is None:
                break
            
            # Check if current element has the navigation container class
            classes = current.get('class', [])
            if self.nav_container_class in classes:
                return True
            
            # Move to parent
            current = current.parent
        
        return False
    
    def url_matches_nav_link(self, url):
        """Check if URL matches any navigation link"""
        for nav_link in self.nav_links:
            # Exact match
            if url == nav_link:
                return True
            # Pattern match (if nav_link is a path pattern)
            if nav_link in url:
                return True
        return False
    
    def extract_page_links(self, soup, base_url):
        """Extract all links from page"""
        links = []
        seen_key = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            
            # Skip invalid links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Remove fragment
            absolute_url = absolute_url.split('#')[0]
            
            # Get link text
            link_text = a_tag.get_text(strip=True) or "Link"
            
            # Check if this is a navigation link:
            # Must be BOTH in navigation container AND match nav link list
            in_nav_container = self.is_in_navigation_container(a_tag)
            matches_nav_url = self.url_matches_nav_link(absolute_url)
            is_nav = in_nav_container and matches_nav_url
            
            # Create unique key (url, is_nav) to allow same URL with different context
            key = (absolute_url, is_nav)
            if key in seen_key:
                continue
            seen_key.add(key)
            
            links.append({
                'url': absolute_url,
                'link_text': link_text,
                'is_navigation': is_nav
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
        
        # Main domain - crawl up to max_depth (or unlimited if max_depth is None)
        if self.is_main_domain(url):
            return True
        
        # Include domain - crawl up to depth 2
        if self.is_include_domain(url):
            if url not in self.domain_depths:
                self.domain_depths[url] = current_depth
            return self.domain_depths[url] <= 2
        
        # External - don't crawl
        return False
    
    def crawl(self):
        """Main crawling function"""
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
            return
        
        queue = deque([(self.start_url, 0)])
        
        try:
            while queue:
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
                
                domain_type = "MAIN" if self.is_main_domain(url) else "INCLUDE"
                print(f"\n{'='*80}")
                print(f"[{domain_type}] Depth {depth}: {url}")
                print(f"{'='*80}")
                
                self.visited.add(url)
                
                # Add node to graph
                self.graph.add_node(url, 
                                   depth=depth, 
                                   label=self.get_entity_name_from_url(url),
                                   is_external=False, 
                                   is_pdf=False)
                
                # Load page
                print("  → Loading page...")
                self.driver.get(url)
                self.wait_for_page_load()
                
                # Verify content loaded
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                if len(page_text.strip()) < 50 or "loading" in page_text.lower()[:100]:
                    print(f"  → Waiting for content to load...")
                    time.sleep(5)
                
                # Get HTML
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text
                for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                
                # Double-check content quality
                if len(text) < 100 or "loading" in text.lower()[:200]:
                    print(f"  → Content incomplete, waiting more...")
                    time.sleep(5)
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                
                # Save content
                filepath = self.save_content(url, text)
                if filepath:
                    self.page_contents[url] = filepath
                    print(f"  ✓ Content saved ({len(text)} chars)")
                
                # Extract links
                print(f"  → Extracting links...")
                links = self.extract_page_links(soup, url)
                
                # Count navigation vs content links
                nav_count = sum(1 for l in links if l['is_navigation'])
                new_count = len(links) - nav_count
                print(f"  ✓ Found {len(links)} links ({nav_count} navigation, {new_count} content)")
                
                # Process links
                queued = 0
                pdf_count = 0
                external_count = 0
                
                for link in links:
                    target_url = link['url']
                    link_text = link['link_text']
                    is_nav = link['is_navigation']
                    
                    # Use "back_to_page" for navigation links
                    relationship = "back_to_page" if is_nav else link_text
                    
                    # Add edge to graph
                    self.graph.add_edge(url, target_url, relationship=relationship)
                    
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
                        # Main domain - queue if not visited and within depth limit
                        if target_url not in self.visited:
                            if self.max_depth is None or depth < self.max_depth:
                                queue.append((target_url, depth + 1))
                                queued += 1
                            else:
                                # Max depth reached - add as external entity
                                entity_name = self.get_entity_name_from_url(target_url)
                                self.graph.add_node(target_url, 
                                                   label=entity_name, 
                                                   is_external=True, 
                                                   is_pdf=False)
                                external_count += 1
                            
                    elif self.is_include_domain(target_url):
                        # Include domain - queue only if depth < 2
                        if depth < 2 and target_url not in self.visited:
                            queue.append((target_url, depth + 1))
                            queued += 1
                        elif depth >= 2:
                            # Too deep - add as external entity
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
                print(f"  → Progress: Queue={len(queue)}, Visited={len(self.visited)}")
                
                # Be polite
                time.sleep(1)
        
        finally:
            if self.driver:
                self.driver.quit()
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
        print(f"{'='*80}\n")
    
    def save_json(self):
        """Save graph to JSON"""
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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Graph JSON: {filepath}")
        return data
    
    def create_interactive_graph(self, graph_data):
        """Create interactive HTML graph visualization"""
        
        # Validate graph data
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            print("✗ Error: Invalid graph data")
            return
        
        print(f"  Creating interactive graph...")
        print(f"    Nodes: {len(graph_data['nodes'])}")
        print(f"    Edges: {len(graph_data['edges'])}")
        
        # Serialize graph data to JSON with proper escaping
        try:
            json_data = json.dumps(graph_data, ensure_ascii=False, indent=None)
        except Exception as e:
            print(f"✗ Error serializing graph data: {e}")
            return
        
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
        .info { position: fixed; top: 200px; right: 20px; background: rgba(22,33,62,0.95); padding: 20px; border-radius: 10px; max-width: 350px; display: none; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-height: 60vh; overflow-y: auto; z-index: 1000; }
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
        .legend { position: fixed; bottom: 20px; left: 20px; background: rgba(22,33,62,0.95); padding: 15px; border-radius: 10px; font-size: 12px; z-index: 1000; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .legend-line { width: 30px; height: 2px; margin-right: 10px; }
        .loading { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(22,33,62,0.95); padding: 30px; border-radius: 10px; display: none; z-index: 2000; }
    </style>
</head>
<body>
    <div class="loading" id="loading">Loading graph...</div>
    
    <div class="header">
        <h1>🕸️ Knowledge Graph Visualization</h1>
    </div>
    
    <div class="stats">
        <div class="stat"><div class="value" id="nodes">0</div><div class="label">Total Nodes</div></div>
        <div class="stat"><div class="value" id="edges">0</div><div class="label">Total Edges</div></div>
        <div class="stat"><div class="value" id="crawled">0</div><div class="label">Pages Crawled</div></div>
        <div class="stat"><div class="value" id="pdfs">0</div><div class="label">PDF Documents</div></div>
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
        <div class="legend-item"><div class="legend-line" style="background: #3498db; border-style: dashed;"></div>Navigation</div>
        <div class="legend-item"><div class="legend-line" style="background: #666;"></div>Content Link</div>
    </div>
    
    <script>
        // Show loading
        document.getElementById('loading').style.display = 'block';
        
        try {
            const data = __GRAPH_DATA__;
            
            if (!data || !data.nodes || !data.edges) {
                throw new Error('Invalid graph data');
            }
            
            console.log('Graph data loaded:', data.nodes.length, 'nodes,', data.edges.length, 'edges');
            
            // Update statistics
            document.getElementById('nodes').textContent = data.nodes.length;
            document.getElementById('edges').textContent = data.edges.length;
            document.getElementById('crawled').textContent = data.metadata.crawled_pages;
            const pdfCount = data.nodes.filter(n => n.is_pdf).length;
            const extCount = data.nodes.filter(n => n.is_external && !n.is_pdf).length;
            document.getElementById('pdfs').textContent = pdfCount;
            document.getElementById('external').textContent = extCount;
            
            // Prepare nodes for vis.js
            const nodes = data.nodes.map(n => ({
                id: n.id,
                label: n.label || 'Unknown',
                title: n.url,
                color: n.is_pdf ? '#e67e22' : (n.is_external ? '#e74c3c' : '#667eea'),
                shape: n.is_external ? 'box' : 'dot',
                size: n.is_external ? 15 : 25,
                font: { color: 'white', size: 12 },
                ...n
            }));
            
            // Prepare edges for vis.js
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
            
            console.log('Creating network visualization...');
            
            // Create network
            const container = document.getElementById('network');
            const network = new vis.Network(
                container,
                { nodes: nodes, edges: edges },
                {
                    physics: {
                        enabled: true,
                        barnesHut: {
                            gravitationalConstant: -8000,
                            centralGravity: 0.3,
                            springLength: 150,
                            springConstant: 0.04
                        },
                        stabilization: { 
                            iterations: 200,
                            updateInterval: 25
                        }
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 100,
                        navigationButtons: true,
                        keyboard: true
                    }
                }
            );
            
            // Hide loading when stabilized
            network.once('stabilizationIterationsDone', function() {
                document.getElementById('loading').style.display = 'none';
                console.log('Graph stabilized and ready');
            });
            
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
                        const label = e.label === 'back_to_page' ? '[NAV]' : e.label;
                        contentHtml += `<li>${target ? target.label : 'Unknown'} (${label})</li>`;
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
            document.getElementById('search').addEventListener('input', e => {
                const term = e.target.value.toLowerCase();
                if (!term) {
                    network.setData({ nodes: nodes, edges: edges });
                    return;
                }
                const filtered = nodes.filter(n => 
                    n.label.toLowerCase().includes(term) || n.url.toLowerCase().includes(term)
                );
                const nodeIds = filtered.map(n => n.id);
                const filteredEdges = edges.filter(e => nodeIds.includes(e.from) && nodeIds.includes(e.to));
                network.setData({ nodes: filtered, edges: filteredEdges });
            });
            
            // Filter functionality
            document.getElementById('filter').addEventListener('change', e => {
                let filtered = nodes;
                if (e.target.value === 'crawled') filtered = nodes.filter(n => !n.is_external);
                if (e.target.value === 'external') filtered = nodes.filter(n => n.is_external && !n.is_pdf);
                if (e.target.value === 'pdf') filtered = nodes.filter(n => n.is_pdf);
                const nodeIds = filtered.map(n => n.id);
                const filteredEdges = edges.filter(e => nodeIds.includes(e.from) && nodeIds.includes(e.to));
                network.setData({ nodes: filtered, edges: filteredEdges });
            });
            
            function resetView() {
                network.setData({ nodes: nodes, edges: edges });
                document.getElementById('search').value = '';
                document.getElementById('filter').value = '';
                closeInfo();
            }
            
            function fitNetwork() {
                network.fit();
            }
            
            // Make functions global
            window.resetView = resetView;
            window.fitNetwork = fitNetwork;
            window.closeInfo = closeInfo;
            
            console.log('Graph visualization complete!');
            
        } catch (error) {
            document.getElementById('loading').innerHTML = 'Error loading graph: ' + error.message;
            console.error('Error:', error);
        }
    </script>
</body>
</html>"""
        
        # Replace placeholder with actual data
        html = html.replace('__GRAPH_DATA__', json_data)
        
        # Save HTML file
        filepath = os.path.join(self.output_dir, 'interactive_graph.html')
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            # Verify file was created and has content
            file_size = os.path.getsize(filepath)
            if file_size > 1000:  # Should be at least a few KB
                print(f"✓ Interactive graph: {filepath}")
                print(f"  File size: {file_size:,} bytes")
                print(f"  → Open this file in your browser to view the graph!")
            else:
                print(f"✗ Warning: Generated file seems too small ({file_size} bytes)")
                
        except Exception as e:
            print(f"✗ Error saving HTML file: {e}")
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
        <div class="stat"><div class="value" id="pdfs">0</div><div class="label">PDF Documents</div></div>
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
        <div class="legend-item"><div class="legend-line" style="background: #3498db; border-style: dashed;"></div>Navigation</div>
        <div class="legend-item"><div class="legend-line" style="background: #666;"></div>Content Link</div>
    </div>
    
    <script>
        const data = GRAPH_DATA;
        
        // Update statistics
        document.getElementById('nodes').textContent = data.nodes.length;
        document.getElementById('edges').textContent = data.edges.length;
        document.getElementById('crawled').textContent = data.metadata.crawled_pages;
        const pdfCount = data.nodes.filter(n => n.is_pdf).length;
        const extCount = data.nodes.filter(n => n.is_external && !n.is_pdf).length;
        document.getElementById('pdfs').textContent = pdfCount;
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
        
        // Prepare edges for vis.js
        const edges = data.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            arrows: 'to',
            color: { color: e.label === 'back_to_page' ? '#3498db' : '#666' },
            font: { size: 9, color: '#aaa', strokeWidth: 0 },
            dashes: e.label === 'back_to_page',
            width: e.label === 'back_to_page' ? 2 : 1
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
    print("  • Configurable max depth")
    print("  • Explicit navigation link control")
    print("  • PDF detection by .pdf extension")
    print("  • Interactive graph with validation")
    print("="*80 + "\n")
    
    # Get starting URL
    start_url = input("Enter starting URL: ").strip()
    if not start_url:
        print("Error: Starting URL required")
        return
    
    # Get main domains
    print("\nMain domains to crawl completely (comma-separated):")
    print("  Example: example.com,www.example.com")
    main_domains = input("Main domains: ").strip()
    main_domains = [d.strip() for d in main_domains.split(',') if d.strip()]
    
    if not main_domains:
        print("Error: At least one main domain required")
        return
    
    # Get include domains
    print("\nInclude domains to crawl up to depth 2 (optional, comma-separated):")
    include_domains = input("Include domains: ").strip()
    include_domains = [d.strip() for d in include_domains.split(',') if d.strip()] if include_domains else []
    
    # Get max depth
    print("\nMaximum crawl depth (leave empty for unlimited):")
    print("  Example: 3, 4, 5")
    print("  Empty = crawl until no more pages")
    max_depth_input = input("Max depth: ").strip()
    max_depth = int(max_depth_input) if max_depth_input else None
    
    # Get navigation configuration
    print("\nNavigation container CSS class:")
    print("  This is the CSS class that wraps your navigation menu")
    print("  Example: page-header, navbar, navigation, header-nav")
    nav_container = input("Nav container class (default: page-header): ").strip()
    nav_container = nav_container if nav_container else "page-header"
    
    print("\nNavigation links (URLs or paths, comma-separated):")
    print("  Example: /home,/about,/products,/contact")
    print("  Or: https://example.com/home,https://example.com/about")
    print("  These will use 'back_to_page' when found in the navigation container")
    nav_links_input = input("Nav links: ").strip()
    nav_links = []
    if nav_links_input:
        nav_links = [link.strip() for link in nav_links_input.split(',') if link.strip()]
    
    # Show configuration
    print(f"\n{'='*80}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"Start URL: {start_url}")
    depth_display = f"{max_depth}" if max_depth is not None else "unlimited"
    print(f"Max depth: {depth_display}")
    print(f"Main domains ({len(main_domains)}): {', '.join(main_domains)}")
    if include_domains:
        print(f"Include domains ({len(include_domains)}): {', '.join(include_domains)}")
    print(f"Navigation container: .{nav_container}")
    if nav_links:
        print(f"Navigation links ({len(nav_links)}):")
        for link in nav_links[:5]:
            print(f"  • {link}")
        if len(nav_links) > 5:
            print(f"  ... and {len(nav_links) - 5} more")
    else:
        print("Navigation links: None provided")
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
        include_domains=include_domains,
        nav_links=nav_links,
        max_depth=max_depth,
        nav_container_class=nav_container
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
    print(f"  • Total nodes: {crawler.graph.number_of_nodes()}")
    print(f"  • Total edges: {crawler.graph.number_of_edges()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()