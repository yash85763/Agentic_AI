from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
import hashlib
from collections import deque
import time
import networkx as nx

class SmartKGCrawler:
    def __init__(self, start_url, main_domains, include_domains, pdf_domain, output_dir="kg_output"):
        """
        Smart single-threaded crawler
        
        Args:
            start_url: Starting URL
            main_domains: List of domains to crawl completely
            include_domains: List of domains to crawl up to depth 2
            pdf_domain: Domain pattern for PDFs
            output_dir: Output directory
        """
        self.start_url = start_url
        self.main_domains = [d.lower() for d in main_domains]
        self.include_domains = [d.lower() for d in include_domains]
        self.pdf_domain = pdf_domain.lower() if pdf_domain else ""
        self.output_dir = output_dir
        
        self.visited = set()
        self.graph = nx.DiGraph()
        self.page_contents = {}
        self.domain_depths = {}
        
        # Track navigation menu links (to avoid re-discovering)
        self.navigation_links = set()
        
        self.driver = None
        
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
            print(f"Error: {e}")
            return False
    
    def wait_for_page_load(self):
        """Wait for page to fully load"""
        try:
            # Wait for body
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for "Loading..." to disappear
            try:
                WebDriverWait(self.driver, 5).until_not(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Loading')]"))
                )
            except:
                pass
            
            # Wait for JavaScript
            time.sleep(3)
            
            # Scroll to trigger lazy loading
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            
            # Wait for document ready
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            
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
    
    def is_pdf_domain(self, url):
        """Check if URL is from PDF domain"""
        if not self.pdf_domain:
            return False
        domain = self.get_domain(url)
        return domain == self.pdf_domain or domain.endswith('.' + self.pdf_domain)
    
    def get_entity_name_from_url(self, url):
        """Extract entity name from URL"""
        try:
            path = urlparse(url).path.strip('/')
            if not path:
                return self.get_domain(url)
            parts = path.split('/')
            name = parts[-1] if parts else self.get_domain(url)
            name = name.split('?')[0].split('#')[0]
            return name if name else self.get_domain(url)
        except:
            return "external_link"
    
    def discover_navigation_menus(self, current_url):
        """
        Discover navigation menus ONCE and mark them
        Returns: set of URLs from navigation menus
        """
        discovered = set()
        actions = ActionChains(self.driver)
        
        # Find navbar elements
        nav_selectors = [
            ".navbar-nav a",
            ".navbar-nav button",
            "nav a",
            "nav button",
            "[role='navigation'] a"
        ]
        
        nav_elements = []
        for selector in nav_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                nav_elements.extend(elements)
            except:
                pass
        
        print(f"    → Found {len(nav_elements)} navigation elements to check")
        
        # Hover over each to reveal dropdowns
        for element in nav_elements[:40]:
            try:
                if not element.is_displayed():
                    continue
                
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.2)
                actions.move_to_element(element).perform()
                time.sleep(1.5)
                
                # Extract dropdown links
                dropdown_selectors = [
                    ".navbar-nav.sub-navbar-nav a[href]",
                    "div.navbar-nav.sub-navbar-nav a[href]",
                    ".sub-navbar-nav a[href]",
                    ".dropdown-menu a[href]",
                    ".submenu a[href]"
                ]
                
                for dropdown_sel in dropdown_selectors:
                    try:
                        links = self.driver.find_elements(By.CSS_SELECTOR, dropdown_sel)
                        for link in links:
                            if link.is_displayed():
                                href = link.get_attribute('href')
                                if href and not href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                                    absolute_url = urljoin(current_url, href)
                                    discovered.add(absolute_url)
                                    # Mark as navigation link
                                    self.navigation_links.add(absolute_url)
                    except:
                        pass
            except:
                pass
        
        # Also check for visible sub-navbar-nav divs
        try:
            navbar_divs = self.driver.find_elements(By.CSS_SELECTOR, 
                "div.navbar-nav.sub-navbar-nav, .navbar-nav.sub-navbar-nav")
            for nav_div in navbar_divs:
                try:
                    links = nav_div.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute('href')
                        if href and not href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                            absolute_url = urljoin(current_url, href)
                            discovered.add(absolute_url)
                            self.navigation_links.add(absolute_url)
                except:
                    pass
        except:
            pass
        
        print(f"    ✓ Discovered {len(discovered)} navigation menu links (will use 'back_to_page')")
        return discovered
    
    def extract_page_links(self, soup, base_url):
        """Extract links from page content (non-navigation)"""
        links = []
        seen = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            absolute_url = urljoin(base_url, href)
            absolute_url = absolute_url.split('#')[0]
            
            if absolute_url in seen:
                continue
            seen.add(absolute_url)
            
            link_text = a_tag.get_text(strip=True) or "Link"
            
            # Check if this is a navigation link
            is_nav = absolute_url in self.navigation_links
            
            links.append({
                'url': absolute_url,
                'link_text': link_text,
                'is_navigation': is_nav
            })
        
        return links
    
    def create_safe_filename(self, url):
        """Create safe filename"""
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
        """Save page content"""
        filename = self.create_safe_filename(url)
        filepath = os.path.join(self.output_dir, 'pages', filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(url)
                f.write('\n\n')
                f.write(content)
            return filepath
        except:
            return None
    
    def should_crawl(self, url, current_depth):
        """Determine if URL should be crawled"""
        if self.is_pdf_domain(url):
            return False
        if self.is_main_domain(url):
            return True
        if self.is_include_domain(url):
            if url not in self.domain_depths:
                self.domain_depths[url] = current_depth
            return self.domain_depths[url] <= 2
        return False
    
    def crawl(self):
        """Main crawling function"""
        print("="*80)
        print("SMART SINGLE-THREADED CRAWLER")
        print("="*80)
        print(f"Main domains: {', '.join(self.main_domains)}")
        if self.include_domains:
            print(f"Include domains: {', '.join(self.include_domains)}")
        if self.pdf_domain:
            print(f"PDF domain: {self.pdf_domain}")
        print("="*80 + "\n")
        
        if not self.setup_driver():
            return
        
        queue = deque([(self.start_url, 0)])
        first_page = True
        
        try:
            while queue:
                url, depth = queue.popleft()
                
                if url in self.visited:
                    continue
                
                if not self.should_crawl(url, depth):
                    entity_name = self.get_entity_name_from_url(url)
                    self.graph.add_node(url, depth=depth, label=entity_name, 
                                       is_external=True, is_pdf=self.is_pdf_domain(url))
                    continue
                
                domain_type = "MAIN" if self.is_main_domain(url) else "INCLUDE"
                print(f"\n{'='*80}")
                print(f"[{domain_type}] Depth {depth}: {url}")
                print(f"{'='*80}")
                
                self.visited.add(url)
                self.graph.add_node(url, depth=depth, 
                                   label=self.get_entity_name_from_url(url),
                                   is_external=False, is_pdf=False)
                
                # Load page
                print("  → Loading page...")
                self.driver.get(url)
                self.wait_for_page_load()
                
                # Verify content loaded
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                if len(page_text.strip()) < 50 or "loading" in page_text.lower()[:100]:
                    print(f"  → Waiting for content...")
                    time.sleep(5)
                
                # On FIRST page only, discover navigation menus
                if first_page:
                    print("  → Discovering navigation menus (one-time)...")
                    nav_urls = self.discover_navigation_menus(url)
                    first_page = False
                else:
                    nav_urls = set()
                    print("  → Skipping navigation menu discovery (already done)")
                
                # Get HTML
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text
                for tag in soup(["script", "style", "nav", "footer", "header", "iframe"]):
                    tag.decompose()
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                
                # Double check content
                if len(text) < 100 or "loading" in text.lower()[:200]:
                    print(f"  → Content incomplete, waiting more...")
                    time.sleep(5)
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in soup(["script", "style", "nav", "footer", "header", "iframe"]):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                
                # Save content
                filepath = self.save_content(url, text)
                if filepath:
                    self.page_contents[url] = filepath
                    print(f"  ✓ Content saved ({len(text)} chars)")
                
                # Extract page links
                print(f"  → Extracting links...")
                links = self.extract_page_links(soup, url)
                
                # Count navigation vs new links
                nav_count = sum(1 for l in links if l['is_navigation'])
                new_count = len(links) - nav_count
                print(f"  ✓ Found {len(links)} links ({nav_count} navigation, {new_count} new)")
                
                # Process links
                queued = 0
                for link in links:
                    target_url = link['url']
                    link_text = link['link_text']
                    is_nav = link['is_navigation']
                    
                    # Use "back_to_page" for navigation links
                    relationship = "back_to_page" if is_nav else link_text
                    
                    # Add edge
                    self.graph.add_edge(url, target_url, relationship=relationship)
                    
                    # Handle different types
                    if self.is_pdf_domain(target_url):
                        entity_name = self.get_entity_name_from_url(target_url)
                        self.graph.add_node(target_url, label=entity_name, 
                                           is_external=True, is_pdf=True)
                    elif self.is_main_domain(target_url):
                        if target_url not in self.visited:
                            queue.append((target_url, depth + 1))
                            queued += 1
                    elif self.is_include_domain(target_url):
                        if depth < 2 and target_url not in self.visited:
                            queue.append((target_url, depth + 1))
                            queued += 1
                        elif depth >= 2:
                            entity_name = self.get_entity_name_from_url(target_url)
                            self.graph.add_node(target_url, label=entity_name, 
                                               is_external=True, is_pdf=False)
                    else:
                        entity_name = self.get_entity_name_from_url(target_url)
                        self.graph.add_node(target_url, label=entity_name, 
                                           is_external=True, is_pdf=False)
                
                print(f"  ✓ Queued {queued} new URLs")
                print(f"  → Progress: Queue={len(queue)}, Visited={len(self.visited)}")
                
                time.sleep(1)
        
        finally:
            if self.driver:
                self.driver.quit()
        
        print(f"\n{'='*80}")
        print("CRAWL COMPLETE")
        print(f"{'='*80}")
        print(f"Pages crawled: {len(self.visited)}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"Navigation links identified: {len(self.navigation_links)}")
        print(f"{'='*80}\n")
    
    def save_json(self):
        """Save graph to JSON"""
        data = {
            'metadata': {
                'start_url': self.start_url,
                'main_domains': self.main_domains,
                'include_domains': self.include_domains,
                'pdf_domain': self.pdf_domain,
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'crawled_pages': len(self.visited),
                'navigation_links': len(self.navigation_links)
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
        
        print(f"✓ Graph JSON saved: {filepath}")
        return data
    
    def create_interactive_graph(self, graph_data):
        """Create interactive HTML visualization"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; }
        .header h1 { margin: 0; }
        .stats { background: #16213e; padding: 15px; display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .stat { text-align: center; }
        .stat .value { font-size: 28px; font-weight: bold; color: #667eea; }
        .stat .label { font-size: 12px; color: #aaa; }
        .controls { background: #16213e; padding: 15px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .controls input, .controls select, .controls button { padding: 10px; border-radius: 5px; border: none; }
        .controls input { flex: 1; max-width: 300px; }
        .controls button { background: #667eea; color: white; cursor: pointer; font-weight: bold; }
        .controls button:hover { background: #5568d3; }
        #network { height: calc(100vh - 220px); background: #0f3460; }
        .info { position: fixed; top: 200px; right: 20px; background: rgba(22,33,62,0.95); padding: 20px; border-radius: 10px; max-width: 350px; display: none; box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-height: 60vh; overflow-y: auto; }
        .info h3 { color: #667eea; margin-bottom: 10px; }
        .info p { margin: 8px 0; font-size: 13px; }
        .close { float: right; cursor: pointer; font-size: 20px; color: #aaa; }
        .close:hover { color: white; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 11px; margin: 2px; font-weight: bold; }
        .badge-nav { background: #3498db; color: white; }
        .badge-external { background: #e74c3c; color: white; }
        .badge-pdf { background: #e67e22; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🕸️ Knowledge Graph</h1>
    </div>
    <div class="stats">
        <div class="stat"><div class="value" id="nodes">0</div><div class="label">Total Nodes</div></div>
        <div class="stat"><div class="value" id="edges">0</div><div class="label">Total Edges</div></div>
        <div class="stat"><div class="value" id="crawled">0</div><div class="label">Pages Crawled</div></div>
        <div class="stat"><div class="value" id="external">0</div><div class="label">External Entities</div></div>
    </div>
    <div class="controls">
        <input type="text" id="search" placeholder="🔍 Search nodes..." />
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
    <script>
        const data = GRAPH_DATA;
        
        document.getElementById('nodes').textContent = data.nodes.length;
        document.getElementById('edges').textContent = data.edges.length;
        document.getElementById('crawled').textContent = data.metadata.crawled_pages;
        document.getElementById('external').textContent = data.nodes.filter(n => n.is_external).length;
        
        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            title: n.url,
            color: n.is_external ? (n.is_pdf ? '#e67e22' : '#e74c3c') : '#667eea',
            shape: n.is_external ? 'box' : 'dot',
            size: n.is_external ? 15 : 25,
            font: { color: 'white', size: 12 },
            ...n
        }));
        
        const edges = data.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            arrows: 'to',
            color: { color: e.label === 'back_to_page' ? '#3498db' : '#666' },
            font: { size: 9, color: '#aaa' },
            dashes: e.label === 'back_to_page'
        }));
        
        const network = new vis.Network(
            document.getElementById('network'),
            { nodes, edges },
            {
                physics: {
                    barnesHut: { gravitationalConstant: -8000, springLength: 150 },
                    stabilization: { iterations: 200 }
                },
                interaction: { hover: true, navigationButtons: true, keyboard: true }
            }
        );
        
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
            if (node.is_external) badges += '<span class="badge badge-external">EXTERNAL</span>';
            if (node.is_pdf) badges += '<span class="badge badge-pdf">PDF</span>';
            if (navEdges.length > 0) badges += '<span class="badge badge-nav">' + navEdges.length + ' NAV LINKS</span>';
            
            document.getElementById('info-content').innerHTML = `
                <h3>${node.label}</h3>
                ${badges}
                <p><strong>URL:</strong><br><a href="${node.url}" target="_blank" style="color: #667eea; word-break: break-all;">${node.url}</a></p>
                <p><strong>Depth:</strong> ${node.depth}</p>
                <p><strong>Outgoing:</strong> ${outEdges.length} (${navEdges.length} navigation)</p>
                <p><strong>Incoming:</strong> ${inEdges.length}</p>
            `;
            document.getElementById('info').style.display = 'block';
        }
        
        function closeInfo() {
            document.getElementById('info').style.display = 'none';
        }
        
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
        
        document.getElementById('filter').onchange = e => {
            let filtered = nodes;
            if (e.target.value === 'crawled') filtered = nodes.filter(n => !n.is_external);
            if (e.target.value === 'external') filtered = nodes.filter(n => n.is_external);
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
        
        print(f"✓ Interactive graph HTML saved: {filepath}")
        print(f"  → Open this file in your browser to view the graph!")


def main():
    print("\n" + "="*80)
    print("SMART KNOWLEDGE GRAPH CRAWLER")
    print("="*80)
    print("Features:")
    print("  • Single-threaded (no parallel processing)")
    print("  • Discovers navigation menus ONCE on first page")
    print("  • Uses 'back_to_page' relationship for repeated nav links")
    print("  • Creates interactive graph visualization")
    print("="*80 + "\n")
    
    start_url = input("Enter starting URL: ").strip()
    if not start_url:
        return
    
    print("\nMain domains (comma-separated):")
    main_domains = input("Main: ").strip()
    main_domains = [d.strip() for d in main_domains.split(',') if d.strip()]
    
    print("\nInclude domains (optional, comma-separated):")
    include_domains = input("Include: ").strip()
    include_domains = [d.strip() for d in include_domains.split(',') if d.strip()] if include_domains else []
    
    print("\nPDF domain (optional):")
    pdf_domain = input("PDF: ").strip()
    
    if not main_domains:
        print("Error: Need at least one main domain")
        return
    
    print(f"\n{'='*80}")
    print("Configuration:")
    print(f"  Start: {start_url}")
    print(f"  Main: {', '.join(main_domains)}")
    if include_domains:
        print(f"  Include: {', '.join(include_domains)}")
    if pdf_domain:
        print(f"  PDF: {pdf_domain}")
    print(f"{'='*80}\n")
    
    proceed = input("Start crawling? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Cancelled")
        return
    
    # Create and run crawler
    crawler = SmartKGCrawler(start_url, main_domains, include_domains, pdf_domain)
    crawler.crawl()
    
    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80 + "\n")
    
    graph_data = crawler.save_json()
    crawler.create_interactive_graph(graph_data)
    
    print("\n" + "="*80)
    print("✓ ALL COMPLETE!")
    print("="*80)
    print(f"\nOutputs in '{crawler.output_dir}/':")
    print(f"  📊 interactive_graph.html ← OPEN THIS to view the graph!")
    print(f"  📄 graph.json ← Complete graph data")
    print(f"  📁 pages/*.txt ← Page contents")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
