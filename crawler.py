from langchain_community.document_loaders import RecursiveUrlLoader
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional
import time
import json
import os
from dataclasses import dataclass
from collections import deque

@dataclass
class PageNode:
    """Represents a node in the content tree"""
    url: str
    title: str
    content: str
    depth: int
    children: List['PageNode']
    metadata: Dict
    media_placeholders: List[Dict]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'depth': self.depth,
            'metadata': self.metadata,
            'media_placeholders': self.media_placeholders,
            'children': [child.to_dict() for child in self.children]
        }

class EnhancedTreeCrawler:
    """Tree crawler with Selenium dropdown handling and media placeholders"""
    
    def __init__(
        self,
        start_url: str,
        max_depth: int = 3,
        same_domain_only: bool = True,
        max_pages: int = 100,
        use_selenium: bool = True,
        headless: bool = True
    ):
        self.start_url = start_url
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.max_pages = max_pages
        self.use_selenium = use_selenium
        self.headless = headless
        self.visited: Set[str] = set()
        self.base_domain = urlparse(start_url).netloc
        
        # Media file extensions
        self.video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']
        self.audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
        self.pdf_extensions = ['.pdf']
        
        # Initialize Selenium driver if needed
        self.driver = None
        if self.use_selenium:
            self.driver = self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Use webdriver_manager to auto-download ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        parsed = urlparse(url)
        
        if parsed.scheme not in ['http', 'https']:
            return False
        
        if self.same_domain_only and parsed.netloc != self.base_domain:
            return False
        
        # Skip media and document files
        skip_extensions = (
            self.video_extensions + 
            self.audio_extensions + 
            self.pdf_extensions +
            ['.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.rar', '.doc', '.docx']
        )
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def handle_dropdowns_selenium(self, driver):
        """Expand dropdowns using Selenium"""
        try:
            # Scroll to load lazy content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            
            # Common dropdown selectors
            dropdown_selectors = [
                "//select",  # Standard select dropdowns
                "//*[@role='button' and @aria-expanded='false']",  # ARIA dropdowns
                "//button[contains(@class, 'dropdown')]",
                "//button[@data-toggle='dropdown']",
                "//button[contains(@class, 'accordion') and contains(@class, 'collapsed')]",
                "//details[not(@open)]",  # HTML5 details
                "//*[contains(@class, 'collapse') and not(contains(@class, 'show'))]",
            ]
            
            for selector in dropdown_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    for element in elements[:10]:  # Limit to avoid too many clicks
                        try:
                            # Scroll element into view
                            driver.execute_script("arguments[0].scrollIntoView(true);", element)
                            time.sleep(0.2)
                            
                            # Try to click
                            element.click()
                            time.sleep(0.3)
                        except:
                            pass
                except:
                    pass
            
            # Handle select dropdowns specifically
            try:
                selects = driver.find_elements(By.TAG_NAME, "select")
                for select in selects:
                    try:
                        driver.execute_script("arguments[0].scrollIntoView(true);", select)
                    except:
                        pass
            except:
                pass
            
            # Final scroll to capture everything
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error handling dropdowns: {e}")
    
    def fetch_page_with_selenium(self, url: str) -> Optional[str]:
        """Fetch page using Selenium with dropdown handling"""
        try:
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Wait for dynamic content
            
            # Handle dropdowns
            self.handle_dropdowns_selenium(self.driver)
            
            # Get page source
            html = self.driver.page_source
            return html
            
        except Exception as e:
            print(f"Error fetching {url} with Selenium: {e}")
            return None
    
    def extract_media_placeholders(self, html: str, base_url: str) -> tuple[List[Dict], str]:
        """Extract media elements and create placeholders"""
        soup = BeautifulSoup(html, 'html.parser')
        placeholders = []
        
        # Extract videos
        for video in soup.find_all('video'):
            src = video.get('src', '') or video.get('data-src', '')
            if not src:
                source = video.find('source')
                src = source.get('src', '') if source else ''
            
            if src:
                full_url = urljoin(base_url, src)
                placeholder = {
                    'type': 'VIDEO',
                    'src': full_url,
                    'alt': video.get('title', '') or video.get('aria-label', '') or 'Video content',
                    'attributes': {
                        'width': video.get('width', ''),
                        'height': video.get('height', ''),
                        'controls': video.has_attr('controls')
                    }
                }
                placeholders.append(placeholder)
                video.replace_with(f"[VIDEO_PLACEHOLDER: {placeholder['alt']} - {full_url}]")
        
        # Extract iframes (embedded videos)
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if src:
                video_platforms = ['youtube', 'vimeo', 'dailymotion', 'wistia', 'embed']
                if any(platform in src.lower() for platform in video_platforms):
                    placeholder = {
                        'type': 'EMBEDDED_VIDEO',
                        'src': src,
                        'platform': self._detect_platform(src),
                        'alt': iframe.get('title', '') or 'Embedded video'
                    }
                    placeholders.append(placeholder)
                    iframe.replace_with(f"[EMBEDDED_VIDEO_PLACEHOLDER: {placeholder['platform']} - {src}]")
        
        # Extract audio
        for audio in soup.find_all('audio'):
            src = audio.get('src', '')
            if not src:
                source = audio.find('source')
                src = source.get('src', '') if source else ''
            
            if src:
                full_url = urljoin(base_url, src)
                placeholder = {
                    'type': 'AUDIO',
                    'src': full_url,
                    'alt': audio.get('title', '') or 'Audio content',
                    'attributes': {
                        'controls': audio.has_attr('controls')
                    }
                }
                placeholders.append(placeholder)
                audio.replace_with(f"[AUDIO_PLACEHOLDER: {placeholder['alt']} - {full_url}]")
        
        # Extract PDF links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(href.lower().endswith(ext) for ext in self.pdf_extensions):
                full_url = urljoin(base_url, href)
                placeholder = {
                    'type': 'PDF',
                    'src': full_url,
                    'alt': link.get_text().strip() or 'PDF document',
                    'filename': os.path.basename(urlparse(full_url).path)
                }
                placeholders.append(placeholder)
                link.replace_with(f"[PDF_PLACEHOLDER: {placeholder['alt']} - {full_url}]")
        
        # Extract video file links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(href.lower().endswith(ext) for ext in self.video_extensions):
                full_url = urljoin(base_url, href)
                placeholder = {
                    'type': 'VIDEO_FILE',
                    'src': full_url,
                    'alt': link.get_text().strip() or 'Video file',
                    'filename': os.path.basename(urlparse(full_url).path)
                }
                placeholders.append(placeholder)
                link.replace_with(f"[VIDEO_FILE_PLACEHOLDER: {placeholder['filename']} - {full_url}]")
        
        # Extract audio file links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(href.lower().endswith(ext) for ext in self.audio_extensions):
                full_url = urljoin(base_url, href)
                placeholder = {
                    'type': 'AUDIO_FILE',
                    'src': full_url,
                    'alt': link.get_text().strip() or 'Audio file',
                    'filename': os.path.basename(urlparse(full_url).path)
                }
                placeholders.append(placeholder)
                link.replace_with(f"[AUDIO_FILE_PLACEHOLDER: {placeholder['filename']} - {full_url}]")
        
        return placeholders, str(soup)
    
    def _detect_platform(self, url: str) -> str:
        """Detect video platform from URL"""
        url_lower = url.lower()
        if 'youtube' in url_lower or 'youtu.be' in url_lower:
            return 'YouTube'
        elif 'vimeo' in url_lower:
            return 'Vimeo'
        elif 'dailymotion' in url_lower:
            return 'Dailymotion'
        elif 'wistia' in url_lower:
            return 'Wistia'
        return 'Unknown'
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all valid links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            full_url = full_url.split('#')[0]  # Remove fragments
            
            if self.is_valid_url(full_url) and full_url not in self.visited:
                links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def extract_content(self, html: str) -> tuple[str, str]:
        """Extract title and main content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript']):
            element.decompose()
        
        # Extract text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = ' '.join(chunk for chunk in chunks if chunk)
        
        return title, content
    
    def crawl_page(self, url: str, depth: int) -> Optional[PageNode]:
        """Crawl a single page and return a PageNode"""
        
        if depth > self.max_depth or len(self.visited) >= self.max_pages:
            return None
        
        if url in self.visited:
            return None
        
        print(f"{'  ' * depth}Crawling: {url} (depth: {depth})")
        self.visited.add(url)
        
        try:
            # Fetch page
            if self.use_selenium and self.driver:
                html = self.fetch_page_with_selenium(url)
            else:
                # Fallback to basic fetch (without dropdown handling)
                import requests
                response = requests.get(url, timeout=10)
                html = response.text
            
            if not html:
                return None
            
            # Extract media placeholders and clean HTML
            media_placeholders, cleaned_html = self.extract_media_placeholders(html, url)
            
            # Extract content and links
            title, content = self.extract_content(cleaned_html)
            child_urls = self.extract_links(html, url)
            
            # Create node
            node = PageNode(
                url=url,
                title=title,
                content=content[:5000],  # Limit content length
                depth=depth,
                children=[],
                metadata={
                    'content_length': len(content),
                    'num_links': len(child_urls),
                    'num_media_items': len(media_placeholders)
                },
                media_placeholders=media_placeholders
            )
            
            # Recursively crawl children
            if depth < self.max_depth and len(self.visited) < self.max_pages:
                for child_url in child_urls[:5]:  # Limit children per page
                    if child_url not in self.visited:
                        child_node = self.crawl_page(child_url, depth + 1)
                        if child_node:
                            node.children.append(child_node)
            
            return node
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    def crawl(self) -> Optional[PageNode]:
        """Start crawling from the root URL"""
        try:
            root = self.crawl_page(self.start_url, depth=0)
            return root
        finally:
            if self.driver:
                self.driver.quit()
    
    def save_tree(self, root: PageNode, output_dir: str = "crawled_tree"):
        """Save the tree structure to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, "content_tree.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved tree to: {json_path}")
        
        # Save media inventory
        media_path = os.path.join(output_dir, "media_inventory.json")
        media_items = self._collect_media(root)
        with open(media_path, 'w', encoding='utf-8') as f:
            json.dump(media_items, f, indent=2, ensure_ascii=False)
        print(f"Saved media inventory to: {media_path}")
        
        # Save as readable text tree
        text_path = os.path.join(output_dir, "tree_structure.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            self._write_tree_text(f, root, 0)
        print(f"Saved tree structure to: {text_path}")
        
        # Save content with placeholders
        content_path = os.path.join(output_dir, "content_with_placeholders.txt")
        with open(content_path, 'w', encoding='utf-8') as f:
            self._write_content_with_media(f, root)
        print(f"Saved content with placeholders to: {content_path}")
    
    def _collect_media(self, node: PageNode, media_list: List = None) -> List[Dict]:
        """Recursively collect all media items"""
        if media_list is None:
            media_list = []
        
        for media in node.media_placeholders:
            media_list.append({
                **media,
                'page_url': node.url,
                'page_title': node.title,
                'page_depth': node.depth
            })
        
        for child in node.children:
            self._collect_media(child, media_list)
        
        return media_list
    
    def _write_tree_text(self, file, node: PageNode, indent: int):
        """Recursively write tree structure to text file"""
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        file.write(f"{prefix}{node.title or 'No Title'}\n")
        file.write(f"{' ' * (indent * 2 + 3)}URL: {node.url}\n")
        file.write(f"{' ' * (indent * 2 + 3)}Depth: {node.depth}, Children: {len(node.children)}\n")
        
        if node.media_placeholders:
            file.write(f"{' ' * (indent * 2 + 3)}Media items: {len(node.media_placeholders)}\n")
            for media in node.media_placeholders:
                file.write(f"{' ' * (indent * 2 + 6)}- {media['type']}: {media.get('alt', 'N/A')}\n")
        
        file.write(f"{' ' * (indent * 2 + 3)}Content preview: {node.content[:100]}...\n\n")
        
        for child in node.children:
            self._write_tree_text(file, child, indent + 1)
    
    def _write_content_with_media(self, file, node: PageNode, visited_urls: Set = None):
        """Write content with media placeholders"""
        if visited_urls is None:
            visited_urls = set()
        
        if node.url in visited_urls:
            return
        
        visited_urls.add(node.url)
        
        file.write(f"\n{'='*80}\n")
        file.write(f"Page: {node.title}\n")
        file.write(f"URL: {node.url}\n")
        file.write(f"Depth: {node.depth}\n")
        file.write(f"{'='*80}\n\n")
        
        if node.media_placeholders:
            file.write("MEDIA ITEMS:\n")
            file.write("-" * 40 + "\n")
            for i, media in enumerate(node.media_placeholders, 1):
                file.write(f"{i}. [{media['type']}] {media['alt']}\n")
                file.write(f"   Source: {media['src']}\n")
                if 'filename' in media:
                    file.write(f"   Filename: {media['filename']}\n")
                if 'platform' in media:
                    file.write(f"   Platform: {media['platform']}\n")
                file.write("\n")
            file.write("\n")
        
        file.write("CONTENT:\n")
        file.write("-" * 40 + "\n")
        file.write(node.content)
        file.write("\n\n")
        
        for child in node.children:
            self._write_content_with_media(file, child, visited_urls)


# Alternative: Using LangChain without dropdown handling
class SimpleLangChainCrawler:
    """Simpler version using only LangChain (no dropdown handling)"""
    
    def __init__(self, start_url: str, max_depth: int = 2):
        self.start_url = start_url
        self.max_depth = max_depth
        self.visited = set()
    
    def extract_text_with_placeholders(self, html: str) -> str:
        """Extract text and replace media with placeholders"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Replace media elements
        for video in soup.find_all(['video', 'iframe']):
            src = video.get('src', '')
            video.replace_with(f"[VIDEO_PLACEHOLDER: {src}]")
        
        for audio in soup.find_all('audio'):
            src = audio.get('src', '')
            audio.replace_with(f"[AUDIO_PLACEHOLDER: {src}]")
        
        for link in soup.find_all('a', href=True):
            if link['href'].lower().endswith('.pdf'):
                link.replace_with(f"[PDF_PLACEHOLDER: {link.get_text()}]")
        
        # Remove scripts and styles
        for element in soup(['script', 'style']):
            element.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        return ' '.join(line for line in lines if line)
    
    def crawl(self):
        """Crawl using LangChain"""
        loader = RecursiveUrlLoader(
            url=self.start_url,
            max_depth=self.max_depth,
            extractor=self.extract_text_with_placeholders,
            prevent_outside=True
        )
        
        docs = loader.load()
        return docs


# Usage example
def main():
    # Option 1: Full featured with Selenium (handles dropdowns)
    print("Starting crawl with Selenium (dropdown handling enabled)...\n")
    crawler = EnhancedTreeCrawler(
        start_url="https://example.com",  # Replace with your URL
        max_depth=2,
        same_domain_only=True,
        max_pages=30,
        use_selenium=True,
        headless=True  # Set to False to see browser
    )
    
    root = crawler.crawl()
    
    if root:
        print(f"\nCrawl completed!")
        print(f"Total pages visited: {len(crawler.visited)}")
        
        # Save the tree
        crawler.save_tree(root, "crawled_tree_selenium")
        
        # Print summary
        media_count = sum(len(node.media_placeholders) for node in flatten_tree(root))
        print(f"Total media items found: {media_count}")
    
    # Option 2: Simple LangChain version (no dropdown handling)
    # print("\nUsing simple LangChain crawler...\n")
    # simple_crawler = SimpleLangChainCrawler(
    #     start_url="https://example.com",
    #     max_depth=2
    # )
    # docs = simple_crawler.crawl()
    # print(f"Crawled {len(docs)} pages")

def flatten_tree(node: PageNode, result: List = None) -> List[PageNode]:
    """Flatten tree to list"""
    if result is None:
        result = []
    result.append(node)
    for child in node.children:
        flatten_tree(child, result)
    return result
    
# Add these functions after your main() function

def print_crawl_summary(root: PageNode):
    """Print a summary of the crawl"""
    nodes = flatten_tree(root)
    
    print("\n" + "="*80)
    print("CRAWL SUMMARY")
    print("="*80)
    print(f"Total pages crawled: {len(nodes)}")
    print(f"Max depth reached: {max(node.depth for node in nodes)}")
    
    # Media summary
    total_videos = sum(1 for node in nodes for m in node.media_placeholders if 'VIDEO' in m['type'])
    total_audio = sum(1 for node in nodes for m in node.media_placeholders if 'AUDIO' in m['type'])
    total_pdfs = sum(1 for node in nodes for m in node.media_placeholders if m['type'] == 'PDF')
    
    print(f"\nMedia found:")
    print(f"  - Videos: {total_videos}")
    print(f"  - Audio: {total_audio}")
    print(f"  - PDFs: {total_pdfs}")
    
    print("\n" + "-"*80)
    print("TREE STRUCTURE:")
    print("-"*80)
    print_tree_visual(root, "", True)
    
    print("\n" + "-"*80)
    print("ALL PAGES (by depth):")
    print("-"*80)
    for node in nodes:
        print(f"[Depth {node.depth}] {node.url}")
        print(f"  Title: {node.title}")
        print(f"  Content length: {node.metadata['content_length']} chars")
        print(f"  Children: {len(node.children)}, Media: {len(node.media_placeholders)}")
        print()

def print_tree_visual(node: PageNode, prefix: str = "", is_last: bool = True):
    """Print tree structure visually"""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{node.title[:60] or node.url[:60]}")
    print(f"{prefix}{'    ' if is_last else '│   '}    URL: {node.url}")
    print(f"{prefix}{'    ' if is_last else '│   '}    Media: {len(node.media_placeholders)}, Children: {len(node.children)}")
    
    extension = "    " if is_last else "│   "
    for i, child in enumerate(node.children):
        print_tree_visual(child, prefix + extension, i == len(node.children) - 1)

# Update your main() function to use these:
def main():
    crawler = EnhancedTreeCrawler(
        start_url="https://example.com",  # Your URL
        max_depth=2,
        same_domain_only=True,
        max_pages=30,
        use_selenium=True,
        headless=True
    )
    
    root = crawler.crawl()
    
    if root:
        # Print summary to console
        print_crawl_summary(root)
        
        # Save files
        crawler.save_tree(root, "crawled_tree_selenium")
        
        print("\n" + "="*80)
        print("Files saved to: crawled_tree_selenium/")
        print("  - content_tree.json (full tree structure)")
        print("  - tree_structure.txt (readable tree)")
        print("  - content_with_placeholders.txt (all content)")
        print("  - media_inventory.json (all media items)")
        print("="*80)

if __name__ == "__main__":
    main()