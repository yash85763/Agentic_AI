from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional
import asyncio
import json
import os
from dataclasses import dataclass
import re

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

class DynamicTreeCrawler:
    """Enhanced crawler with dropdown handling and media placeholders"""
    
    def __init__(
        self,
        start_url: str,
        max_depth: int = 3,
        same_domain_only: bool = True,
        max_pages: int = 100,
        wait_for_dynamic_content: int = 2000  # milliseconds
    ):
        self.start_url = start_url
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.max_pages = max_pages
        self.wait_time = wait_for_dynamic_content
        self.visited: Set[str] = set()
        self.base_domain = urlparse(start_url).netloc
        
        # Media file extensions to replace with placeholders
        self.video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']
        self.audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
        self.pdf_extensions = ['.pdf']
        
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
            ['.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.rar']
        )
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    async def handle_dropdowns(self, page: Page):
        """Expand all dropdowns and collapsible elements"""
        try:
            # Common dropdown selectors
            dropdown_selectors = [
                'select',  # Standard select dropdowns
                '[role="button"][aria-expanded="false"]',  # ARIA dropdowns
                '.dropdown-toggle',
                'button[data-toggle="dropdown"]',
                '.accordion-button.collapsed',  # Bootstrap accordion
                'details:not([open])',  # HTML5 details element
                '[class*="dropdown"]:not(.open)',
                '[class*="collapse"]:not(.show)',
            ]
            
            for selector in dropdown_selectors:
                elements = await page.locator(selector).all()
                for element in elements:
                    try:
                        # Try to click to expand
                        await element.click(timeout=1000)
                        await page.wait_for_timeout(300)  # Wait for animation
                    except:
                        pass
            
            # Handle select dropdowns - extract all options
            selects = await page.locator('select').all()
            for select in selects:
                try:
                    options = await select.locator('option').all()
                    for option in options:
                        await option.scroll_into_view_if_needed()
                except:
                    pass
            
            # Scroll to load lazy content
            await self.scroll_page(page)
            
        except Exception as e:
            print(f"Error handling dropdowns: {e}")
    
    async def scroll_page(self, page: Page):
        """Scroll through page to load lazy content"""
        try:
            # Get page height
            height = await page.evaluate('document.body.scrollHeight')
            viewport_height = await page.evaluate('window.innerHeight')
            
            # Scroll in steps
            current = 0
            while current < height:
                await page.evaluate(f'window.scrollTo(0, {current})')
                await page.wait_for_timeout(200)
                current += viewport_height
            
            # Scroll back to top
            await page.evaluate('window.scrollTo(0, 0)')
            await page.wait_for_timeout(500)
            
        except Exception as e:
            print(f"Error scrolling page: {e}")
    
    def extract_media_placeholders(self, html: str, base_url: str) -> List[Dict]:
        """Extract media elements and create placeholders"""
        soup = BeautifulSoup(html, 'html.parser')
        placeholders = []
        
        # Extract videos
        for video in soup.find_all(['video', 'iframe']):
            src = video.get('src', '') or video.get('data-src', '')
            if src:
                full_url = urljoin(base_url, src)
                placeholder = {
                    'type': 'VIDEO',
                    'src': full_url,
                    'alt': video.get('title', 'Video content'),
                    'attributes': {
                        'width': video.get('width', ''),
                        'height': video.get('height', ''),
                        'controls': video.get('controls', False)
                    }
                }
                placeholders.append(placeholder)
                # Replace with placeholder text
                video.replace_with(f"[VIDEO_PLACEHOLDER: {full_url}]")
        
        # Check for embedded video platforms
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if any(platform in src.lower() for platform in ['youtube', 'vimeo', 'dailymotion', 'wistia']):
                placeholder = {
                    'type': 'EMBEDDED_VIDEO',
                    'src': src,
                    'platform': self._detect_platform(src),
                    'alt': iframe.get('title', 'Embedded video')
                }
                placeholders.append(placeholder)
                iframe.replace_with(f"[EMBEDDED_VIDEO_PLACEHOLDER: {src}]")
        
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
                    'alt': audio.get('title', 'Audio content'),
                    'attributes': {
                        'controls': audio.get('controls', False)
                    }
                }
                placeholders.append(placeholder)
                audio.replace_with(f"[AUDIO_PLACEHOLDER: {full_url}]")
        
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
                link.replace_with(f"[PDF_PLACEHOLDER: {full_url} - {placeholder['alt']}]")
        
        # Extract video/audio file links
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
                link.replace_with(f"[VIDEO_FILE_PLACEHOLDER: {full_url}]")
            
            elif any(href.lower().endswith(ext) for ext in self.audio_extensions):
                full_url = urljoin(base_url, href)
                placeholder = {
                    'type': 'AUDIO_FILE',
                    'src': full_url,
                    'alt': link.get_text().strip() or 'Audio file',
                    'filename': os.path.basename(urlparse(full_url).path)
                }
                placeholders.append(placeholder)
                link.replace_with(f"[AUDIO_FILE_PLACEHOLDER: {full_url}]")
        
        return placeholders, str(soup)
    
    def _detect_platform(self, url: str) -> str:
        """Detect video platform from URL"""
        url_lower = url.lower()
        if 'youtube' in url_lower:
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
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Extract text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = ' '.join(chunk for chunk in chunks if chunk)
        
        return title, content
    
    async def crawl_page(
        self,
        browser: Browser,
        url: str,
        depth: int
    ) -> Optional[PageNode]:
        """Crawl a single page and return a PageNode"""
        
        if depth > self.max_depth or len(self.visited) >= self.max_pages:
            return None
        
        if url in self.visited:
            return None
        
        print(f"{'  ' * depth}Crawling: {url} (depth: {depth})")
        self.visited.add(url)
        
        try:
            # Create new page
            page = await browser.new_page()
            
            # Navigate to URL
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for dynamic content
            await page.wait_for_timeout(self.wait_time)
            
            # Handle dropdowns and dynamic elements
            await self.handle_dropdowns(page)
            
            # Get final HTML after all interactions
            html = await page.content()
            
            # Extract media placeholders and clean HTML
            media_placeholders, cleaned_html = self.extract_media_placeholders(html, url)
            
            # Extract content and links
            title, content = self.extract_content(cleaned_html)
            child_urls = self.extract_links(html, url)
            
            # Close page
            await page.close()
            
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
                        child_node = await self.crawl_page(browser, child_url, depth + 1)
                        if child_node:
                            node.children.append(child_node)
            
            return node
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    async def crawl(self) -> Optional[PageNode]:
        """Start crawling from the root URL"""
        async with async_playwright() as p:
            # Launch browser (headless=True for production)
            browser = await p.chromium.launch(headless=True)
            
            try:
                root = await self.crawl_page(browser, self.start_url, depth=0)
                return root
            finally:
                await browser.close()
    
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
                file.write(f"{' ' * (indent * 2 + 6)}- {media['type']}: {media['src'][:60]}...\n")
        
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
                file.write("\n")
            file.write("\n")
        
        file.write("CONTENT:\n")
        file.write("-" * 40 + "\n")
        file.write(node.content)
        file.write("\n\n")
        
        for child in node.children:
            self._write_content_with_media(file, child, visited_urls)

# Usage example
async def main():
    crawler = DynamicTreeCrawler(
        start_url="https://example.com",  # Replace with your URL
        max_depth=2,
        same_domain_only=True,
        max_pages=30,
        wait_for_dynamic_content=2000  # Wait 2 seconds for dropdowns
    )
    
    print("Starting enhanced tree crawl with dropdown handling...")
    root = await crawler.crawl()
    
    if root:
        print(f"\nCrawl completed!")
        print(f"Total pages visited: {len(crawler.visited)}")
        
        # Save the tree
        crawler.save_tree(root, "crawled_tree_enhanced")
        
        # Print summary
        media_count = sum(len(node.media_placeholders) for node in flatten_tree(root))
        print(f"Total media items found: {media_count}")

def flatten_tree(node: PageNode, result: List = None) -> List[PageNode]:
    """Flatten tree to list"""
    if result is None:
        result = []
    result.append(node)
    for child in node.children:
        flatten_tree(child, result)
    return result

if __name__ == "__main__":
    asyncio.run(main())