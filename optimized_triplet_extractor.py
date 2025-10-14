from openai import OpenAI
import os
from typing import List, Dict, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from bs4 import BeautifulSoup
import re

class OptimizedTripletExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        print(f"✓ Optimized Triplet Extractor initialized with model: {model}")
    
    def extract_link_context(self, page_html: str, target_url: str, 
                            context_chars: int = 400) -> Optional[str]:
        """
        Extract context around a specific link in the page
        
        Args:
            page_html: HTML content of the page
            target_url: URL to find
            context_chars: Characters to extract around link
            
        Returns:
            Context string or None
        """
        try:
            soup = BeautifulSoup(page_html, 'html.parser')
            
            # Find the link
            link = soup.find('a', href=lambda href: href and target_url in href)
            if not link:
                return None
            
            # Get parent paragraph or div
            parent = link.find_parent(['p', 'div', 'section', 'article'])
            if not parent:
                parent = link.parent
            
            # Get text content
            text = parent.get_text(separator=' ', strip=True)
            
            # Find link position in text
            link_text = link.get_text(strip=True)
            link_pos = text.find(link_text)
            
            if link_pos == -1:
                return text[:context_chars]
            
            # Extract context around link
            start = max(0, link_pos - context_chars // 2)
            end = min(len(text), link_pos + len(link_text) + context_chars // 2)
            
            context = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            return context
            
        except Exception as e:
            print(f"    Warning: Could not extract context: {e}")
            return None
    
    def extract_triplets_batch(self, page_pairs: List[Dict], 
                              batch_size: int = 5) -> List[Dict]:
        """
        Extract triplets for multiple page pairs in one LLM call
        
        Args:
            page_pairs: List of {source_page, target_page, link_text, context}
            batch_size: Number of pairs to process per LLM call
            
        Returns:
            List of all triplets
        """
        all_triplets = []
        
        # Process in batches
        for i in range(0, len(page_pairs), batch_size):
            batch = page_pairs[i:i+batch_size]
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(page_pairs)-1)//batch_size + 1} ({len(batch)} pairs)")
            
            # Build batch prompt
            pairs_text = ""
            for idx, pair in enumerate(batch, 1):
                pairs_text += f"""
PAIR {idx}:
Source Page: {pair['source_page']['title']}
Source Context: {pair.get('source_context', pair['source_page']['text'][:500])}

Target Page: {pair['target_page']['title']}
Target Context: {pair.get('target_context', pair['target_page']['text'][:500])}

Link Text: "{pair['link_text']}"
Link Context: {pair.get('link_context', 'N/A')}

---
"""
            
            prompt = f"""You are extracting semantic triplets from multiple linked web pages.

{pairs_text}

TASK:
For EACH pair above, extract 1-3 semantic triplets.

RULES:
1. Triplet format: (source_entity, relationship, target_entity)
2. Use semantic relationships: "explains", "prerequisite_for", "implements", "extends", etc.
3. NO generic relationships like "links_to"
4. Focus on content relationships, not hyperlinks
5. If no meaningful relationship, return empty list for that pair

OUTPUT FORMAT (JSON):
{{
  "results": [
    {{
      "pair_id": 1,
      "triplets": [
        {{
          "source_entity": "concept from source",
          "relationship": "semantic_relationship",
          "target_entity": "concept from target",
          "confidence": 0.9,
          "explanation": "brief explanation"
        }}
      ]
    }},
    ...
  ]
}}

Extract triplets now:"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You extract semantic knowledge graph triplets efficiently."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Process results
                for pair_result in result.get('results', []):
                    pair_id = pair_result.get('pair_id', 0)
                    if pair_id > 0 and pair_id <= len(batch):
                        pair_data = batch[pair_id - 1]
                        
                        for triplet in pair_result.get('triplets', []):
                            triplet['source_url'] = pair_data['source_page']['url']
                            triplet['target_url'] = pair_data['target_page']['url']
                            triplet['link_text'] = pair_data['link_text']
                            all_triplets.append(triplet)
                
                print(f"    ✓ Extracted {len([t for r in result.get('results', []) for t in r.get('triplets', [])])} triplets from batch")
                
            except Exception as e:
                print(f"    ✗ Error processing batch: {e}")
            
            # Rate limiting
            time.sleep(0.5)
        
        return all_triplets
    
    def extract_entities_batch(self, pages: List[Dict], 
                              batch_size: int = 10) -> Dict[str, List[str]]:
        """
        Extract entities from multiple pages in one call
        
        Args:
            pages: List of page dictionaries
            batch_size: Pages per batch
            
        Returns:
            Dictionary mapping page URLs to entity lists
        """
        all_entities = {}
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i+batch_size]
            
            print(f"  Extracting entities from batch {i//batch_size + 1}/{(len(pages)-1)//batch_size + 1} ({len(batch)} pages)")
            
            pages_text = ""
            for idx, page in enumerate(batch, 1):
                text = page.get('text', '')[:1000]  # First 1000 chars
                pages_text += f"""
PAGE {idx}:
Title: {page.get('title', 'Unknown')}
URL: {page.get('url', '')}
Content: {text}

---
"""
            
            prompt = f"""Extract key entities from these web pages.

{pages_text}

TASK:
For EACH page, extract 5-15 key entities (concepts, topics, technologies, terms).

RULES:
1. Extract nouns and noun phrases
2. Focus on technical terms, concepts, and important topics
3. Skip generic words like "page", "website", "documentation"
4. Normalize similar entities (e.g., "Python programming" → "Python")

OUTPUT FORMAT (JSON):
{{
  "results": [
    {{
      "page_id": 1,
      "entities": ["Entity1", "Entity2", ...]
    }},
    ...
  ]
}}

Extract entities:"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You extract entities from web content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                for page_result in result.get('results', []):
                    page_id = page_result.get('page_id', 0)
                    if page_id > 0 and page_id <= len(batch):
                        page_url = batch[page_id - 1]['url']
                        all_entities[page_url] = page_result.get('entities', [])
                
                print(f"    ✓ Extracted entities from {len(result.get('results', []))} pages")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            time.sleep(0.5)
        
        return all_entities
    
    def extract_entity_relationships(self, entity_pairs: List[Dict],
                                    batch_size: int = 10) -> List[Dict]:
        """
        Extract relationships between known entities
        
        Args:
            entity_pairs: List of {source_entity, target_entity, context}
            batch_size: Pairs per batch
            
        Returns:
            List of relationship triplets
        """
        all_relationships = []
        
        for i in range(0, len(entity_pairs), batch_size):
            batch = entity_pairs[i:i+batch_size]
            
            pairs_text = ""
            for idx, pair in enumerate(batch, 1):
                pairs_text += f"""
PAIR {idx}:
Entity 1: {pair['source_entity']}
Entity 2: {pair['target_entity']}
Context: {pair.get('context', 'These entities appear on linked pages')}

---
"""
            
            prompt = f"""Determine relationships between entity pairs.

{pairs_text}

TASK:
For each pair, determine if there's a semantic relationship.

Valid relationships:
- is_a, part_of, uses, requires, implements, extends
- prerequisite_for, explains, documents, demonstrates
- similar_to, related_to, contrasts_with

OUTPUT FORMAT (JSON):
{{
  "results": [
    {{
      "pair_id": 1,
      "has_relationship": true/false,
      "relationship": "relationship_type",
      "confidence": 0.9,
      "explanation": "why"
    }},
    ...
  ]
}}

Analyze relationships:"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You determine semantic relationships between entities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                for pair_result in result.get('results', []):
                    if pair_result.get('has_relationship', False):
                        pair_id = pair_result.get('pair_id', 0)
                        if pair_id > 0 and pair_id <= len(batch):
                            pair_data = batch[pair_id - 1]
                            
                            all_relationships.append({
                                'source_entity': pair_data['source_entity'],
                                'relationship': pair_result.get('relationship', 'related_to'),
                                'target_entity': pair_data['target_entity'],
                                'confidence': pair_result.get('confidence', 0.5),
                                'explanation': pair_result.get('explanation', '')
                            })
                
                print(f"    ✓ Found {len([r for r in result.get('results', []) if r.get('has_relationship')])} relationships")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            time.sleep(0.5)
        
        return all_relationships
