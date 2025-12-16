No problem! We can use **NetworkX** (pure Python graph library) with **in-memory vector search**. No external database needed!

## Install dependencies

```bash
pip install networkx numpy requests
```

## Updated graph_builder.py

```python
# graph_builder.py
import re
import requests
import networkx as nx
import numpy as np
from typing import Dict, Any, List
from utils.embed import embed

class SimpleHighchartsGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for parent-child relationships
        self.embeddings = {}  # Store embeddings separately for fast search
    
    def clean_description(self, text: str) -> str:
        """Remove markdown links and clean text"""
        if not text:
            return ""
        
        # Remove markdown links: [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove code blocks: `code` -> code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove reference links like [#123]
        text = re.sub(r'\[#\d+\]', '', text)
        
        # Remove see also links
        text = re.sub(r'see \[([^\]]+)\]\([^\)]+\)', r'see \1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_graph(self, tree_url: str):
        """Fetch JSON from URL and convert to graph"""
        print(f"Fetching tree from: {tree_url}")
        response = requests.get(tree_url)
        response.raise_for_status()
        tree_json = response.json()
        print("JSON fetched successfully!")
        
        print("Building graph...")
        self._process_node(tree_json, parent_path=None)
        print(f"Graph built! {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _process_node(self, node: Dict, parent_path: str = None):
        """Recursively process each node"""
        for key, value in node.items():
            if key in ['_meta']:
                continue
            
            if not isinstance(value, dict):
                continue
                
            doclet = value.get('doclet', {})
            meta = value.get('meta', {})
            
            if doclet:
                full_path = meta.get('fullname', key)
                
                # Clean description
                description = self.clean_description(
                    doclet.get('description', '')
                )
                
                # Create embedding
                embedding_text = f"{key}: {description}"
                embedding = embed(embedding_text)
                
                # Add node to graph
                self.graph.add_node(
                    full_path,
                    name=key,
                    description=description,
                    type=doclet.get('type', {}).get('names', []),
                    defaultValue=doclet.get('defaultvalue'),
                    since=doclet.get('since'),
                    samples=doclet.get('samples', []),
                    requires=doclet.get('requires', [])
                )
                
                # Store embedding separately
                self.embeddings[full_path] = np.array(embedding)
                
                # Create edge from child to parent
                if parent_path:
                    self.graph.add_edge(full_path, parent_path)
                
                # Process children
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=full_path)
            else:
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=parent_path)
    
    def save(self, filepath: str = "highcharts_graph.gpickle"):
        """Save graph to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'embeddings': self.embeddings
            }, f)
        print(f"Graph saved to {filepath}")
    
    def load(self, filepath: str = "highcharts_graph.gpickle"):
        """Load graph from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.embeddings = data['embeddings']
        print(f"Graph loaded from {filepath}")
```

## Updated retriever.py

```python
# retriever.py
import numpy as np
from typing import List, Dict
from utils.embed import embed

class SimpleRetriever:
    def __init__(self, graph_builder):
        self.graph = graph_builder.graph
        self.embeddings = graph_builder.embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """
        Semantic search + get surrounding context
        """
        # Get query embedding
        query_embedding = np.array(embed(user_query))
        
        # Calculate similarity with all nodes
        similarities = {}
        for node_id, node_embedding in self.embeddings.items():
            sim = self.cosine_similarity(query_embedding, node_embedding)
            similarities[node_id] = sim
        
        # Get top K most similar
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build results with context
        results = []
        for node_id, score in top_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Get ancestors (parents)
            ancestors = list(nx.ancestors(self.graph, node_id))
            
            # Get children
            children = list(self.graph.predecessors(node_id))  # predecessors because edges go child->parent
            
            result = {
                'path': node_id,
                'name': node_data['name'],
                'description': node_data['description'],
                'type': node_data['type'],
                'defaultValue': node_data.get('defaultValue'),
                'since': node_data.get('since'),
                'samples': node_data.get('samples', []),
                'ancestors': ancestors,
                'children': children,
                'relevance': float(score)
            }
            results.append(result)
        
        return results
    
    def get_subgraph(self, property_path: str, depth: int = 2) -> Dict:
        """Get subgraph around a specific property"""
        if property_path not in self.graph:
            return None
        
        center_data = self.graph.nodes[property_path]
        
        # Get ancestors (up to depth levels)
        ancestors = []
        current = property_path
        for _ in range(depth):
            parents = list(self.graph.successors(current))
            if parents:
                ancestors.extend(parents)
                current = parents[0]  # Follow first parent
            else:
                break
        
        # Get descendants (down to depth levels)
        descendants = []
        for node in nx.descendants(self.graph, property_path):
            if nx.shortest_path_length(self.graph, node, property_path) <= depth:
                descendants.append(node)
        
        # Get siblings (nodes with same parent)
        siblings = []
        parents = list(self.graph.successors(property_path))
        if parents:
            siblings = [n for n in self.graph.predecessors(parents[0]) 
                       if n != property_path]
        
        return {
            'center': {
                'path': property_path,
                'name': center_data['name'],
                'description': center_data['description'],
                'type': center_data['type'],
                'defaultValue': center_data.get('defaultValue')
            },
            'ancestors': ancestors,
            'descendants': descendants,
            'siblings': siblings
        }
```

## Updated main.py

```python
# main.py
from graph_builder import SimpleHighchartsGraph
from retriever import SimpleRetriever

def main():
    # Build or load graph
    builder = SimpleHighchartsGraph()
    
    # Option 1: Build from URL (first time)
    tree_url = "https://api.highcharts.com/highcharts/tree.json"
    builder.build_graph(tree_url)
    builder.save("highcharts_graph.gpickle")  # Save for later
    
    # Option 2: Load from file (subsequent runs)
    # builder.load("highcharts_graph.gpickle")
    
    # Query
    retriever = SimpleRetriever(builder)
    
    user_query = "change grid line color to green"
    print(f"\nSearching for: '{user_query}'")
    
    results = retriever.search(user_query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (relevance: {result['relevance']:.3f}) ---")
        print(f"Path: {result['path']}")
        print(f"Description: {result['description']}")
        print(f"Type: {result['type']}")
        print(f"Default: {result['defaultValue']}")
        if result['ancestors']:
            print(f"Parent chain: {' > '.join(reversed(result['ancestors']))}")

if __name__ == "__main__":
    main()
```

This is **completely self-contained** - no database needed! Everything runs in memory and can be saved/loaded from a pickle file.​​​​​​​​​​​​​​​​