Great question! Here are the updates to handle multiple tree files:

## Updated graph_builder.py

```python
# graph_builder.py
# ... (keep all existing imports and clean_description method)

class SimpleHighchartsGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embeddings = {}
        self.node_count = 0
    
    # ADD: New method to build from multiple sources
    def build_from_multiple_sources(self, sources: Dict[str, str]):
        """
        Build graph from multiple tree sources
        
        Args:
            sources: Dict of {source_name: url}
                    e.g., {'highcharts': 'url1', 'highstock': 'url2'}
        """
        for source_name, tree_url in sources.items():
            print(f"\n{'='*60}")
            print(f"Processing: {source_name}")
            print(f"{'='*60}")
            self.build_graph(tree_url, source=source_name)
    
    # UPDATE: Add source parameter to build_graph
    def build_graph(self, tree_url: str, source: str = "highcharts"):
        """Fetch JSON from URL and convert to graph"""
        print(f"Fetching tree from: {tree_url}")
        print("This may take a moment, the file is large...")
        
        response = requests.get(tree_url, timeout=30)
        response.raise_for_status()
        tree_json = response.json()
        
        print("✓ JSON fetched successfully!")
        print(f"JSON size: ~{len(str(tree_json)) / 1024:.1f} KB")
        print(f"\nBuilding graph for {source} (this will take a few minutes)...")
        
        start_count = self.node_count
        self._process_node(tree_json, parent_path=None, source=source)
        
        nodes_added = self.node_count - start_count
        print(f"\n✓ {source} graph built successfully!")
        print(f"  Nodes added: {nodes_added}")
        print(f"  Total nodes so far: {self.graph.number_of_nodes()}")
    
    # UPDATE: Add source parameter to _process_node
    def _process_node(self, node: Dict, parent_path: str = None, source: str = "highcharts"):
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
                
                # Show progress
                self.node_count += 1
                if self.node_count % 10 == 0:
                    print(f"  Processing node {self.node_count}...", end='\r')
                
                # Add delay every 50 nodes
                if self.node_count % 50 == 0:
                    time.sleep(0.5)
                
                embedding = embed(embedding_text)
                
                # Add node to graph with source tag
                self.graph.add_node(
                    full_path,
                    name=key,
                    description=description,
                    type=doclet.get('type', {}).get('names', []),
                    defaultValue=doclet.get('defaultvalue'),
                    since=doclet.get('since'),
                    samples=doclet.get('samples', []),
                    requires=doclet.get('requires', []),
                    source=source  # ADD: Tag with source
                )
                
                # Store embedding
                self.embeddings[full_path] = np.array(embedding)
                
                # Create edge from child to parent
                if parent_path:
                    self.graph.add_edge(full_path, parent_path)
                
                # Process children
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=full_path, source=source)
            else:
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=parent_path, source=source)
    
    # ... (keep save and load methods unchanged)
```

## Updated retriever.py

```python
# retriever.py
import numpy as np
from typing import List, Dict, Optional
from utils.embed import embed

class SimpleRetriever:
    def __init__(self, graph_builder):
        self.graph = graph_builder.graph
        self.embeddings = graph_builder.embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # UPDATE: Add source filter
    def search(self, user_query: str, top_k: int = 3, source: Optional[str] = None) -> List[Dict]:
        """
        Semantic search + get surrounding context
        
        Args:
            user_query: Search query
            top_k: Number of results
            source: Filter by source ('highcharts', 'highstock', or None for all)
        """
        # Get query embedding
        query_embedding = np.array(embed(user_query))
        
        # Calculate similarity with all nodes (optionally filtered by source)
        similarities = {}
        for node_id, node_embedding in self.embeddings.items():
            # Filter by source if specified
            if source:
                node_source = self.graph.nodes[node_id].get('source')
                if node_source != source:
                    continue
            
            sim = self.cosine_similarity(query_embedding, node_embedding)
            similarities[node_id] = sim
        
        # Get top K most similar
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build results with context
        results = []
        for node_id, score in top_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Get ancestors
            ancestors = list(nx.ancestors(self.graph, node_id))
            
            # Get children
            children = list(self.graph.predecessors(node_id))
            
            result = {
                'path': node_id,
                'name': node_data['name'],
                'description': node_data['description'],
                'type': node_data['type'],
                'defaultValue': node_data.get('defaultValue'),
                'since': node_data.get('since'),
                'samples': node_data.get('samples', []),
                'source': node_data.get('source', 'unknown'),  # ADD: Include source
                'ancestors': ancestors,
                'children': children,
                'relevance': float(score)
            }
            results.append(result)
        
        return results
    
    # ... (keep get_subgraph method unchanged)
```

## Updated build_graph.py

```python
# build_graph.py
from graph_builder import SimpleHighchartsGraph
import os

def main():
    builder = SimpleHighchartsGraph()
    
    # Check if graph already exists
    if os.path.exists("highcharts_graph.gpickle"):
        print("⚠ Graph file already exists!")
        response = input("Do you want to rebuild? (yes/no): ")
        if response.lower() != 'yes':
            print("Skipping rebuild. Use query.py to search.")
            return
    
    # Define multiple sources
    sources = {
        'highcharts': 'https://api.highcharts.com/highcharts/tree.json',
        'highstock': 'https://api.highcharts.com/highstock/tree.json'
    }
    
    try:
        builder.build_from_multiple_sources(sources)
        builder.save("highcharts_graph.gpickle")
        
        print("\n" + "="*60)
        print("✓ All done! Summary:")
        print(f"  Total nodes: {builder.graph.number_of_nodes()}")
        print(f"  Total edges: {builder.graph.number_of_edges()}")
        
        # Show node count per source
        from collections import Counter
        sources_count = Counter(
            data.get('source', 'unknown') 
            for _, data in builder.graph.nodes(data=True)
        )
        print("\n  Nodes per source:")
        for source, count in sources_count.items():
            print(f"    {source}: {count}")
        
        print("\nYou can now run query.py to search the graph.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("If you got a timeout, try running again with a better internet connection.")

if __name__ == "__main__":
    main()
```

## Updated query.py

```python
# query.py
from graph_builder import SimpleHighchartsGraph
from retriever import SimpleRetriever
import os

def main():
    # Check if graph exists
    if not os.path.exists("highcharts_graph.gpickle"):
        print("✗ Graph file not found!")
        print("Please run build_graph.py first to build the graph.")
        return
    
    # Load graph
    builder = SimpleHighchartsGraph()
    builder.load("highcharts_graph.gpickle")
    
    # Create retriever
    retriever = SimpleRetriever(builder)
    
    # Interactive query loop
    print("\n" + "="*60)
    print("Highcharts/Highstock API Search")
    print("="*60)
    print("Commands:")
    print("  - Type your question to search")
    print("  - 'filter:highcharts <query>' - search only Highcharts")
    print("  - 'filter:highstock <query>' - search only Highstock")
    print("  - 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        user_input = input("Query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Parse filter command
        source_filter = None
        if user_input.startswith('filter:'):
            parts = user_input.split(' ', 1)
            if len(parts) == 2:
                source_filter = parts[0].replace('filter:', '')
                user_query = parts[1]
            else:
                print("Usage: filter:highcharts <query>")
                continue
        else:
            user_query = user_input
        
        print(f"\nSearching for: '{user_query}'", end="")
        if source_filter:
            print(f" (filtered by: {source_filter})")
        else:
            print(" (all sources)")
        print("-" * 60)
        
        results = retriever.search(user_query, top_k=5, source=source_filter)
        
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['path']} [{result['source']}]")
                print(f"    Relevance: {result['relevance']:.3f}")
                print(f"    Type: {', '.join(result['type']) if result['type'] else 'N/A'}")
                print(f"    Default: {result['defaultValue']}")
                print(f"    Description: {result['description'][:150]}...")
                if result['ancestors']:
                    print(f"    Parent: {result['ancestors'][0] if result['ancestors'] else 'N/A'}")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
```

## Summary of Changes:

1. **graph_builder.py**:

- Added `source` parameter to tag each node
- Added `build_from_multiple_sources()` method

1. **retriever.py**:

- Added `source` filter to search method
- Returns source in results

1. **build_graph.py**:

- Now builds from both URLs
- Shows node count per source

1. **query.py**:

- Added filter commands (`filter:highcharts` or `filter:highstock`)
- Displays source in results

Usage:

```bash
# Build both
python build_graph.py

# Search all
Query: grid line color

# Search only highstock
Query: filter:highstock navigator
```