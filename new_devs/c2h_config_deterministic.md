You’re absolutely right! Let’s separate the build and query phases, and add progress tracking + delays:

## Updated graph_builder.py

```python
# graph_builder.py
import re
import requests
import networkx as nx
import numpy as np
import time
from typing import Dict, Any
from utils.embed import embed

class SimpleHighchartsGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embeddings = {}
        self.node_count = 0
    
    def clean_description(self, text: str) -> str:
        """Remove markdown links and clean text"""
        if not text:
            return ""
        
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\[#\d+\]', '', text)
        text = re.sub(r'see \[([^\]]+)\]\([^\)]+\)', r'see \1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_graph(self, tree_url: str):
        """Fetch JSON from URL and convert to graph"""
        print(f"Fetching tree from: {tree_url}")
        print("This may take a moment, the file is large...")
        
        response = requests.get(tree_url, timeout=30)  # 30 second timeout
        response.raise_for_status()
        tree_json = response.json()
        
        print("✓ JSON fetched successfully!")
        print(f"JSON size: ~{len(str(tree_json)) / 1024:.1f} KB")
        print("\nBuilding graph (this will take a few minutes)...")
        
        self.node_count = 0
        self._process_node(tree_json, parent_path=None)
        
        print(f"\n✓ Graph built successfully!")
        print(f"  Total nodes: {self.graph.number_of_nodes()}")
        print(f"  Total edges: {self.graph.number_of_edges()}")
    
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
                
                # Show progress
                self.node_count += 1
                if self.node_count % 10 == 0:
                    print(f"  Processing node {self.node_count}...", end='\r')
                
                # Add small delay every 50 nodes to avoid rate limiting
                if self.node_count % 50 == 0:
                    time.sleep(0.5)  # 500ms delay
                
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
                
                # Store embedding
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
        print(f"\nSaving graph to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'embeddings': self.embeddings
            }, f)
        print(f"✓ Graph saved successfully!")
    
    def load(self, filepath: str = "highcharts_graph.gpickle"):
        """Load graph from file"""
        import pickle
        print(f"Loading graph from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.embeddings = data['embeddings']
        print(f"✓ Graph loaded successfully!")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
```

## Updated main.py - Split into two scripts

### build_graph.py (Run once to build)

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
    
    # Build graph
    tree_url = "https://api.highcharts.com/highcharts/tree.json"
    
    try:
        builder.build_graph(tree_url)
        builder.save("highcharts_graph.gpickle")
        print("\n✓ All done! You can now run query.py to search the graph.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("If you got a timeout, try running again with a better internet connection.")

if __name__ == "__main__":
    main()
```

### query.py (Run to search)

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
    print("Highcharts API Search")
    print("="*60)
    print("Type your question or 'quit' to exit\n")
    
    while True:
        user_query = input("Query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        print(f"\nSearching for: '{user_query}'")
        print("-" * 60)
        
        results = retriever.search(user_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['path']} (relevance: {result['relevance']:.3f})")
            print(f"    Type: {', '.join(result['type']) if result['type'] else 'N/A'}")
            print(f"    Default: {result['defaultValue']}")
            print(f"    Description: {result['description'][:150]}...")
            if result['ancestors']:
                print(f"    Parent: {result['ancestors'][0] if result['ancestors'] else 'N/A'}")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
```

## Usage:

```bash
# Step 1: Build the graph (run once, takes several minutes)
python build_graph.py

# Step 2: Query the graph (run as many times as you want)
python query.py
```

Example query session:

```
Query: change grid line color to green

[1] xAxis.gridLineColor (relevance: 0.892)
    Type: ColorString
    Default: #e6e6e6
    Description: Color of the grid lines extending from the axis across the plot area...
    Parent: xAxis

[2] yAxis.gridLineColor (relevance: 0.885)
    ...
```

This approach:

- ✓ Separates build and query
- ✓ Shows progress during build
- ✓ Adds delays to avoid rate limiting
- ✓ Saves graph for reuse
- ✓ Interactive query mode​​​​​​​​​​​​​​​​