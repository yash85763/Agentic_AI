Got it! Let me update the code to use your custom embed function:

## Updated Graph Builder

```python
# graph_builder.py
import re
from typing import Dict, Any
from utils.embed import embed  # Your custom embed function

class SimpleHighchartsGraph:
    def __init__(self, driver):
        self.driver = driver
    
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
    
    def build_graph(self, tree_json: Dict[str, Any]):
        """Convert JSON tree directly to graph"""
        self._process_node(tree_json, parent_path=None)
    
    def _process_node(self, node: Dict, parent_path: str = None):
        """
        Recursively process each node.
        Create a graph node if 'doclet' exists.
        """
        for key, value in node.items():
            # Skip metadata keys
            if key in ['_meta']:
                continue
            
            # Check if this is a property node (has doclet)
            if not isinstance(value, dict):
                continue
                
            doclet = value.get('doclet', {})
            meta = value.get('meta', {})
            
            # Only create node if doclet exists (actual property)
            if doclet:
                full_path = meta.get('fullname', key)
                
                # Clean description
                description = self.clean_description(
                    doclet.get('description', '')
                )
                
                # Prepare node data
                node_data = {
                    'id': full_path,
                    'name': key,
                    'fullPath': full_path,
                    'description': description,
                    'type': doclet.get('type', {}).get('names', []),
                    'defaultValue': doclet.get('defaultvalue'),
                    'since': doclet.get('since'),
                    'samples': doclet.get('samples', []),
                    'requires': doclet.get('requires', []),
                }
                
                # Create embedding using custom embed function
                embedding_text = f"{key}: {description}"
                node_data['embedding'] = embed(embedding_text)  # Your custom function
                
                # Create node in graph
                self._create_node(node_data)
                
                # Create relationship to parent
                if parent_path:
                    self._create_relationship(full_path, parent_path)
                
                # Process children
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=full_path)
            else:
                # No doclet but might have children (like at root level)
                children = value.get('children', {})
                if children:
                    self._process_node(children, parent_path=parent_path)
    
    def _create_node(self, node_data: Dict):
        """Create property node in Neo4j"""
        query = """
        CREATE (p:Property {
            id: $id,
            name: $name,
            fullPath: $fullPath,
            description: $description,
            type: $type,
            defaultValue: $defaultValue,
            since: $since,
            samples: $samples,
            requires: $requires,
            embedding: $embedding
        })
        """
        self.driver.execute_query(query, **node_data)
    
    def _create_relationship(self, child_path: str, parent_path: str):
        """Create CHILD_OF relationship"""
        query = """
        MATCH (child:Property {id: $child_path})
        MATCH (parent:Property {id: $parent_path})
        CREATE (child)-[:CHILD_OF]->(parent)
        """
        self.driver.execute_query(
            query, 
            child_path=child_path, 
            parent_path=parent_path
        )
```

## Updated Retriever

```python
# retriever.py
from utils.embed import embed  # Your custom embed function

class SimpleRetriever:
    def __init__(self, driver):
        self.driver = driver
    
    def search(self, user_query: str, top_k: int = 3):
        """
        Simple semantic search + get surrounding context
        Query: "change grid line color to green"
        """
        
        query_embedding = embed(user_query)  # Your custom function
        
        cypher = """
        // 1. Find most relevant nodes by vector similarity
        CALL db.index.vector.queryNodes(
            'property_embeddings', 
            $top_k, 
            $query_embedding
        )
        YIELD node, score
        
        // 2. Get parent chain (for context)
        OPTIONAL MATCH path = (node)-[:CHILD_OF*]->(ancestor)
        WITH node, score, collect(ancestor.fullPath) AS ancestors
        
        // 3. Get immediate children (if any)
        OPTIONAL MATCH (node)<-[:CHILD_OF]-(child)
        WITH node, score, ancestors, collect(child.fullPath) AS children
        
        // 4. Return everything
        RETURN {
            path: node.fullPath,
            name: node.name,
            description: node.description,
            type: node.type,
            defaultValue: node.defaultValue,
            since: node.since,
            samples: node.samples,
            ancestors: ancestors,
            children: children,
            relevance: score
        } AS result
        ORDER BY score DESC
        """
        
        results = self.driver.execute_query(
            cypher,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        return results
    
    def get_subgraph(self, property_path: str, depth: int = 2):
        """
        Get subgraph around a specific property
        """
        cypher = """
        MATCH (center:Property {fullPath: $property_path})
        
        // Get ancestors
        OPTIONAL MATCH (center)-[:CHILD_OF*1..3]->(ancestor)
        
        // Get descendants
        OPTIONAL MATCH (center)<-[:CHILD_OF*1..2]-(descendant)
        
        // Get siblings (same parent)
        OPTIONAL MATCH (center)-[:CHILD_OF]->(parent)<-[:CHILD_OF]-(sibling)
        WHERE sibling.id <> center.id
        
        RETURN {
            center: {
                path: center.fullPath,
                name: center.name,
                description: center.description,
                type: center.type,
                defaultValue: center.defaultValue
            },
            ancestors: collect(DISTINCT ancestor.fullPath),
            descendants: collect(DISTINCT descendant.fullPath),
            siblings: collect(DISTINCT sibling.fullPath)
        } AS subgraph
        """
        
        result = self.driver.execute_query(
            cypher,
            property_path=property_path
        )
        
        return result
```

## Main Script

```python
# main.py
from neo4j import GraphDatabase
from graph_builder import SimpleHighchartsGraph
from retriever import SimpleRetriever
import json

def main():
    # 1. Load JSON
    with open('highcharts_tree.json') as f:
        tree = json.load(f)
    
    # 2. Connect to Neo4j
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    # 3. Build graph
    print("Building graph...")
    builder = SimpleHighchartsGraph(driver)
    builder.build_graph(tree)
    print("Graph built successfully!")
    
    # 4. Create vector index (run once)
    print("Creating vector index...")
    create_index = """
    CREATE VECTOR INDEX property_embeddings IF NOT EXISTS
    FOR (p:Property)
    ON p.embedding
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1024,  // Update based on your embed() output
        `vector.similarity_function`: 'cosine'
      }
    }
    """
    driver.execute_query(create_index)
    print("Vector index created!")
    
    # 5. Query example
    retriever = SimpleRetriever(driver)
    
    user_query = "change grid line color to green"
    print(f"\nSearching for: {user_query}")
    
    results = retriever.search(user_query, top_k=3)
    
    for record in results:
        result = record['result']
        print(f"\n--- Result (relevance: {result['relevance']:.2f}) ---")
        print(f"Path: {result['path']}")
        print(f"Description: {result['description']}")
        print(f"Type: {result['type']}")
        print(f"Default: {result['defaultValue']}")
        print(f"Ancestors: {' > '.join(reversed(result['ancestors']))}")
    
    driver.close()

if __name__ == "__main__":
    main()
```

## Your utils/embed.py should look something like:

```python
# utils/embed.py
# Example structure - adjust based on your actual implementation

def embed(text: str) -> list:
    """
    Your custom embedding function.
    Should return a list of floats (the embedding vector).
    
    Args:
        text: Input text to embed
        
    Returns:
        list: Embedding vector (e.g., length 1024 or 768)
    """
    # Your implementation here
    # Could be OpenAI, Cohere, local model, etc.
    pass
```

Make sure to update the `vector.dimensions` in the index creation to match your embed() function’s output dimension!​​​​​​​​​​​​​​​​