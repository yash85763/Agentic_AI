"""
Complete GraphRAG Implementation with Custom LLM
Processes text documents, builds knowledge graph, and enables querying
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from collections import defaultdict
import logging

# Install these if needed: pip install networkx pandas numpy
import networkx as nx
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== CUSTOM LLM WRAPPER ====================

@dataclass
class LLMConfig:
    """Configuration for your custom LLM"""
    api_key: str
    model_name: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 120


class CustomLLMAdapter:
    """
    Adapter for your custom LLM SDK.
    
    INSTRUCTIONS TO ADAPT:
    1. Import your SDK: from your_sdk import YourLLMClient
    2. Initialize in __init__: self.client = YourLLMClient(...)
    3. Modify _call_llm() to use your SDK's API
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        # TODO: Replace this with your actual SDK initialization
        # Example:
        # from your_sdk import YourLLMClient
        # self.client = YourLLMClient(
        #     api_key=config.api_key,
        #     base_url=config.base_url
        # )
        
        logger.info(f"Initialized LLM: {config.model_name}")
    
    async def _call_llm(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """
        Call your custom LLM SDK.
        
        TODO: Replace this method with your actual SDK call.
        
        Example for most SDKs:
        response = await self.client.generate(
            model=self.config.model_name,
            prompt=prompt,
            system=system_prompt,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        return response.text  # or however your SDK returns the text
        """
        
        # PLACEHOLDER: This simulates an LLM response for testing
        # Replace this entire block with your SDK call
        await asyncio.sleep(0.5)  # Simulate API call
        
        if "extract entities" in prompt.lower():
            return json.dumps([
                {"name": "GraphRAG", "type": "TECHNOLOGY", "description": "Knowledge graph RAG system"},
                {"name": "Python", "type": "TECHNOLOGY", "description": "Programming language"},
            ])
        elif "extract relationships" in prompt.lower():
            return json.dumps([
                {"source": "GraphRAG", "target": "Python", "relationship": "IMPLEMENTED_IN", 
                 "description": "GraphRAG is implemented using Python"}
            ])
        else:
            return "This is a placeholder response. Replace _call_llm() with your SDK."
    
    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate response from LLM."""
        try:
            response = await self._call_llm(prompt, system_prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


# ==================== DATA MODELS ====================

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    type: str
    description: str
    source_doc: str
    mentions: int = 1
    
    def __hash__(self):
        return hash(self.name.lower())
    
    def __eq__(self, other):
        return self.name.lower() == other.name.lower()


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relationship: str
    description: str
    source_doc: str
    weight: float = 1.0


# ==================== DOCUMENT PROCESSOR ====================

class DocumentLoader:
    """Loads and processes text documents."""
    
    @staticmethod
    def load_documents(directory: str) -> List[Dict[str, str]]:
        """
        Load all .txt files from directory.
        
        Args:
            directory: Path to directory containing .txt files
            
        Returns:
            List of dicts with 'filename', 'content', and 'path'
        """
        documents = []
        doc_path = Path(directory)
        
        if not doc_path.exists():
            logger.error(f"Directory not found: {directory}")
            return documents
        
        txt_files = list(doc_path.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append({
                            'filename': filepath.name,
                            'content': content,
                            'path': str(filepath)
                        })
                        logger.info(f"Loaded: {filepath.name} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        return documents
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks


# ==================== GRAPH RAG CORE ====================

class GraphRAGEngine:
    """Main GraphRAG engine for building and querying knowledge graphs."""
    
    def __init__(self, llm_adapter: CustomLLMAdapter):
        self.llm = llm_adapter
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.documents: List[Dict] = []
        
    async def build_graph_from_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of document dicts with 'content' and 'filename'
        """
        logger.info(f"Building graph from {len(documents)} documents...")
        self.documents = documents
        
        for idx, doc in enumerate(documents):
            logger.info(f"Processing [{idx+1}/{len(documents)}]: {doc['filename']}")
            
            # Chunk large documents
            content = doc['content']
            if len(content) > 2000:
                chunks = DocumentLoader.chunk_text(content, chunk_size=1500, overlap=300)
                logger.info(f"  Split into {len(chunks)} chunks")
            else:
                chunks = [content]
            
            # Process each chunk
            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"  Processing chunk {chunk_idx+1}/{len(chunks)}")
                
                # Extract entities
                entities = await self._extract_entities(chunk, doc['filename'])
                logger.info(f"    Found {len(entities)} entities")
                
                # Extract relationships
                relationships = await self._extract_relationships(
                    chunk, entities, doc['filename']
                )
                logger.info(f"    Found {len(relationships)} relationships")
                
                # Add to graph
                self._add_to_graph(entities, relationships)
        
        logger.info(f"Graph built: {len(self.entities)} entities, "
                   f"{len(self.relationships)} relationships")
        logger.info(f"Graph stats: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
    
    async def _extract_entities(self, text: str, source_doc: str) -> List[Entity]:
        """Extract entities using LLM."""
        
        prompt = f"""Extract all important entities from the text below.

For each entity, provide:
- name: The entity name (be specific and use full names)
- type: Entity type (choose from: PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT, EVENT, PRODUCT, OTHER)
- description: Brief description (1-2 sentences)

Text:
{text}

Return ONLY a valid JSON array with no additional text. Format:
[
  {{"name": "Entity Name", "type": "TYPE", "description": "Description here"}},
  ...
]
"""
        
        system_prompt = "You are an expert at extracting structured information. Return only valid JSON."
        
        try:
            response = await self.llm.generate(prompt, system_prompt)
            entities_data = self._parse_json_response(response)
            
            entities = []
            for item in entities_data:
                if all(k in item for k in ['name', 'type', 'description']):
                    entity = Entity(
                        name=item['name'].strip(),
                        type=item['type'].strip().upper(),
                        description=item['description'].strip(),
                        source_doc=source_doc
                    )
                    entities.append(entity)
            
            return entities
        
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    async def _extract_relationships(
        self, 
        text: str, 
        entities: List[Entity],
        source_doc: str
    ) -> List[Relationship]:
        """Extract relationships between entities using LLM."""
        
        if len(entities) < 2:
            return []
        
        entity_list = [f"- {e.name} ({e.type})" for e in entities[:20]]  # Limit to 20
        entity_names = "\n".join(entity_list)
        
        prompt = f"""Given these entities:
{entity_names}

Find relationships between them in the text below.

For each relationship provide:
- source: Source entity name (must match an entity from the list)
- target: Target entity name (must match an entity from the list)
- relationship: Relationship type (e.g., WORKS_FOR, LOCATED_IN, PART_OF, CREATED_BY, USES, RELATED_TO)
- description: Brief description of the relationship

Text:
{text}

Return ONLY a valid JSON array with no additional text. Format:
[
  {{"source": "Entity1", "target": "Entity2", "relationship": "RELATIONSHIP_TYPE", "description": "Description"}},
  ...
]
"""
        
        system_prompt = "You are an expert at extracting relationships. Return only valid JSON."
        
        try:
            response = await self.llm.generate(prompt, system_prompt)
            rels_data = self._parse_json_response(response)
            
            relationships = []
            entity_names_set = {e.name.lower() for e in entities}
            
            for item in rels_data:
                if all(k in item for k in ['source', 'target', 'relationship', 'description']):
                    # Validate entities exist
                    source = item['source'].strip()
                    target = item['target'].strip()
                    
                    if (source.lower() in entity_names_set and 
                        target.lower() in entity_names_set):
                        rel = Relationship(
                            source=source,
                            target=target,
                            relationship=item['relationship'].strip().upper(),
                            description=item['description'].strip(),
                            source_doc=source_doc
                        )
                        relationships.append(rel)
            
            return relationships
        
        except Exception as e:
            logger.error(f"Relationship extraction error: {e}")
            return []
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Parse JSON from LLM response, handling common formatting issues."""
        try:
            # Try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            logger.warning("Could not parse JSON response")
            return []
    
    def _add_to_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """Add entities and relationships to the graph."""
        
        # Add/update entities
        for entity in entities:
            entity_key = entity.name.lower()
            if entity_key in self.entities:
                # Entity exists, increment mentions
                self.entities[entity_key].mentions += 1
            else:
                # New entity
                self.entities[entity_key] = entity
                self.graph.add_node(
                    entity.name,
                    type=entity.type,
                    description=entity.description,
                    mentions=entity.mentions
                )
        
        # Add relationships
        for rel in relationships:
            self.relationships.append(rel)
            
            # Add edge to graph
            if self.graph.has_node(rel.source) and self.graph.has_node(rel.target):
                if self.graph.has_edge(rel.source, rel.target):
                    # Edge exists, increase weight
                    self.graph[rel.source][rel.target]['weight'] += 1
                else:
                    # New edge
                    self.graph.add_edge(
                        rel.source,
                        rel.target,
                        relationship=rel.relationship,
                        description=rel.description,
                        weight=rel.weight
                    )
    
    async def query(self, question: str, top_k: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            question: User's question
            top_k: Number of relevant context items to retrieve
            
        Returns:
            Tuple of (answer, context_info)
        """
        logger.info(f"Query: {question}")
        
        # Step 1: Extract key entities/concepts from question
        query_entities = await self._extract_query_entities(question)
        logger.info(f"Query entities: {query_entities}")
        
        # Step 2: Retrieve relevant subgraph
        context, context_info = self._retrieve_subgraph(query_entities, top_k)
        logger.info(f"Retrieved context: {len(context_info['entities'])} entities, "
                   f"{len(context_info['relationships'])} relationships")
        
        # Step 3: Generate answer
        answer = await self._generate_answer(question, context)
        
        return answer, context_info
    
    async def _extract_query_entities(self, question: str) -> List[str]:
        """Extract key entities/concepts from the question."""
        
        prompt = f"""Extract key entities, concepts, or topics from this question that would be relevant for searching a knowledge graph.

Question: {question}

Return ONLY a JSON array of strings (entity names), no additional text:
["entity1", "entity2", ...]
"""
        
        try:
            response = await self.llm.generate(prompt)
            entities = self._parse_json_response(response)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
            return []
        except Exception as e:
            logger.error(f"Query entity extraction error: {e}")
            # Fallback: extract capitalized words
            return [word for word in question.split() if word and word[0].isupper()]
    
    def _retrieve_subgraph(
        self, 
        query_entities: List[str], 
        top_k: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Retrieve relevant subgraph based on query entities."""
        
        relevant_nodes = set()
        relevant_edges = []
        
        # Find matching nodes (case-insensitive)
        query_lower = [q.lower() for q in query_entities]
        for node in self.graph.nodes():
            node_lower = node.lower()
            # Check if any query entity is in node name or vice versa
            if any(q in node_lower or node_lower in q for q in query_lower):
                relevant_nodes.add(node)
        
        # If no direct matches, use most mentioned entities
        if not relevant_nodes:
            sorted_entities = sorted(
                self.entities.values(),
                key=lambda x: x.mentions,
                reverse=True
            )
            relevant_nodes = {e.name for e in sorted_entities[:top_k]}
        
        # Expand to neighbors
        expanded_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            if self.graph.has_node(node):
                neighbors = list(self.graph.neighbors(node))
                predecessors = list(self.graph.predecessors(node))
                expanded_nodes.update(neighbors[:3])  # Add top 3 neighbors
                expanded_nodes.update(predecessors[:3])
        
        # Limit total nodes
        final_nodes = list(expanded_nodes)[:top_k]
        
        # Get relationships between these nodes
        for source in final_nodes:
            for target in final_nodes:
                if self.graph.has_edge(source, target):
                    edge_data = self.graph[source][target]
                    relevant_edges.append({
                        'source': source,
                        'target': target,
                        'relationship': edge_data.get('relationship', 'RELATED_TO'),
                        'description': edge_data.get('description', '')
                    })
        
        # Format context
        context_parts = []
        
        # Add entity information
        context_parts.append("=== ENTITIES ===")
        for node in final_nodes:
            if self.graph.has_node(node):
                node_data = self.graph.nodes[node]
                context_parts.append(
                    f"- {node} ({node_data.get('type', 'UNKNOWN')}): "
                    f"{node_data.get('description', 'No description')}"
                )
        
        # Add relationships
        if relevant_edges:
            context_parts.append("\n=== RELATIONSHIPS ===")
            for edge in relevant_edges[:top_k]:
                context_parts.append(
                    f"- {edge['source']} --[{edge['relationship']}]--> {edge['target']}: "
                    f"{edge['description']}"
                )
        
        context = "\n".join(context_parts)
        
        context_info = {
            'entities': final_nodes,
            'relationships': relevant_edges,
            'entity_count': len(final_nodes),
            'relationship_count': len(relevant_edges)
        }
        
        return context, context_info
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using retrieved context."""
        
        prompt = f"""Answer the question based ONLY on the provided knowledge graph context. Be specific and detailed.

{context}

Question: {question}

Instructions:
- Use only information from the context above
- If the context doesn't contain enough information, say so
- Be concise but complete
- Cite specific entities and relationships when possible

Answer:"""
        
        system_prompt = "You are a helpful assistant that answers questions based on knowledge graph context."
        
        try:
            answer = await self.llm.generate(prompt, system_prompt, temperature=0.3)
            return answer.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "Error generating answer."
    
    def save_graph(self, filepath: str):
        """Save graph to file."""
        data = {
            'entities': [asdict(e) for e in self.entities.values()],
            'relationships': [asdict(r) for r in self.relationships]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load graph from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct entities and relationships
        for e_dict in data['entities']:
            entity = Entity(**e_dict)
            self.entities[entity.name.lower()] = entity
            self.graph.add_node(
                entity.name,
                type=entity.type,
                description=entity.description,
                mentions=entity.mentions
            )
        
        for r_dict in data['relationships']:
            rel = Relationship(**r_dict)
            self.relationships.append(rel)
            if self.graph.has_node(rel.source) and self.graph.has_node(rel.target):
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    relationship=rel.relationship,
                    description=rel.description,
                    weight=rel.weight
                )
        
        logger.info(f"Graph loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'entity_types': {},
            'relationship_types': {},
            'most_connected_entities': []
        }
        
        # Entity types
        for entity in self.entities.values():
            stats['entity_types'][entity.type] = stats['entity_types'].get(entity.type, 0) + 1
        
        # Relationship types
        for rel in self.relationships:
            stats['relationship_types'][rel.relationship] = \
                stats['relationship_types'].get(rel.relationship, 0) + 1
        
        # Most connected entities
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            top_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['most_connected_entities'] = [
                {'entity': name, 'connections': round(score * 100, 2)}
                for name, score in top_entities
            ]
        
        return stats


# ==================== MAIN APPLICATION ====================

async def main():
    """Main application to test GraphRAG."""
    
    print("=" * 80)
    print("GraphRAG System - Document Processing and Query")
    print("=" * 80)
    
    # Step 1: Configure your LLM
    print("\n[1] Configuring LLM...")
    config = LLMConfig(
        api_key="your-api-key-here",  # Replace with your API key
        model_name="your-model-name",  # Replace with your model name
        base_url="https://your-api-endpoint.com",  # Optional: your API endpoint
        temperature=0.7,
        max_tokens=3000
    )
    
    llm_adapter = CustomLLMAdapter(config)
    
    # Step 2: Load documents
    print("\n[2] Loading documents...")
    documents_dir = "./documents"  # Directory containing your .txt files
    
    # Create example documents if directory doesn't exist
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created directory: {documents_dir}")
        print("Please add your .txt files to this directory and run again.")
        
        # Create sample documents for testing
        sample_docs = {
            "graphrag_intro.txt": """GraphRAG is an advanced retrieval-augmented generation technique developed by Microsoft Research. 
It combines knowledge graphs with large language models to improve information retrieval and question answering.
GraphRAG was created by the Microsoft Research team to address limitations in traditional RAG systems.
The system works by extracting entities and relationships from documents to build a knowledge graph.
This knowledge graph enables more sophisticated queries and better context understanding.""",
            
            "python_ml.txt": """Python is a popular programming language widely used in machine learning and data science.
TensorFlow and PyTorch are two major deep learning frameworks built with Python.
Guido van Rossum created Python in 1991 at Centrum Wiskunde & Informatica in the Netherlands.
Python's simplicity and extensive libraries make it ideal for artificial intelligence research.
Many tech companies including Google, Facebook, and OpenAI use Python for their ML systems.""",
            
            "ai_companies.txt": """OpenAI is an AI research company that created ChatGPT and GPT models.
Microsoft invested heavily in OpenAI and integrated GPT models into their products.
Google DeepMind, formed by merging Google Brain and DeepMind, focuses on AI safety research.
Anthropic, founded by former OpenAI researchers, developed Claude AI assistant.
These companies are leading the development of large language models and artificial general intelligence."""
        }
        
        for filename, content in sample_docs.items():
            with open(os.path.join(documents_dir, filename), 'w') as f:
                f.write(content)
        print(f"Created {len(sample_docs)} sample documents for testing.")
    
    documents = DocumentLoader.load_documents(documents_dir)
    
    if not documents:
        print("No documents found. Please add .txt files to the documents directory.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Step 3: Build knowledge graph
    print("\n[3] Building knowledge graph...")
    engine = GraphRAGEngine(llm_adapter)
    await engine.build_graph_from_documents(documents)
    
    # Step 4: Display statistics
    print("\n[4] Graph Statistics:")
    stats = engine.get_statistics()
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Total Relationships: {stats['total_relationships']}")
    print(f"  Graph Nodes: {stats['graph_nodes']}")
    print(f"  Graph Edges: {stats['graph_edges']}")
    
    print(f"\n  Entity Types:")
    for etype, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {etype}: {count}")
    
    print(f"\n  Relationship Types:")
    for rtype, count in sorted(stats['relationship_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {rtype}: {count}")
    
    if stats['most_connected_entities']:
        print(f"\n  Most Connected Entities:")
        for item in stats['most_connected_entities'][:5]:
            print(f"    - {item['entity']}: {item['connections']}%")
    
    # Step 5: Save graph
    graph_file = "knowledge_graph.json"
    engine.save_graph(graph_file)
    print(f"\n[5] Knowledge graph saved to: {graph_file}")
    
    # Step 6: Test queries
    print("\n[6] Testing GraphRAG with sample queries...")
    print("=" * 80)
    
    test_questions = [
        "What is GraphRAG and how does it work?",
        "Which companies are involved in AI research?",
        "What is the relationship between Microsoft and OpenAI?",
        "Who created Python and when?",
        "What are the main AI frameworks used in Python?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {question}")
        print(f"{'='*80}")
        
        answer, context_info = await engine.query(question, top_k=8)
        
        print(f"\nContext Retrieved:")
        print(f"  - Entities: {context_info['entity_count']}")
        print(f"  - Relationships: {context_info['relationship_count']}")
        
        print(f"\nAnswer:")
        print(f"{answer}")
        
        print(f"\nRelevant Entities: {', '.join(context_info['entities'][:5])}")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Step 7: Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Query Mode (type 'exit' to quit, 'stats' for statistics)")
    print("=" * 80)
    
    while True:
        try:
            user_question = input("\nYour question: ").strip()
            
            if user_question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if user_question.lower() == 'stats':
                stats = engine.get_statistics()
                print(json.dumps(stats, indent=2))
                continue
            
            if not user_question:
                continue
            
            answer, context_info = await engine.query(user_question, top_k=10)
            
            print(f"\n--- Answer ---")
            print(answer)
            print(f"\n--- Context Used ---")
            print(f"Entities: {', '.join(context_info['entities'][:8])}")
            print(f"Relationships: {context_info['relationship_count']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("\nIMPORTANT: Replace the _call_llm() method in CustomLLMAdapter")
    print("with your actual LLM SDK implementation.\n")
    
    asyncio.run(main())