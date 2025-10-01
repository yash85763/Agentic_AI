"""
Microsoft GraphRAG Implementation with Custom LLM/Embeddings
Uses the official graphrag library with your custom SDK for both LLM and embeddings

Installation:
pip install graphrag numpy pandas tiktoken

Setup:
1. Uncomment the SDK import and initialization in CustomLLMClient
2. Add your API key and model names in main()
3. Put your .txt files in ./documents folder
4. Run: python script.py
"""

import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# GraphRAG imports
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    LLM,
    CompletionInput,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)
from graphrag.config import (
    GraphRagConfig,
    LLMConfig as GraphRagLLMConfig,
    EmbeddingConfig,
    ChunkingConfig,
    EntityExtractionConfig,
    ClaimExtractionConfig,
    CommunityReportsConfig,
    SummarizationConfig,
)
from graphrag.index import create_pipeline_config
from graphrag.index.run import run_pipeline_with_config
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.vector_stores.lancedb import LanceDBVectorStore

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== CUSTOM LLM WRAPPER ====================

class CustomLLMClient(BaseLLM):
    """
    Custom LLM wrapper that implements GraphRAG's BaseLLM interface.
    Integrates your SDK's client.chat.create() pattern.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize custom LLM client.
        
        Args:
            api_key: Your API key
            model: Model name
            api_base: Optional API base URL
            temperature: Generation temperature
            max_tokens: Max tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # TODO: Uncomment and configure your SDK
        # from my-sdk import MyLLMClient
        # self.client = MyLLMClient(
        #     api_key=api_key,
        #     # Add other initialization parameters
        # )
        
        # PLACEHOLDER: Remove when adding your SDK
        self.client = None
        
        logger.info(f"Initialized Custom LLM: {model}")
    
    async def _execute_llm(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Execute LLM call using your SDK pattern.
        """
        # Build messages in the format your SDK expects
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # TODO: Uncomment when you add your SDK
        """
        response = self.client.chat.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            n=1,
        )
        
        return response.get("choices")[0].get("text")
        """
        
        # PLACEHOLDER: Remove this when adding real SDK
        await asyncio.sleep(0.1)
        return f"Placeholder response for: {prompt[:50]}..."
    
    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        streaming: bool = False,
        **kwargs
    ) -> LLMOutput:
        """
        GraphRAG calls this method for text generation.
        Converts messages to prompt and calls your SDK.
        """
        # Convert messages to single prompt
        prompt = "\n\n".join([
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        # Call your LLM
        response_text = await self._execute_llm(prompt, **kwargs)
        
        # Return in GraphRAG expected format
        return LLMOutput(
            output=response_text,
            json=None
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        streaming: bool = False,
        **kwargs
    ) -> LLMOutput:
        """Synchronous version of agenerate."""
        return asyncio.run(self.agenerate(messages, streaming, **kwargs))
    
    async def agenerate_completion(
        self,
        prompt: str,
        **kwargs
    ) -> CompletionOutput:
        """Generate completion from prompt."""
        response_text = await self._execute_llm(prompt, **kwargs)
        return CompletionOutput(output=response_text)
    
    def generate_completion(
        self,
        prompt: str,
        **kwargs
    ) -> CompletionOutput:
        """Synchronous completion generation."""
        return asyncio.run(self.agenerate_completion(prompt, **kwargs))


# ==================== CUSTOM EMBEDDINGS WRAPPER ====================

class CustomEmbeddingsClient:
    """
    Custom embeddings wrapper using your LLM SDK.
    Many LLM APIs also provide embedding endpoints.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize custom embeddings client.
        
        Args:
            api_key: Your API key
            model: Embedding model name (e.g., "text-embedding-ada-002")
            api_base: Optional API base URL
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        
        # TODO: Uncomment and configure your SDK
        # from my-sdk import MyLLMClient
        # self.client = MyLLMClient(
        #     api_key=api_key,
        #     # Add other initialization parameters
        # )
        
        # PLACEHOLDER
        self.client = None
        
        logger.info(f"Initialized Custom Embeddings: {model}")
    
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using your SDK.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # TODO: Uncomment when you add your SDK
        # If your SDK has an embeddings endpoint:
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract embeddings (adjust based on your SDK's response format)
        embeddings = [item.get("embedding") for item in response.get("data", [])]
        return embeddings
        """
        
        # PLACEHOLDER: Returns random embeddings for testing
        # Remove this when adding real SDK
        await asyncio.sleep(0.05)
        embedding_dim = 1536  # Standard dimension
        return [np.random.randn(embedding_dim).tolist() for _ in texts]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Synchronous version of aembed."""
        return asyncio.run(self.aembed(texts))


# ==================== GRAPHRAG CONFIGURATION ====================

def create_graphrag_config(
    llm_client: CustomLLMClient,
    embeddings_client: CustomEmbeddingsClient,
    root_dir: str
) -> GraphRagConfig:
    """
    Create GraphRAG configuration using custom LLM and embeddings.
    
    Args:
        llm_client: Your custom LLM client
        embeddings_client: Your custom embeddings client
        root_dir: Root directory for GraphRAG data
        
    Returns:
        GraphRagConfig object
    """
    
    config = GraphRagConfig(
        root_dir=root_dir,
        
        # LLM Configuration - uses your custom client
        llm=GraphRagLLMConfig(
            api_key=llm_client.api_key,
            type="openai_chat",  # GraphRAG expects OpenAI-compatible format
            model=llm_client.model,
            api_base=llm_client.api_base,
            temperature=llm_client.temperature,
            max_tokens=llm_client.max_tokens,
        ),
        
        # Embeddings Configuration - uses your custom client
        embeddings=EmbeddingConfig(
            api_key=embeddings_client.api_key,
            type="openai_embedding",
            model=embeddings_client.model,
            api_base=embeddings_client.api_base,
        ),
        
        # Chunking configuration
        chunks=ChunkingConfig(
            size=1200,
            overlap=100,
            group_by_columns=["id"],
        ),
        
        # Entity extraction - uses GraphRAG's tested prompts
        entity_extraction=EntityExtractionConfig(
            enabled=True,
            max_gleanings=1,  # Number of times to refine extraction
        ),
        
        # Community reports
        community_reports=CommunityReportsConfig(
            enabled=True,
            max_length=2000,
        ),
        
        # Summarization
        summarize_descriptions=SummarizationConfig(
            enabled=True,
            max_length=500,
        ),
    )
    
    return config


# ==================== DOCUMENT PROCESSING ====================

class GraphRAGPipeline:
    """
    Main pipeline for GraphRAG with custom LLM.
    """
    
    def __init__(
        self,
        llm_client: CustomLLMClient,
        embeddings_client: CustomEmbeddingsClient,
        root_dir: str = "./graphrag_output"
    ):
        """
        Initialize GraphRAG pipeline.
        
        Args:
            llm_client: Custom LLM client
            embeddings_client: Custom embeddings client
            root_dir: Output directory for GraphRAG artifacts
        """
        self.llm_client = llm_client
        self.embeddings_client = embeddings_client
        self.root_dir = root_dir
        self.config = create_graphrag_config(llm_client, embeddings_client, root_dir)
        
        # Create directories
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(f"{root_dir}/input", exist_ok=True)
        os.makedirs(f"{root_dir}/output", exist_ok=True)
        
        logger.info(f"GraphRAG pipeline initialized at: {root_dir}")
    
    def load_documents(self, documents_dir: str) -> List[Dict[str, str]]:
        """
        Load documents from directory.
        
        Args:
            documents_dir: Directory containing .txt files
            
        Returns:
            List of document dictionaries
        """
        documents = []
        doc_path = Path(documents_dir)
        
        if not doc_path.exists():
            logger.error(f"Directory not found: {documents_dir}")
            return documents
        
        txt_files = list(doc_path.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        for idx, filepath in enumerate(txt_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # GraphRAG expects documents in specific format
                        doc = {
                            'id': f"doc_{idx}",
                            'text': content,
                            'title': filepath.stem,
                        }
                        documents.append(doc)
                        logger.info(f"Loaded: {filepath.name} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        return documents
    
    async def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents using GraphRAG.
        This builds the knowledge graph with entities and relationships.
        
        Args:
            documents: List of document dictionaries
        """
        logger.info(f"Starting GraphRAG indexing for {len(documents)} documents...")
        
        # Save documents to input directory in GraphRAG format
        input_dir = f"{self.root_dir}/input"
        
        # Create a CSV file with documents (GraphRAG expects CSV)
        df = pd.DataFrame(documents)
        input_file = f"{input_dir}/documents.csv"
        df.to_csv(input_file, index=False)
        logger.info(f"Documents saved to: {input_file}")
        
        # Create pipeline configuration
        pipeline_config = create_pipeline_config(self.config)
        
        # Inject custom LLM and embeddings into the pipeline
        # This ensures GraphRAG uses YOUR clients for all operations
        pipeline_config['llm'] = self.llm_client
        pipeline_config['embeddings'] = self.embeddings_client
        
        try:
            # Run the GraphRAG indexing pipeline
            # This will:
            # 1. Extract entities using your LLM (with GraphRAG's prompts)
            # 2. Extract relationships using your LLM
            # 3. Build communities
            # 4. Generate embeddings using your embeddings client
            # 5. Create the knowledge graph
            
            logger.info("Running GraphRAG pipeline (this may take several minutes)...")
            await run_pipeline_with_config(pipeline_config)
            
            logger.info("GraphRAG indexing complete!")
            logger.info(f"Artifacts saved to: {self.root_dir}/output")
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise
    
    def load_graph_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the generated graph data.
        
        Returns:
            Dictionary of DataFrames with entities, relationships, etc.
        """
        output_dir = f"{self.root_dir}/output"
        
        data = {}
        
        # Load entities
        entities_file = f"{output_dir}/create_final_entities.parquet"
        if os.path.exists(entities_file):
            data['entities'] = pd.read_parquet(entities_file)
            logger.info(f"Loaded {len(data['entities'])} entities")
        
        # Load relationships
        relationships_file = f"{output_dir}/create_final_relationships.parquet"
        if os.path.exists(relationships_file):
            data['relationships'] = pd.read_parquet(relationships_file)
            logger.info(f"Loaded {len(data['relationships'])} relationships")
        
        # Load communities
        communities_file = f"{output_dir}/create_final_communities.parquet"
        if os.path.exists(communities_file):
            data['communities'] = pd.read_parquet(communities_file)
            logger.info(f"Loaded {len(data['communities'])} communities")
        
        # Load community reports
        reports_file = f"{output_dir}/create_final_community_reports.parquet"
        if os.path.exists(reports_file):
            data['reports'] = pd.read_parquet(reports_file)
            logger.info(f"Loaded {len(data['reports'])} community reports")
        
        return data
    
    async def query_global(self, question: str) -> str:
        """
        Perform global search query.
        Good for questions about overall themes and trends.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        logger.info(f"Global query: {question}")
        
        # Load graph data
        data = self.load_graph_data()
        
        if 'reports' not in data:
            return "Graph not indexed yet. Please run index_documents first."
        
        # Initialize global search
        search = GlobalSearch(
            llm=self.llm_client,
            context_builder_params={
                "data": data['reports'],
            }
        )
        
        # Execute search
        result = await search.asearch(question)
        
        return result.response
    
    async def query_local(self, question: str) -> str:
        """
        Perform local search query.
        Good for specific questions about entities.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        logger.info(f"Local query: {question}")
        
        # Load graph data
        data = self.load_graph_data()
        
        if 'entities' not in data:
            return "Graph not indexed yet. Please run index_documents first."
        
        # Initialize local search
        search = LocalSearch(
            llm=self.llm_client,
            context_builder_params={
                "entities": data['entities'],
                "relationships": data.get('relationships'),
                "reports": data.get('reports'),
            }
        )
        
        # Execute search
        result = await search.asearch(question)
        
        return result.response
    
    def print_statistics(self):
        """Print statistics about the knowledge graph."""
        data = self.load_graph_data()
        
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*80)
        
        if 'entities' in data:
            print(f"\n📊 Total Entities: {len(data['entities'])}")
            if 'type' in data['entities'].columns:
                print("\nEntity Types:")
                type_counts = data['entities']['type'].value_counts()
                for entity_type, count in type_counts.items():
                    print(f"  - {entity_type}: {count}")
        
        if 'relationships' in data:
            print(f"\n🔗 Total Relationships: {len(data['relationships'])}")
            if 'type' in data['relationships'].columns:
                print("\nRelationship Types:")
                rel_counts = data['relationships']['type'].value_counts().head(10)
                for rel_type, count in rel_counts.items():
                    print(f"  - {rel_type}: {count}")
        
        if 'communities' in data:
            print(f"\n👥 Total Communities: {len(data['communities'])}")
        
        if 'reports' in data:
            print(f"\n📄 Community Reports: {len(data['reports'])}")
        
        print("="*80 + "\n")


# ==================== MAIN APPLICATION ====================

async def main():
    """Main application."""
    
    print("="*80)
    print("Microsoft GraphRAG with Custom LLM Integration")
    print("="*80)
    
    # Step 1: Configure your custom LLM
    print("\n[1] Initializing Custom LLM...")
    
    # TODO: Update with your actual credentials
    llm_client = CustomLLMClient(
        api_key="your-api-key-here",
        model="your-llm-model-name",  # e.g., "gpt-4", "claude-3-opus"
        api_base=None,  # Optional: your API endpoint
        temperature=0.0,  # Lower for factual extraction
        max_tokens=4000,
    )
    
    # Step 2: Configure your custom embeddings
    print("[2] Initializing Custom Embeddings...")
    
    embeddings_client = CustomEmbeddingsClient(
        api_key="your-api-key-here",
        model="your-embedding-model",  # e.g., "text-embedding-ada-002"
        api_base=None,
    )
    
    # Step 3: Initialize GraphRAG pipeline
    print("[3] Initializing GraphRAG Pipeline...")
    
    pipeline = GraphRAGPipeline(
        llm_client=llm_client,
        embeddings_client=embeddings_client,
        root_dir="./graphrag_output"
    )
    
    # Step 4: Load documents
    print("\n[4] Loading Documents...")
    documents_dir = "./documents"
    
    # Create sample documents if directory doesn't exist
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created directory: {documents_dir}")
        
        # Create sample documents
        sample_docs = {
            "ai_research.txt": """Microsoft Research developed GraphRAG as an advanced retrieval-augmented generation system. 
The project combines knowledge graphs with large language models to improve information retrieval.
GraphRAG uses entity extraction and relationship mapping to build structured representations of documents.
The system was designed to handle complex queries that require understanding of multiple interconnected concepts.""",
            
            "tech_companies.txt": """OpenAI created GPT models and ChatGPT, revolutionizing natural language processing.
Microsoft partnered with OpenAI and invested billions to integrate AI into their products.
Google developed LaMDA and Bard, competing in the conversational AI space.
Anthropic, founded by former OpenAI researchers, created Claude AI with a focus on safety.
These companies are driving innovation in large language model development.""",
            
            "python_ecosystem.txt": """Python is widely used for machine learning and data science applications.
PyTorch and TensorFlow are the dominant deep learning frameworks in the Python ecosystem.
NumPy and Pandas provide essential tools for numerical computing and data manipulation.
Hugging Face transformed NLP by providing easy access to pre-trained models through Python.
The Python community continues to develop tools that make AI more accessible."""
        }
        
        for filename, content in sample_docs.items():
            with open(os.path.join(documents_dir, filename), 'w') as f:
                f.write(content)
        print(f"Created {len(sample_docs)} sample documents")
    
    documents = pipeline.load_documents(documents_dir)
    
    if not documents:
        print("No documents found. Add .txt files to the documents directory.")
        return
    
    # Step 5: Index documents with GraphRAG
    print(f"\n[5] Building Knowledge Graph with GraphRAG...")
    print("⚠️  This uses GraphRAG's built-in prompts for entity/relationship extraction")
    print("⚠️  Make sure you've uncommented the SDK code in Custom classes!\n")
    
    await pipeline.index_documents(documents)
    
    # Step 6: Display statistics
    print("\n[6] Knowledge Graph Statistics:")
    pipeline.print_statistics()
    
    # Step 7: Test queries
    print("\n[7] Testing Queries...")
    print("="*80)
    
    test_questions = [
        ("What is GraphRAG and how does it work?", "global"),
        ("Which companies are developing AI models?", "global"),
        ("What is the relationship between Microsoft and OpenAI?", "local"),
        ("What frameworks are used for deep learning in Python?", "local"),
    ]
    
    for idx, (question, search_type) in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx} ({search_type.upper()} SEARCH): {question}")
        print(f"{'='*80}")
        
        try:
            if search_type == "global":
                answer = await pipeline.query_global(question)
            else:
                answer = await pipeline.query_local(question)
            
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(1)
    
    # Step 8: Interactive mode
    print("\n" + "="*80)
    print("Interactive Query Mode")
    print("Commands: 'global:<question>' or 'local:<question>' or 'stats' or 'exit'")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nYour query: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                pipeline.print_statistics()
                continue
            
            if not user_input:
                continue
            
            # Parse search type
            if user_input.startswith('global:'):
                question = user_input[7:].strip()
                answer = await pipeline.query_global(question)
            elif user_input.startswith('local:'):
                question = user_input[6:].strip()
                answer = await pipeline.query_local(question)
            else:
                # Default to global search
                answer = await pipeline.query_global(user_input)
            
            print(f"\n--- Answer ---\n{answer}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("\n⚠️  SETUP REQUIRED:")
    print("1. Install: pip install graphrag numpy pandas tiktoken")
    print("2. Uncomment SDK code in CustomLLMClient and CustomEmbeddingsClient")
    print("3. Update API keys in main()")
    print("4. Add .txt files to ./documents folder\n")
    
    asyncio.run(main())