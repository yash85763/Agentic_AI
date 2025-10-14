from openai import OpenAI
import os
from typing import List, Optional
import time
from dotenv import load_dotenv
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI Embedding Generator
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env variable)
            model: Embedding model to use (default: text-embedding-3-small)
                   Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        print(f"✓ Embedding Generator initialized with model: {model}")
    
    def generate_embedding(self, text: str, max_tokens: int = 8000) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            max_tokens: Maximum tokens to process (truncate if longer)
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        if not text or not text.strip():
            print("  ⚠ Empty text, skipping embedding")
            return None
        
        try:
            # Truncate text if too long (rough estimation: 1 token ≈ 4 characters)
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]
                print(f"  ⚠ Text truncated to {max_chars} characters")
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            print(f"  ✗ Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], max_tokens: int = 8000, 
                                  delay: float = 0.1) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts with rate limiting
        
        Args:
            texts: List of texts to embed
            max_tokens: Maximum tokens per text
            delay: Delay between requests in seconds
            
        Returns:
            List of embeddings (same order as input texts)
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"  Generating embedding {i+1}/{len(texts)}...")
            embedding = self.generate_embedding(text, max_tokens)
            embeddings.append(embedding)
            
            # Rate limiting
            if i < len(texts) - 1:  # Don't delay after last request
                time.sleep(delay)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)
    


if __name__ == "__main__":
    # Example usage
    api_key = os.getenv('OPENAI_API_KEY')
    generator = EmbeddingGenerator(api_key=api_key, model="text-embedding-3-small")
    texts = [
        "Hello world!",
        "This is a test of the OpenAI embedding API.",
        "Embeddings are useful for many NLP tasks."
    ]
    embeddings = generator.generate_embeddings_batch(texts, delay=0.5)
    for i, emb in enumerate(embeddings):
        if emb:
            print(f"Embedding {i+1} (dim {len(emb)}): {emb[:5]}...")  # Print first 5 values
        else:
            print(f"Embedding {i+1}: Failed to generate")
                
