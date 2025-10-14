import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from embeddings import EmbeddingGenerator
from openai import OpenAI
import os

class GraphSimilaritySearch:
    def __init__(self, graph_json_path: str, openai_api_key: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o-mini"):
        """
        Initialize Graph Similarity Search
        
        Args:
            graph_json_path: Path to the graph.json file
            openai_api_key: OpenAI API key
            embedding_model: Embedding model to use
            llm_model: LLM model for query decomposition
        """
        self.graph_json_path = graph_json_path
        self.graph_data = None
        self.nodes_with_embeddings = []
        self.llm_model = llm_model
        
        # Load graph data
        self.load_graph()
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            api_key=openai_api_key,
            model=embedding_model
        )
        
        # Initialize OpenAI client for query decomposition
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        
        print(f"✓ Graph loaded: {len(self.nodes_with_embeddings)} nodes with embeddings")
        print(f"✓ LLM initialized: {llm_model}")
    
    def load_graph(self):
        """Load graph data from JSON file"""
        try:
            with open(self.graph_json_path, 'r', encoding='utf-8') as f:
                self.graph_data = json.load(f)
            
            # Filter nodes that have embeddings
            for node in self.graph_data['nodes']:
                if node.get('embeddings') and isinstance(node['embeddings'], list):
                    self.nodes_with_embeddings.append(node)
            
            if not self.nodes_with_embeddings:
                print("⚠ Warning: No nodes with embeddings found in the graph!")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {self.graph_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {self.graph_json_path}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def decompose_query(self, query: str, website_context: str = "myucretirement.com") -> Dict:
        """
        Decompose user query into sub-queries and reconstruct for better search
        
        Args:
            query: Original user query
            website_context: Website domain for context
            
        Returns:
            Dictionary with reconstructed query and sub-queries
        """
        print(f"\n{'='*80}")
        print(f"QUERY DECOMPOSITION & RECONSTRUCTION")
        print(f"{'='*80}")
        print(f"Original query: {query}")
        print(f"{'='*80}\n")
        
        prompt = f"""You are helping users find information on {website_context}, a website about retirement planning, UC retirement benefits, pension plans, 401k, 403b, 457b plans, IRA accounts, and financial education for retirement.

USER QUERY: "{query}"

TASK:
1. Reconstruct the query to be clearer and more specific for retirement planning search
2. Decompose the query into 2-5 focused sub-queries that cover different aspects
3. Identify key retirement concepts mentioned

CONTEXT ABOUT THE WEBSITE:
- UC Retirement Plans (UCRP, DC Plan, 403b, 457b)
- Pension vs 401k/403b differences
- Traditional vs Roth contributions
- Catch-up contributions
- Retirement age and vesting
- Tax implications
- Beneficiary designations
- Required Minimum Distributions (RMDs)
- Investment strategies
- Social Security coordination

OUTPUT FORMAT (JSON):
{{
  "reconstructed_query": "Clear, specific version of the query",
  "intent": "What the user wants to know",
  "sub_queries": [
    "Focused sub-query 1",
    "Focused sub-query 2",
    "Focused sub-query 3"
  ],
  "key_concepts": ["concept1", "concept2"],
  "expected_pages": ["type of page that might answer this"]
}}

EXAMPLES:
Query: "How much can I put in my 401k?"
Output:
{{
  "reconstructed_query": "What are the 401k contribution limits and how do I maximize contributions?",
  "intent": "Find contribution limits and strategies",
  "sub_queries": [
    "401k annual contribution limits",
    "401k catch-up contributions for age 50+",
    "employer match limits"
  ],
  "key_concepts": ["401k", "contribution limits", "catch-up contributions"],
  "expected_pages": ["contribution limits page", "401k overview", "benefits calculator"]
}}

Query: "Should I do Roth or traditional?"
Output:
{{
  "reconstructed_query": "What are the differences between Roth and Traditional retirement contributions and which is better for my situation?",
  "intent": "Compare Roth vs Traditional to make a decision",
  "sub_queries": [
    "Roth 401k tax advantages",
    "Traditional 401k tax deductions",
    "Roth vs Traditional comparison for different income levels",
    "future tax implications Roth Traditional"
  ],
  "key_concepts": ["Roth", "Traditional", "tax implications", "comparison"],
  "expected_pages": ["Roth vs Traditional comparison", "tax implications", "contribution types"]
}}

Now decompose the user's query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at understanding retirement planning questions and breaking them into searchable components."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            print(f"→ Reconstructed Query: {result.get('reconstructed_query', query)}")
            print(f"→ Intent: {result.get('intent', 'Unknown')}")
            print(f"→ Key Concepts: {', '.join(result.get('key_concepts', []))}")
            print(f"\n→ Sub-queries ({len(result.get('sub_queries', []))}):")
            for i, sq in enumerate(result.get('sub_queries', []), 1):
                print(f"  {i}. {sq}")
            print()
            
            return result
            
        except Exception as e:
            print(f"✗ Error in query decomposition: {e}")
            # Fallback to original query
            return {
                "reconstructed_query": query,
                "intent": "Unknown",
                "sub_queries": [query],
                "key_concepts": [],
                "expected_pages": []
            }
    
    def search_with_decomposition(self, query: str, top_k_per_query: int = 3,
                                  min_similarity: float = 0.5) -> Dict:
        """
        Search using query decomposition for comprehensive results
        
        Args:
            query: Original user query
            top_k_per_query: Results per sub-query
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with all search results and metadata
        """
        # Step 1: Decompose query
        decomposition = self.decompose_query(query)
        
        # Step 2: Search with reconstructed query
        print(f"{'='*80}")
        print(f"SEARCHING WITH RECONSTRUCTED QUERY")
        print(f"{'='*80}\n")
        
        reconstructed_results = self.search_similar_nodes(
            decomposition['reconstructed_query'],
            top_k=top_k_per_query,
            min_similarity=min_similarity
        )
        
        # Step 3: Search with each sub-query
        print(f"{'='*80}")
        print(f"SEARCHING WITH SUB-QUERIES")
        print(f"{'='*80}\n")
        
        sub_query_results = {}
        all_results_map = {}  # URL -> result dict
        
        for i, sub_query in enumerate(decomposition['sub_queries'], 1):
            print(f"Sub-query {i}/{len(decomposition['sub_queries'])}: {sub_query}")
            results = self.search_similar_nodes(
                sub_query,
                top_k=top_k_per_query,
                min_similarity=min_similarity
            )
            sub_query_results[sub_query] = results
            
            # Add to aggregated results
            for result in results:
                url = result['url']
                if url not in all_results_map:
                    all_results_map[url] = result.copy()
                    all_results_map[url]['matched_queries'] = []
                    all_results_map[url]['total_score'] = 0
                    all_results_map[url]['max_similarity'] = result['similarity']
                
                all_results_map[url]['matched_queries'].append({
                    'query': sub_query,
                    'similarity': result['similarity']
                })
                all_results_map[url]['total_score'] += result['similarity']
                all_results_map[url]['max_similarity'] = max(
                    all_results_map[url]['max_similarity'],
                    result['similarity']
                )
        
        # Step 4: Rank aggregated results
        aggregated_results = list(all_results_map.values())
        
        # Sort by: number of matched queries (desc), then total score (desc)
        aggregated_results.sort(
            key=lambda x: (len(x['matched_queries']), x['total_score']),
            reverse=True
        )
        
        return {
            'original_query': query,
            'decomposition': decomposition,
            'reconstructed_results': reconstructed_results,
            'sub_query_results': sub_query_results,
            'aggregated_results': aggregated_results[:10],  # Top 10 overall
            'total_unique_pages': len(aggregated_results)
        }
    
    def print_decomposed_results(self, search_results: Dict):
        """
        Pretty print results from decomposed search
        
        Args:
            search_results: Results from search_with_decomposition
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE SEARCH RESULTS")
        print(f"{'='*80}")
        print(f"Original Query: {search_results['original_query']}")
        print(f"Total Unique Pages Found: {search_results['total_unique_pages']}")
        print(f"{'='*80}\n")
        
        print("TOP 10 MOST RELEVANT PAGES:")
        print("-" * 80)
        
        for i, result in enumerate(search_results['aggregated_results'], 1):
            print(f"\n{i}. {result['label']}")
            print(f"   URL: {result['url']}")
            print(f"   Max Similarity: {result['max_similarity']:.4f}")
            print(f"   Matched {len(result['matched_queries'])} sub-queries:")
            
            for match in result['matched_queries']:
                print(f"     • [{match['similarity']:.3f}] {match['query']}")
            
            if result.get('text_preview'):
                preview = result['text_preview']
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                print(f"   Preview: {preview}")
        
        print(f"\n{'='*80}\n")
    
    def generate_answer(self, query: str, search_results: Dict, 
                       top_n_pages: int = 5) -> str:
        """
        Generate a comprehensive answer using search results
        
        Args:
            query: Original query
            search_results: Results from search_with_decomposition
            top_n_pages: Number of top pages to use for answer
            
        Returns:
            Generated answer
        """
        print(f"\n{'='*80}")
        print(f"GENERATING ANSWER")
        print(f"{'='*80}\n")
        
        # Get top pages
        top_pages = search_results['aggregated_results'][:top_n_pages]
        
        # Build context from pages
        context_parts = []
        for i, page in enumerate(top_pages, 1):
            # Get full text from graph
            full_node = self.get_node_details(page['url'])
            if full_node:
                text = full_node.get('text', '')[:1500]  # First 1500 chars
                context_parts.append(f"SOURCE {i} - {page['label']}:\n{text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a retirement planning expert helping users understand their UC retirement benefits.

USER QUESTION: "{query}"

RELEVANT INFORMATION FROM UC RETIREMENT WEBSITE:

{context}

TASK:
Provide a clear, comprehensive answer to the user's question based on the information above.

GUIDELINES:
1. Answer directly and concisely
2. Use specific numbers and details from the sources
3. Organize information clearly with sections if needed
4. If comparing options (like Roth vs Traditional), present both sides fairly
5. Include important caveats or conditions
6. Mention if consultation with HR or financial advisor is recommended for personalized advice
7. Cite which source(s) you're using (e.g., "According to the 401k Overview page...")

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at explaining retirement planning in clear, helpful terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            answer = response.choices[0].message.content
            
            print(f"✓ Answer generated\n")
            print(f"{'='*80}")
            print(f"ANSWER:")
            print(f"{'='*80}\n")
            print(answer)
            print(f"\n{'='*80}")
            print(f"Sources used: {', '.join([p['label'] for p in top_pages])}")
            print(f"{'='*80}\n")
            
            return answer
            
        except Exception as e:
            print(f"✗ Error generating answer: {e}")
            return "Unable to generate answer. Please review the search results above."
    
    def search_similar_nodes(self, query: str, top_k: int = 5, 
                            min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for nodes similar to the query
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (0 to 1)
            
        Returns:
            List of dictionaries containing node info and similarity scores
        """
        if not self.nodes_with_embeddings:
            print("✗ No nodes with embeddings available for search")
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        if not query_embedding:
            print("✗ Failed to generate query embedding")
            return []
        
        # Calculate similarities
        results = []
        
        for node in self.nodes_with_embeddings:
            node_embedding = node['embeddings']
            similarity = self.cosine_similarity(query_embedding, node_embedding)
            
            if similarity >= min_similarity:
                results.append({
                    'node_id': node['id'],
                    'url': node['url'],
                    'label': node['label'],
                    'similarity': similarity,
                    'text_preview': node.get('text', '')[:200] + '...' if node.get('text') else '',
                    'depth': node.get('depth', 0),
                    'is_external': node.get('is_external', False)
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def print_results(self, results: List[Dict], show_text_preview: bool = True):
        """
        Pretty print search results
        
        Args:
            results: List of search results
            show_text_preview: Whether to show text preview
        """
        if not results:
            print("No results found.")
            return
        
        print(f"{'='*80}")
        print(f"SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['similarity']:.4f}] {result['label']}")
            print(f"   URL: {result['url']}")
            print(f"   Depth: {result['depth']} | External: {result['is_external']}")
            
            if show_text_preview and result['text_preview']:
                print(f"   Preview: {result['text_preview']}")
            
            print()
        
        print(f"{'='*80}\n")
    
    def get_node_details(self, node_id: str) -> Optional[Dict]:
        """
        Get full details of a specific node
        
        Args:
            node_id: Node ID (URL)
            
        Returns:
            Node data dictionary or None if not found
        """
        for node in self.graph_data['nodes']:
            if node['id'] == node_id:
                return node
        return None
    
    def search_and_display(self, query: str, top_k: int = 5, 
                          min_similarity: float = 0.0, show_text_preview: bool = True):
        """
        Search and display results in one call
        
        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            show_text_preview: Show text preview
        """
        results = self.search_similar_nodes(query, top_k, min_similarity)
        self.print_results(results, show_text_preview)
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search multiple queries at once
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        print(f"\n{'='*80}")
        print(f"BATCH SEARCH: {len(queries)} queries")
        print(f"{'='*80}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query}")
            query_results = self.search_similar_nodes(query, top_k)
            results[query] = query_results
            print()
        
        return results
    
    def find_related_nodes(self, node_id: str, top_k: int = 5, 
                          exclude_self: bool = True) -> List[Dict]:
        """
        Find nodes similar to a given node (by its embedding)
        
        Args:
            node_id: Node ID (URL) to find similar nodes for
            top_k: Number of similar nodes to return
            exclude_self: Exclude the query node from results
            
        Returns:
            List of similar nodes
        """
        # Find the node
        query_node = None
        for node in self.nodes_with_embeddings:
            if node['id'] == node_id:
                query_node = node
                break
        
        if not query_node:
            print(f"✗ Node not found: {node_id}")
            return []
        
        print(f"\n{'='*80}")
        print(f"FINDING RELATED NODES")
        print(f"{'='*80}")
        print(f"Source node: {query_node['label']}")
        print(f"URL: {query_node['url']}")
        print(f"{'='*80}\n")
        
        # Calculate similarities
        query_embedding = query_node['embeddings']
        results = []
        
        for node in self.nodes_with_embeddings:
            # Skip self if requested
            if exclude_self and node['id'] == node_id:
                continue
            
            node_embedding = node['embeddings']
            similarity = self.cosine_similarity(query_embedding, node_embedding)
            
            results.append({
                'node_id': node['id'],
                'url': node['url'],
                'label': node['label'],
                'similarity': similarity,
                'text_preview': node.get('text', '')[:200] + '...' if node.get('text') else '',
                'depth': node.get('depth', 0),
                'is_external': node.get('is_external', False)
            })
        
        # Sort and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:top_k]
        
        print(f"✓ Found {len(top_results)} related nodes\n")
        
        return top_results


def main():
    """Interactive similarity search CLI"""
    print("\n" + "="*80)
    print("RETIREMENT PLANNING GRAPH SEARCH - myucretirement.com")
    print("="*80 + "\n")
    
    # Get graph file path
    graph_path = input("Enter path to graph.json (default: kg_output/graph.json): ").strip()
    if not graph_path:
        graph_path = "kg_output/graph.json"
    
    # Get OpenAI API key
    api_key = input("OpenAI API Key (or press Enter to use env variable): ").strip() or None
    
    # Initialize search
    try:
        searcher = GraphSimilaritySearch(
            graph_json_path=graph_path,
            openai_api_key=api_key
        )
    except Exception as e:
        print(f"✗ Error initializing search: {e}")
        return
    
    # Interactive search loop
    while True:
        print("\n" + "="*80)
        print("OPTIONS:")
        print("  1. Search with query decomposition (Recommended)")
        print("  2. Simple search by query")
        print("  3. Find related nodes (by URL)")
        print("  4. Batch search")
        print("  5. Exit")
        print("="*80)
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == "1":
            # Query decomposition search (NEW!)
            query = input("\nEnter your retirement planning question: ").strip()
            if not query:
                print("Empty query, skipping...")
                continue
            
            top_k = input("Results per sub-query (default: 3): ").strip()
            top_k = int(top_k) if top_k else 3
            
            min_sim = input("Minimum similarity 0-1 (default: 0.5): ").strip()
            min_sim = float(min_sim) if min_sim else 0.5
            
            generate_answer = input("Generate comprehensive answer? (y/n, default: y): ").strip().lower()
            generate_answer = generate_answer != 'n'
            
            # Search with decomposition
            results = searcher.search_with_decomposition(query, top_k, min_sim)
            searcher.print_decomposed_results(results)
            
            # Optionally generate answer
            if generate_answer:
                searcher.generate_answer(query, results)
        
        elif choice == "2":
            # Simple query search
            query = input("\nEnter search query: ").strip()
            if not query:
                print("Empty query, skipping...")
                continue
            
            top_k = input("Number of results (default: 5): ").strip()
            top_k = int(top_k) if top_k else 5
            
            min_sim = input("Minimum similarity threshold 0-1 (default: 0.0): ").strip()
            min_sim = float(min_sim) if min_sim else 0.0
            
            searcher.search_and_display(query, top_k, min_sim)
        
        elif choice == "3":
            # Find related nodes
            node_url = input("\nEnter node URL: ").strip()
            if not node_url:
                print("Empty URL, skipping...")
                continue
            
            top_k = input("Number of results (default: 5): ").strip()
            top_k = int(top_k) if top_k else 5
            
            results = searcher.find_related_nodes(node_url, top_k)
            searcher.print_results(results)
        
        elif choice == "4":
            # Batch search
            print("\nEnter queries (one per line, empty line to finish):")
            queries = []
            while True:
                query = input("  Query: ").strip()
                if not query:
                    break
                queries.append(query)
            
            if not queries:
                print("No queries provided, skipping...")
                continue
            
            top_k = input("Number of results per query (default: 5): ").strip()
            top_k = int(top_k) if top_k else 5
            
            batch_results = searcher.batch_search(queries, top_k)
            
            # Display all results
            for query, results in batch_results.items():
                print(f"\n{'='*80}")
                print(f"Query: {query}")
                print(f"{'='*80}")
                searcher.print_results(results, show_text_preview=False)
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()


# ## Key Features Added:

# ### 1. **Query Decomposition** (`decompose_query`)
# - Uses LLM to understand retirement planning context
# - Reconstructs vague queries into clear, specific ones
# - Breaks down complex questions into 2-5 focused sub-queries
# - Identifies key retirement concepts (401k, Roth, RMD, etc.)

# ### 2. **Multi-Query Search** (`search_with_decomposition`)
# - Searches with reconstructed query
# - Searches each sub-query independently
# - Aggregates results across all queries
# - Ranks by: # of matched queries + total similarity score

# ### 3. **Answer Generation** (`generate_answer`)
# - Retrieves full text from top 5 relevant pages
# - Uses LLM to synthesize comprehensive answer
# - Cites sources
# - Provides retirement-specific guidance

# ### 4. **Smart Result Aggregation**
# - Tracks which sub-queries matched each page
# - Shows similarity scores for each match
# - Prioritizes pages that answer multiple aspects

