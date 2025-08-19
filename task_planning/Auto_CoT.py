"""
Improved Auto-CoT (Automatic Chain of Thought) Implementation

This is an enhanced implementation of the Auto-CoT algorithm from the paper:
"Automatic Chain of Thought Prompting in Large Language Models" 
by Zhang et al. (2022)

Key improvements over the original implementation:
1. Better error handling and logging
2. More flexible clustering methods
3. Enhanced quality control mechanisms
4. Configurable parameters for different use cases
5. Comprehensive documentation and type hints
6. Support for different embedding models
7. Async processing for better performance
8. Better validation and filtering heuristics
"""

import json
import logging
import asyncio
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import openai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Supported clustering methods"""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    AGGLOMERATIVE = "agglomerative"


class EmbeddingModel(Enum):
    """Supported embedding models"""
    SENTENCE_BERT = "all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "all-mpnet-base-v2"
    PARAPHRASE_BERT = "paraphrase-MiniLM-L6-v2"


@dataclass
class AutoCoTConfig:
    """Configuration class for Auto-CoT algorithm"""
    
    # Core algorithm parameters
    num_clusters: int = 8
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_BERT
    
    # Quality control parameters
    min_question_length: int = 10  # minimum tokens in question
    max_question_length: int = 60  # maximum tokens in question
    min_reasoning_steps: int = 2   # minimum steps in reasoning chain
    max_reasoning_steps: int = 8   # maximum steps in reasoning chain
    
    # LLM parameters
    temperature: float = 0.0
    max_tokens_reasoning: int = 256
    max_tokens_answer: int = 32
    api_delay: float = 1.0  # delay between API calls
    
    # Prompts
    zero_shot_cot_trigger: str = "Let's think step by step."
    answer_trigger: str = "Therefore, the answer is"
    
    # Output settings
    save_intermediate_results: bool = True
    output_dir: Path = Path("auto_cot_output")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.num_clusters < 2:
            raise ValueError("Number of clusters must be at least 2")
        if self.min_question_length >= self.max_question_length:
            raise ValueError("min_question_length must be less than max_question_length")
        if self.min_reasoning_steps >= self.max_reasoning_steps:
            raise ValueError("min_reasoning_steps must be less than max_reasoning_steps")


@dataclass
class Question:
    """Represents a question with metadata"""
    text: str
    id: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    
    def __post_init__(self):
        if not self.tokens:
            # Simple tokenization - could be enhanced with proper tokenizer
            self.tokens = self.text.split()


@dataclass
class Demonstration:
    """Represents a CoT demonstration"""
    question: str
    reasoning_chain: str
    final_answer: str
    quality_score: float = 0.0
    num_reasoning_steps: int = 0
    
    def __post_init__(self):
        # Count reasoning steps (simple heuristic)
        self.num_reasoning_steps = len([
            line for line in self.reasoning_chain.split('.')
            if line.strip() and any(word in line.lower() for word in ['so', 'then', 'therefore', 'thus', 'because'])
        ])


class QualityFilter:
    """Enhanced quality filtering for demonstrations"""
    
    @staticmethod
    def has_valid_reasoning_structure(reasoning: str) -> bool:
        """Check if reasoning has valid step-by-step structure"""
        reasoning_indicators = [
            'first', 'second', 'then', 'next', 'so', 'therefore',
            'thus', 'because', 'since', 'step', 'now'
        ]
        
        words = reasoning.lower().split()
        indicator_count = sum(1 for word in words if word in reasoning_indicators)
        
        # At least 2 reasoning indicators for valid structure
        return indicator_count >= 2
    
    @staticmethod
    def has_mathematical_consistency(question: str, reasoning: str, answer: str) -> bool:
        """Check mathematical consistency (basic implementation)"""
        # Extract numbers from question and reasoning
        question_numbers = re.findall(r'\d+(?:\.\d+)?', question)
        reasoning_numbers = re.findall(r'\d+(?:\.\d+)?', reasoning)
        answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        
        # Basic check: answer should contain at least one number
        if not answer_numbers:
            return False
            
        # Check if reasoning uses numbers from question
        if question_numbers and reasoning_numbers:
            question_set = set(question_numbers)
            reasoning_set = set(reasoning_numbers)
            return len(question_set.intersection(reasoning_set)) > 0
            
        return True
    
    @staticmethod
    def calculate_quality_score(demonstration: Demonstration) -> float:
        """Calculate overall quality score for demonstration"""
        score = 0.0
        
        # Structure score (0-0.4)
        if QualityFilter.has_valid_reasoning_structure(demonstration.reasoning_chain):
            score += 0.4
        
        # Length score (0-0.3)
        reasoning_length = len(demonstration.reasoning_chain.split())
        if 10 <= reasoning_length <= 100:  # reasonable length
            score += 0.3
        elif reasoning_length > 5:
            score += 0.1
        
        # Mathematical consistency (0-0.3)
        if QualityFilter.has_mathematical_consistency(
            demonstration.question, 
            demonstration.reasoning_chain, 
            demonstration.final_answer
        ):
            score += 0.3
        
        return score


class AutoCoT:
    """
    Enhanced Auto-CoT implementation with the following algorithm:
    
    Algorithm Overview:
    ==================
    
    Auto-CoT (Automatic Chain of Thought) is a two-stage algorithm that automatically 
    generates high-quality demonstrations for Chain-of-Thought prompting without 
    manual effort.
    
    Stage 1: Question Clustering
    ---------------------------
    1. Encode all questions using Sentence-BERT to get semantic embeddings
    2. Apply clustering (K-means or Hierarchical) to group similar questions
    3. This ensures diversity in demonstrations across different question types
    
    Stage 2: Demonstration Sampling & Generation
    -------------------------------------------
    1. For each cluster, select a representative question using heuristics:
       - Avoid questions that are too short or too long
       - Prefer questions with moderate complexity
    2. Generate reasoning chain using Zero-Shot-CoT ("Let's think step by step")
    3. Apply quality control filters:
       - Check reasoning structure and coherence
       - Validate mathematical consistency
       - Ensure appropriate length and step count
    4. Select best demonstrations based on quality scores
    
    Key Innovations:
    ===============
    1. Diversity through clustering ensures broad coverage of question types
    2. Quality control reduces errors in auto-generated demonstrations
    3. Heuristic-based selection improves demonstration relevance
    4. Scalable approach eliminates manual demonstration crafting
    """
    
    def __init__(self, config: AutoCoTConfig, openai_client=None):
        self.config = config
        self.openai_client = openai_client or openai
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding_model.value}")
        self.embedding_model = SentenceTransformer(config.embedding_model.value)
        
        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize quality filter
        self.quality_filter = QualityFilter()
        
        logger.info("Auto-CoT initialized successfully")
    
    def encode_questions(self, questions: List[Question]) -> List[Question]:
        """
        Encode questions using Sentence-BERT embeddings
        
        Args:
            questions: List of Question objects
            
        Returns:
            List of questions with embeddings populated
        """
        logger.info(f"Encoding {len(questions)} questions...")
        
        question_texts = [q.text for q in questions]
        embeddings = self.embedding_model.encode(
            question_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        for question, embedding in zip(questions, embeddings):
            question.embedding = embedding
            
        logger.info("Question encoding completed")
        return questions
    
    def cluster_questions(self, questions: List[Question]) -> List[Question]:
        """
        Cluster questions based on their embeddings for diversity
        
        Args:
            questions: List of questions with embeddings
            
        Returns:
            List of questions with cluster_id populated
        """
        logger.info(f"Clustering questions using {self.config.clustering_method.value}...")
        
        # Extract embeddings
        embeddings = np.array([q.embedding for q in questions])
        
        # Apply clustering
        if self.config.clustering_method == ClusteringMethod.KMEANS:
            clusterer = KMeans(
                n_clusters=min(self.config.num_clusters, len(questions)),
                random_state=42,
                n_init=10
            )
        elif self.config.clustering_method == ClusteringMethod.HIERARCHICAL:
            clusterer = AgglomerativeClustering(
                n_clusters=min(self.config.num_clusters, len(questions)),
                linkage='ward'
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.config.clustering_method}")
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Assign cluster IDs
        for question, cluster_id in zip(questions, cluster_labels):
            question.cluster_id = cluster_id
        
        # Log cluster statistics
        cluster_counts = {}
        for q in questions:
            cluster_counts[q.cluster_id] = cluster_counts.get(q.cluster_id, 0) + 1
        
        logger.info(f"Clustering completed. Cluster distribution: {cluster_counts}")
        return questions
    
    def select_representative_questions(self, questions: List[Question]) -> List[Question]:
        """
        Select representative questions from each cluster using heuristics
        
        Heuristics:
        1. Question length should be within reasonable bounds
        2. Avoid questions with very simple or very complex patterns
        3. Prefer questions that are close to cluster centroid
        
        Args:
            questions: List of clustered questions
            
        Returns:
            List of representative questions (one per cluster)
        """
        logger.info("Selecting representative questions from clusters...")
        
        representatives = []
        clusters = {}
        
        # Group questions by cluster
        for question in questions:
            if question.cluster_id not in clusters:
                clusters[question.cluster_id] = []
            clusters[question.cluster_id].append(question)
        
        for cluster_id, cluster_questions in clusters.items():
            logger.info(f"Selecting representative for cluster {cluster_id} ({len(cluster_questions)} questions)")
            
            # Filter questions by length heuristic
            valid_questions = []
            for q in cluster_questions:
                q_length = len(q.tokens)
                if self.config.min_question_length <= q_length <= self.config.max_question_length:
                    valid_questions.append(q)
            
            if not valid_questions:
                logger.warning(f"No valid questions in cluster {cluster_id}, using first question")
                valid_questions = cluster_questions[:1]
            
            # Select question closest to cluster centroid
            if len(valid_questions) > 1:
                cluster_embeddings = np.array([q.embedding for q in valid_questions])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                distances = [
                    np.linalg.norm(q.embedding - centroid) 
                    for q in valid_questions
                ]
                
                best_idx = np.argmin(distances)
                representative = valid_questions[best_idx]
            else:
                representative = valid_questions[0]
            
            representatives.append(representative)
            logger.info(f"Selected: '{representative.text[:50]}...'")
        
        return representatives
    
    async def generate_reasoning_chain(self, question: str) -> Tuple[str, str]:
        """
        Generate reasoning chain using Zero-Shot-CoT
        
        Args:
            question: The question to generate reasoning for
            
        Returns:
            Tuple of (reasoning_chain, final_answer)
        """
        # First, generate the reasoning chain
        reasoning_prompt = f"Q: {question}\nA: {self.config.zero_shot_cot_trigger}"
        
        try:
            reasoning_response = await self._call_openai(
                reasoning_prompt, 
                max_tokens=self.config.max_tokens_reasoning
            )
            
            # Then extract the final answer
            answer_prompt = f"{reasoning_prompt} {reasoning_response} {self.config.answer_trigger}"
            
            answer_response = await self._call_openai(
                answer_prompt,
                max_tokens=self.config.max_tokens_answer
            )
            
            return reasoning_response.strip(), answer_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating reasoning for question: {question[:50]}... Error: {e}")
            return "", ""
    
    async def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """
        Make async call to OpenAI API with error handling
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                timeout=30
            )
            
            # Add delay to respect rate limits
            await asyncio.sleep(self.config.api_delay)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
    
    async def generate_demonstrations(self, questions: List[Question]) -> List[Demonstration]:
        """
        Generate demonstrations for representative questions
        
        Args:
            questions: List of representative questions
            
        Returns:
            List of generated demonstrations
        """
        logger.info(f"Generating demonstrations for {len(questions)} questions...")
        
        demonstrations = []
        
        for i, question in enumerate(questions):
            logger.info(f"Generating demonstration {i+1}/{len(questions)}")
            
            reasoning_chain, final_answer = await self.generate_reasoning_chain(question.text)
            
            if reasoning_chain and final_answer:
                demo = Demonstration(
                    question=question.text,
                    reasoning_chain=reasoning_chain,
                    final_answer=final_answer
                )
                
                # Calculate quality score
                demo.quality_score = self.quality_filter.calculate_quality_score(demo)
                
                demonstrations.append(demo)
                logger.info(f"Generated demonstration with quality score: {demo.quality_score:.2f}")
            else:
                logger.warning(f"Failed to generate demonstration for question: {question.text[:50]}...")
        
        return demonstrations
    
    def filter_demonstrations(self, demonstrations: List[Demonstration]) -> List[Demonstration]:
        """
        Filter demonstrations based on quality criteria
        
        Args:
            demonstrations: List of generated demonstrations
            
        Returns:
            List of filtered high-quality demonstrations
        """
        logger.info(f"Filtering {len(demonstrations)} demonstrations...")
        
        # Filter by reasoning steps
        valid_demos = []
        for demo in demonstrations:
            if (self.config.min_reasoning_steps <= demo.num_reasoning_steps <= self.config.max_reasoning_steps 
                and demo.quality_score > 0.3):  # Minimum quality threshold
                valid_demos.append(demo)
        
        # Sort by quality score and take top ones
        valid_demos.sort(key=lambda x: x.quality_score, reverse=True)
        
        logger.info(f"Filtered to {len(valid_demos)} high-quality demonstrations")
        return valid_demos
    
    def save_demonstrations(self, demonstrations: List[Demonstration], filename: str = "demonstrations.json"):
        """Save demonstrations to file"""
        output_file = self.config.output_dir / filename
        
        demo_data = {
            "config": {
                "num_clusters": self.config.num_clusters,
                "clustering_method": self.config.clustering_method.value,
                "embedding_model": self.config.embedding_model.value,
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "demonstrations": [
                {
                    "question": demo.question,
                    "reasoning_chain": demo.reasoning_chain,
                    "final_answer": demo.final_answer,
                    "quality_score": demo.quality_score,
                    "num_reasoning_steps": demo.num_reasoning_steps
                }
                for demo in demonstrations
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(demonstrations)} demonstrations to {output_file}")
    
    async def run_auto_cot(self, questions: List[str]) -> List[Demonstration]:
        """
        Main Auto-CoT algorithm execution
        
        Args:
            questions: List of question strings
            
        Returns:
            List of high-quality demonstrations
        """
        logger.info("Starting Auto-CoT algorithm...")
        
        # Step 1: Prepare questions
        question_objects = [Question(text=q, id=str(i)) for i, q in enumerate(questions)]
        
        # Step 2: Encode questions
        question_objects = self.encode_questions(question_objects)
        
        # Step 3: Cluster questions for diversity
        question_objects = self.cluster_questions(question_objects)
        
        # Step 4: Select representative questions
        representatives = self.select_representative_questions(question_objects)
        
        # Step 5: Generate demonstrations
        demonstrations = await self.generate_demonstrations(representatives)
        
        # Step 6: Filter for quality
        filtered_demonstrations = self.filter_demonstrations(demonstrations)
        
        # Step 7: Save results
        if self.config.save_intermediate_results:
            self.save_demonstrations(filtered_demonstrations)
        
        logger.info(f"Auto-CoT completed successfully! Generated {len(filtered_demonstrations)} demonstrations")
        return filtered_demonstrations
    
    def create_few_shot_prompt(self, demonstrations: List[Demonstration], test_question: str) -> str:
        """
        Create few-shot prompt using generated demonstrations
        
        Args:
            demonstrations: List of demonstrations to use
            test_question: The test question to answer
            
        Returns:
            Complete few-shot prompt
        """
        prompt = ""
        
        # Add demonstrations
        for demo in demonstrations:
            prompt += f"Q: {demo.question}\n"
            prompt += f"A: {demo.reasoning_chain} {self.config.answer_trigger} {demo.final_answer}\n\n"
        
        # Add test question
        prompt += f"Q: {test_question}\n"
        prompt += f"A: {self.config.zero_shot_cot_trigger}"
        
        return prompt


# Example usage and demonstration
async def main():
    """Example usage of the improved Auto-CoT implementation"""
    
    # Sample questions for demonstration
    sample_questions = [
        "There were 10 friends playing a video game online when 7 players quit. If each player left had 8 lives, how many lives did they have total?",
        "A baker made 15 cookies. He sold 8 cookies and then made 6 more. How many cookies does he have now?",
        "Sarah has 24 stickers. She gives 7 stickers to her friend and buys 12 more. How many stickers does Sarah have now?",
        "In a classroom, there are 5 rows of desks with 6 desks in each row. How many desks are there in total?",
        "A library has 120 books. If 45 books are checked out and 18 new books are added, how many books are in the library?",
        "Tom collects 36 baseball cards. He trades 12 cards for 8 new cards. How many cards does Tom have now?",
        "A parking lot has 8 rows with 15 parking spaces in each row. How many total parking spaces are there?",
        "Emma had 50 dollars. She spent 18 dollars on lunch and earned 25 dollars from babysitting. How much money does Emma have now?",
    ]
    
    # Configure Auto-CoT
    config = AutoCoTConfig(
        num_clusters=4,
        clustering_method=ClusteringMethod.KMEANS,
        min_question_length=8,
        max_question_length=50,
        min_reasoning_steps=2,
        max_reasoning_steps=6,
        temperature=0.0,
        save_intermediate_results=True
    )
    
    # Initialize Auto-CoT
    # Note: You need to set up OpenAI API key
    # openai.api_key = "your-api-key-here"
    
    auto_cot = AutoCoT(config)
    
    # Run Auto-CoT algorithm
    try:
        demonstrations = await auto_cot.run_auto_cot(sample_questions)
        
        # Create a few-shot prompt for a new question
        test_question = "A store sold 45 apples in the morning and 38 apples in the afternoon. If they started with 100 apples, how many apples are left?"
        
        few_shot_prompt = auto_cot.create_few_shot_prompt(demonstrations, test_question)
        
        print("Generated Few-Shot Prompt:")
        print("=" * 50)
        print(few_shot_prompt)
        
    except Exception as e:
        logger.error(f"Error running Auto-CoT: {e}")
        print("Note: Make sure to set up OpenAI API key to run the full example")


if __name__ == "__main__":
    # For demonstration purposes
    print("""
    Auto-CoT Algorithm Implementation
    =================================
    
    This implementation provides an enhanced version of the Auto-CoT algorithm with:
    
    1. **Two-Stage Process:**
       - Stage 1: Question Clustering for diversity
       - Stage 2: Demonstration generation with quality control
    
    2. **Key Components:**
       - Sentence-BERT for semantic question encoding
       - Multiple clustering algorithms (K-means, Hierarchical)
       - Heuristic-based representative selection
       - Quality-aware demonstration filtering
       - Configurable parameters for different domains
    
    3. **Improvements over original:**
       - Better error handling and logging
       - Async processing for efficiency
       - Enhanced quality control mechanisms
       - Flexible configuration system
       - Comprehensive documentation
    
    To run the full example, set up OpenAI API and execute:
    asyncio.run(main())
    """)
