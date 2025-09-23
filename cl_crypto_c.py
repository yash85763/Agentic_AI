"""
Agentic RAG Implementation for Cryptocurrency Chatbot
=====================================================

This implementation uses multiple specialized agents to handle different aspects
of cryptocurrency information retrieval and persona-aware response generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==================== DATA MODELS ====================

class UserPersona(Enum):
    CRYPTO_NATIVE = "crypto_native"  # Age ~23, Intermediate crypto, Basic finance
    FINANCE_PROFESSIONAL = "finance_professional"  # Age ~40, Basic crypto, Intermediate finance  
    CONSERVATIVE_NEWCOMER = "conservative_newcomer"  # Age ~55, Basic crypto, Basic finance

@dataclass
class UserProfile:
    persona: UserPersona
    age: int
    crypto_knowledge: str
    finance_knowledge: str
    conversation_history: List[Dict]
    preferences: Dict[str, Any]

@dataclass
class QueryContext:
    user_profile: UserProfile
    query: str
    conversation_context: List[str]
    complexity_preference: str
    domain_focus: List[str]  # ['defi', 'trading', 'blockchain', etc.]

@dataclass
class AgentResponse:
    content: str
    confidence: float
    sources: List[Document]
    reasoning: str
    next_suggestions: List[str]

# ==================== SPECIALIZED AGENTS ====================

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, llm, vector_store, embeddings):
        self.name = name
        self.llm = llm
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.tools = self._create_tools()
        
    @abstractmethod
    def _create_tools(self) -> List[Tool]:
        """Create tools specific to this agent"""
        pass
    
    @abstractmethod
    def can_handle(self, query_context: QueryContext) -> float:
        """Return confidence score (0-1) for handling this query"""
        pass
    
    @abstractmethod
    async def process(self, query_context: QueryContext) -> AgentResponse:
        """Process the query and return response"""
        pass

class DocumentRetrievalAgent(BaseAgent):
    """Specializes in finding relevant documents and passages"""
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="semantic_search",
                description="Search documents using semantic similarity",
                func=self._semantic_search
            ),
            Tool(
                name="keyword_search", 
                description="Search documents using keyword matching",
                func=self._keyword_search
            ),
            Tool(
                name="document_filter",
                description="Filter documents by metadata (complexity, topic, etc.)",
                func=self._document_filter
            )
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        # Always can retrieve documents, but other agents handle response generation
        return 0.8
    
    async def process(self, query_context: QueryContext) -> AgentResponse:
        # Multi-step retrieval strategy
        
        # Step 1: Initial semantic search
        initial_docs = await self._semantic_search(
            query_context.query, 
            k=10,
            filter_by_persona=query_context.user_profile.persona
        )
        
        # Step 2: Analyze query complexity and adjust search
        query_complexity = self._assess_query_complexity(query_context.query)
        
        if query_complexity == "high":
            # Get more context with broader search
            additional_docs = await self._keyword_search(
                self._extract_key_concepts(query_context.query),
                k=5
            )
            initial_docs.extend(additional_docs)
        
        # Step 3: Filter by persona appropriateness
        filtered_docs = self._filter_by_persona(
            initial_docs, 
            query_context.user_profile.persona
        )
        
        # Step 4: Re-rank by relevance and quality
        ranked_docs = self._rerank_documents(
            filtered_docs,
            query_context.query,
            query_context.user_profile
        )
        
        return AgentResponse(
            content="Documents retrieved and ranked",
            confidence=0.9,
            sources=ranked_docs[:5],
            reasoning=f"Retrieved {len(ranked_docs)} relevant documents using multi-step strategy",
            next_suggestions=[]
        )
    
    async def _semantic_search(self, query: str, k: int = 5, filter_by_persona=None):
        # Implement semantic search with persona filtering
        search_kwargs = {"k": k}
        if filter_by_persona:
            search_kwargs["filter"] = {"persona_appropriate": filter_by_persona.value}
        
        docs = await self.vector_store.asimilarity_search(query, **search_kwargs)
        return docs
    
    def _assess_query_complexity(self, query: str) -> str:
        # Simple heuristic - can be replaced with ML model
        complex_indicators = [
            "how does", "explain the difference", "compare", "step by step",
            "technical analysis", "smart contract", "defi protocol"
        ]
        return "high" if any(indicator in query.lower() for indicator in complex_indicators) else "low"
    
    def _filter_by_persona(self, docs: List[Document], persona: UserPersona) -> List[Document]:
        # Filter documents based on complexity appropriate for persona
        complexity_map = {
            UserPersona.CRYPTO_NATIVE: ["intermediate", "advanced"],
            UserPersona.FINANCE_PROFESSIONAL: ["basic", "intermediate"], 
            UserPersona.CONSERVATIVE_NEWCOMER: ["basic"]
        }
        
        appropriate_complexity = complexity_map[persona]
        return [doc for doc in docs 
                if doc.metadata.get("complexity", "basic") in appropriate_complexity]
    
    def _rerank_documents(self, docs: List[Document], query: str, user_profile: UserProfile) -> List[Document]:
        # Implement relevance scoring and reranking
        # This is a simplified version - in practice, use more sophisticated reranking
        scored_docs = []
        for doc in docs:
            relevance_score = self._calculate_relevance(doc, query)
            persona_score = self._calculate_persona_fit(doc, user_profile)
            final_score = 0.7 * relevance_score + 0.3 * persona_score
            scored_docs.append((doc, final_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def _calculate_relevance(self, doc: Document, query: str) -> float:
        # Simplified relevance calculation
        # In practice, use embedding similarity or ML model
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words.intersection(doc_words))
        return overlap / len(query_words) if query_words else 0
    
    def _calculate_persona_fit(self, doc: Document, user_profile: UserProfile) -> float:
        # Calculate how well document matches user persona
        doc_complexity = doc.metadata.get("complexity", "basic")
        user_complexity = self._get_user_complexity_preference(user_profile.persona)
        
        if doc_complexity == user_complexity:
            return 1.0
        elif abs(self._complexity_to_score(doc_complexity) - 
                self._complexity_to_score(user_complexity)) <= 1:
            return 0.7
        else:
            return 0.3
    
    def _complexity_to_score(self, complexity: str) -> int:
        return {"basic": 1, "intermediate": 2, "advanced": 3}.get(complexity, 1)

class PersonaAdaptationAgent(BaseAgent):
    """Specializes in adapting responses to user personas"""
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="adapt_complexity",
                description="Adapt content complexity for user knowledge level",
                func=self._adapt_complexity
            ),
            Tool(
                name="add_context",
                description="Add appropriate background context for persona",
                func=self._add_context
            ),
            Tool(
                name="generate_examples",
                description="Generate persona-appropriate examples",
                func=self._generate_examples
            )
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        # Always relevant for response generation
        return 0.9
    
    async def process(self, query_context: QueryContext) -> AgentResponse:
        persona = query_context.user_profile.persona
        
        # Get persona-specific prompt template
        prompt_template = self._get_persona_prompt(persona)
        
        # Adapt content based on persona
        adapted_content = await self._adapt_for_persona(
            query_context,
            prompt_template
        )
        
        return AgentResponse(
            content=adapted_content,
            confidence=0.85,
            sources=[],
            reasoning=f"Adapted response for {persona.value} persona",
            next_suggestions=self._generate_follow_up_suggestions(persona, query_context.query)
        )
    
    def _get_persona_prompt(self, persona: UserPersona) -> ChatPromptTemplate:
        persona_instructions = {
            UserPersona.CRYPTO_NATIVE: """
                You are explaining to a 23-year-old crypto enthusiast with intermediate 
                crypto knowledge but basic finance knowledge. Use crypto terminology freely
                but explain financial concepts simply. Be concise and mention recent trends.
            """,
            UserPersona.FINANCE_PROFESSIONAL: """
                You are explaining to a 40-year-old finance professional with intermediate
                finance knowledge but basic crypto knowledge. Relate crypto concepts to
                traditional finance. Focus on investment and risk aspects.
            """,
            UserPersona.CONSERVATIVE_NEWCOMER: """
                You are explaining to a 55-year-old newcomer with basic knowledge in both
                crypto and finance. Use simple language, provide comprehensive background,
                and focus on practical benefits and risks. Be very clear and step-by-step.
            """
        }
        
        return ChatPromptTemplate.from_messages([
            ("system", persona_instructions[persona]),
            ("human", "{query}"),
            ("assistant", "Based on the following context: {context}")
        ])
    
    async def _adapt_for_persona(self, query_context: QueryContext, prompt_template: ChatPromptTemplate):
        # Implementation of persona adaptation logic
        pass
    
    def _generate_follow_up_suggestions(self, persona: UserPersona, query: str) -> List[str]:
        # Generate persona-appropriate follow-up questions
        suggestions_map = {
            UserPersona.CRYPTO_NATIVE: [
                "Want to see the technical implementation?",
                "Interested in yield farming opportunities?",
                "Check out the latest protocol updates?"
            ],
            UserPersona.FINANCE_PROFESSIONAL: [
                "How does this compare to traditional investments?",
                "What are the regulatory considerations?",
                "What's the risk-return profile?"
            ],
            UserPersona.CONSERVATIVE_NEWCOMER: [
                "Need help with the basic setup?",
                "Want to understand the risks better?",
                "Should we start with something simpler?"
            ]
        }
        return suggestions_map.get(persona, [])

class QualityAssessmentAgent(BaseAgent):
    """Evaluates response quality and appropriateness"""
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="check_accuracy",
                description="Verify factual accuracy of information",
                func=self._check_accuracy
            ),
            Tool(
                name="assess_complexity",
                description="Assess if complexity matches user level",
                func=self._assess_complexity
            ),
            Tool(
                name="validate_completeness",
                description="Check if response fully addresses query",
                func=self._validate_completeness
            )
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        # Always relevant for quality control
        return 0.7
    
    async def process(self, query_context: QueryContext, candidate_response: str) -> AgentResponse:
        # Assess various quality dimensions
        accuracy_score = await self._check_accuracy(candidate_response)
        complexity_score = await self._assess_complexity(candidate_response, query_context.user_profile)
        completeness_score = await self._validate_completeness(candidate_response, query_context.query)
        
        overall_quality = (accuracy_score + complexity_score + completeness_score) / 3
        
        if overall_quality < 0.7:
            improvement_suggestions = self._generate_improvement_suggestions(
                accuracy_score, complexity_score, completeness_score
            )
            return AgentResponse(
                content=f"Response needs improvement: {improvement_suggestions}",
                confidence=overall_quality,
                sources=[],
                reasoning="Quality assessment indicates response needs refinement",
                next_suggestions=improvement_suggestions
            )
        
        return AgentResponse(
            content="Response quality approved",
            confidence=overall_quality,
            sources=[],
            reasoning=f"Quality scores - Accuracy: {accuracy_score}, Complexity: {complexity_score}, Completeness: {completeness_score}",
            next_suggestions=[]
        )

# ==================== ORCHESTRATOR ====================

class AgenticRAGOrchestrator:
    """Main orchestrator that coordinates all agents"""
    
    def __init__(self, llm, vector_store, embeddings):
        self.llm = llm
        self.agents = {
            "retrieval": DocumentRetrievalAgent("retrieval", llm, vector_store, embeddings),
            "persona": PersonaAdaptationAgent("persona", llm, vector_store, embeddings),
            "quality": QualityAssessmentAgent("quality", llm, vector_store, embeddings)
        }
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="output",
            return_messages=True,
            k=10
        )
    
    async def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Main processing pipeline with multiple agents"""
        
        # Step 1: Document Retrieval
        print(f"🔍 Starting document retrieval for: {query_context.query}")
        retrieval_response = await self.agents["retrieval"].process(query_context)
        relevant_docs = retrieval_response.sources
        
        # Step 2: Initial Response Generation
        print(f"🎭 Adapting response for persona: {query_context.user_profile.persona.value}")
        context = self._prepare_context(relevant_docs)
        
        # Create enhanced query context with retrieved documents
        enhanced_context = QueryContext(
            user_profile=query_context.user_profile,
            query=query_context.query,
            conversation_context=query_context.conversation_context + [context],
            complexity_preference=query_context.complexity_preference,
            domain_focus=query_context.domain_focus
        )
        
        persona_response = await self.agents["persona"].process(enhanced_context)
        
        # Step 3: Quality Assessment and Refinement
        print("✅ Assessing response quality...")
        quality_response = await self.agents["quality"].process(
            query_context, 
            persona_response.content
        )
        
        # Step 4: Iterative Refinement (if needed)
        if quality_response.confidence < 0.7:
            print("🔄 Refining response based on quality feedback...")
            # Implement refinement logic here
            # This could involve getting additional documents, 
            # adjusting complexity, or generating alternative responses
            pass
        
        # Step 5: Final Response Assembly
        final_response = AgentResponse(
            content=persona_response.content,
            confidence=min(quality_response.confidence, persona_response.confidence),
            sources=relevant_docs,
            reasoning=f"Multi-agent processing: {retrieval_response.reasoning}, {persona_response.reasoning}",
            next_suggestions=persona_response.next_suggestions
        )
        
        # Update conversation memory
        self.memory.save_context(
            {"input": query_context.query},
            {"output": final_response.content}
        )
        
        return final_response
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(docs[:3]):  # Use top 3 docs
            context_parts.append(f"Source {i+1}: {doc.page_content[:500]}...")
        return "\n\n".join(context_parts)
    
    async def analyze_query_intent(self, query: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze query to understand user intent and routing needs"""
        
        # Simple intent classification - can be replaced with ML model
        intents = {
            "definition": ["what is", "define", "explain"],
            "comparison": ["compare", "difference", "vs", "versus"],
            "how_to": ["how to", "step by step", "tutorial"],
            "analysis": ["analyze", "pros and cons", "advantages"],
            "recommendation": ["should i", "recommend", "best"]
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            "primary_intent": detected_intents[0] if detected_intents else "general",
            "all_intents": detected_intents,
            "complexity_needed": self._infer_complexity_need(query, user_profile),
            "domain_focus": self._extract_domain_focus(query)
        }
    
    def _infer_complexity_need(self, query: str, user_profile: UserProfile) -> str:
        """Infer the level of complexity needed for the response"""
        high_complexity_indicators = [
            "technical", "advanced", "detailed", "deep dive", 
            "smart contract", "consensus mechanism", "cryptographic"
        ]
        
        if any(indicator in query.lower() for indicator in high_complexity_indicators):
            return "high"
        elif user_profile.persona == UserPersona.CRYPTO_NATIVE:
            return "medium"
        else:
            return "low"
    
    def _extract_domain_focus(self, query: str) -> List[str]:
        """Extract crypto domain focus from query"""
        domains = {
            "defi": ["defi", "decentralized finance", "yield", "liquidity"],
            "trading": ["trading", "exchange", "buy", "sell", "price"],
            "blockchain": ["blockchain", "consensus", "mining", "nodes"],
            "security": ["security", "wallet", "private key", "hack"],
            "regulation": ["regulation", "legal", "compliance", "tax"]
        }
        
        detected_domains = []
        for domain, keywords in domains.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains if detected_domains else ["general"]

# ==================== USAGE EXAMPLE ====================

async def main():
    """Example usage of the Agentic RAG system"""
    
    # Initialize components
    llm = ChatOpenAI(temperature=0.1, model="gpt-4")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(embedding_function=embeddings)
    
    # Create orchestrator
    orchestrator = AgenticRAGOrchestrator(llm, vector_store, embeddings)
    
    # Example user profiles
    crypto_native = UserProfile(
        persona=UserPersona.CRYPTO_NATIVE,
        age=23,
        crypto_knowledge="intermediate",
        finance_knowledge="basic",
        conversation_history=[],
        preferences={"detail_level": "high", "examples": "technical"}
    )
    
    # Example query
    query_context = QueryContext(
        user_profile=crypto_native,
        query="How do yield farming strategies work in DeFi protocols?",
        conversation_context=[],
        complexity_preference="medium",
        domain_focus=["defi"]
    )
    
    # Process query
    response = await orchestrator.process_query(query_context)
    
    print(f"Response: {response.content}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {len(response.sources)} documents")
    print(f"Suggestions: {response.next_suggestions}")

if __name__ == "__main__":
    asyncio.run(main())