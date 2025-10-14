import streamlit as st
import json
from similarity_search_updated import GraphSimilaritySearch
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="UC Retirement Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for code environment aesthetic
st.markdown("""
<style>
    /* Import Consolas-like font */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Fira Code', 'Consolas', 'Monaco', monospace !important;
    }
    
    /* Main background - dark code editor theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0f1419 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(88, 166, 255, 0.5);
        letter-spacing: 2px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #8b949e;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #1f6feb 0%, #1158c7 100%);
        color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.8rem 0;
        margin-left: 15%;
        border-left: 4px solid #58a6ff;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #21262d 0%, #161b22 100%);
        color: #c9d1d9;
        padding: 1rem 1.2rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.8rem 0;
        margin-right: 15%;
        border-left: 4px solid #39d353;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        color: #8b949e;
        margin-top: 0.5rem;
        opacity: 0.7;
    }
    
    .message-role {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    /* Links in messages */
    .assistant-message a {
        color: #58a6ff !important;
        text-decoration: none;
        border-bottom: 1px dashed #58a6ff;
        transition: all 0.2s;
    }
    
    .assistant-message a:hover {
        color: #79c0ff !important;
        border-bottom: 1px solid #79c0ff;
        text-shadow: 0 0 8px rgba(88, 166, 255, 0.5);
    }
    
    /* Query decomposition box */
    .decomposition-box {
        background: rgba(56, 139, 253, 0.1);
        border: 1px solid #1f6feb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
    
    .decomposition-title {
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .sub-query {
        color: #8b949e;
        padding: 0.3rem 0;
        padding-left: 1rem;
        border-left: 2px solid #30363d;
        margin: 0.3rem 0;
    }
    
    /* Source cards */
    .source-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    
    .source-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 0 12px rgba(88, 166, 255, 0.2);
        transform: translateX(4px);
    }
    
    .source-title {
        color: #58a6ff;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .source-similarity {
        color: #39d353;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Input box styling */
    .stTextInput input {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #238636 0%, #1a7f37 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%) !important;
        box-shadow: 0 6px 20px rgba(35, 134, 54, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #58a6ff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(22, 27, 34, 0.6) !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(13, 17, 23, 0.8) !important;
        border: 1px solid #30363d !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-online {
        background: #39d353;
        box-shadow: 0 0 8px rgba(57, 211, 83, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Clear chat button */
    .clear-chat-btn {
        background: rgba(248, 81, 73, 0.1) !important;
        color: #f85149 !important;
        border: 1px solid #f85149 !important;
    }
    
    .clear-chat-btn:hover {
        background: rgba(248, 81, 73, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'searcher' not in st.session_state:
    st.session_state.searcher = None

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Status indicator
    status = "🟢 Online" if st.session_state.initialized else "🔴 Offline"
    st.markdown(f"**Status:** {status}")
    
    st.markdown("---")
    
    # Graph file path
    graph_path = st.text_input(
        "Graph JSON Path",
        value="kg_output/graph.json",
        help="Path to your knowledge graph JSON file"
    )
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Leave empty to use environment variable"
    )
    
    # Search parameters
    st.markdown("### 🔍 Search Settings")
    
    top_k = st.slider(
        "Results per sub-query",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of results to fetch for each decomposed sub-query"
    )
    
    min_similarity = st.slider(
        "Minimum Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity threshold for results"
    )
    
    generate_answer = st.checkbox(
        "Generate AI Answer",
        value=True,
        help="Use LLM to generate comprehensive answers"
    )
    
    show_decomposition = st.checkbox(
        "Show Query Decomposition",
        value=True,
        help="Display how the query is broken down"
    )
    
    show_sources = st.checkbox(
        "Show Source Cards",
        value=True,
        help="Display source pages used for answer"
    )
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🚀 Initialize System", use_container_width=True):
        with st.spinner("Initializing knowledge graph search..."):
            try:
                st.session_state.searcher = GraphSimilaritySearch(
                    graph_json_path=graph_path,
                    openai_api_key=api_key if api_key else None
                )
                st.session_state.initialized = True
                st.success("✅ System initialized!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        time.sleep(0.5)
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if st.session_state.initialized and st.session_state.searcher:
        st.metric("Total Pages", len(st.session_state.searcher.nodes_with_embeddings))
        st.metric("Messages", len(st.session_state.chat_history))

# Main content
st.markdown('<div class="main-title">🤖 UC RETIREMENT ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">// AI-Powered Knowledge Graph Search for myucretirement.com</div>', unsafe_allow_html=True)

# Check if initialized
if not st.session_state.initialized:
    st.info("👈 Please initialize the system using the sidebar configuration.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Configure settings** in the sidebar
    2. **Enter your OpenAI API key** (or set as environment variable)
    3. **Click 'Initialize System'** to load the knowledge graph
    4. **Start asking questions** about UC retirement planning!
    
    ### Example Questions:
    - "What are the 401k contribution limits for 2024?"
    - "Should I choose Roth or Traditional contributions?"
    - "How do catch-up contributions work?"
    - "What is the difference between 403b and 457b?"
    """)
    
else:
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="message-role">👤 You</div>
                {message['content']}
                <div class="message-timestamp">{message['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-role">🤖 Assistant</div>
                {message['content']}
                <div class="message-timestamp">{message['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question...",
            key="user_input",
            placeholder="Type your retirement planning question here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    # Process query
    if send_button and user_query:
        # Add user message to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': timestamp
        })
        
        # Process query with chat history context
        with st.spinner("🔍 Analyzing query and searching knowledge graph..."):
            try:
                # Get chat history for context
                recent_history = st.session_state.chat_history[-6:]  # Last 3 exchanges
                history_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
                    for msg in recent_history[:-1]  # Exclude current message
                ])
                
                # Search with decomposition
                results = st.session_state.searcher.search_with_decomposition(
                    user_query,
                    top_k_per_query=top_k,
                    min_similarity=min_similarity
                )
                
                # Build response
                response_parts = []
                
                # Show decomposition if enabled
                if show_decomposition:
                    decomp = results['decomposition']
                    decomp_html = f"""
                    <div class="decomposition-box">
                        <div class="decomposition-title">🧩 Query Analysis</div>
                        <div style="margin: 0.5rem 0;">
                            <strong style="color: #8b949e;">Reconstructed:</strong> 
                            <span style="color: #c9d1d9;">{decomp['reconstructed_query']}</span>
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <strong style="color: #8b949e;">Intent:</strong> 
                            <span style="color: #c9d1d9;">{decomp['intent']}</span>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <strong style="color: #8b949e;">Sub-queries:</strong>
                        </div>
                        {"".join([f'<div class="sub-query">→ {sq}</div>' for sq in decomp['sub_queries']])}
                    </div>
                    """
                    response_parts.append(decomp_html)
                
                # Generate answer
                if generate_answer:
                    # Get top pages for context
                    top_pages = results['aggregated_results'][:5]
                    
                    # Build context
                    context_parts = []
                    source_links = []
                    
                    for i, page in enumerate(top_pages, 1):
                        full_node = st.session_state.searcher.get_node_details(page['url'])
                        if full_node:
                            text = full_node.get('text', '')[:1500]
                            context_parts.append(f"SOURCE {i} - {page['label']}:\n{text}")
                            source_links.append({
                                'title': page['label'],
                                'url': page['url'],
                                'similarity': page['max_similarity']
                            })
                    
                    context = "\n\n".join(context_parts)
                    
                    # Include chat history in prompt
                    prompt = f"""You are a retirement planning expert helping users understand their UC retirement benefits.

CHAT HISTORY (for context):
{history_context if history_context else "No prior conversation"}

CURRENT QUESTION: "{user_query}"

RELEVANT INFORMATION FROM UC RETIREMENT WEBSITE:

{context}

TASK:
Provide a clear, comprehensive answer to the user's question based on the information above. Consider the chat history for context.

GUIDELINES:
1. Answer directly and conversationally
2. Use specific numbers and details from the sources
3. Reference the relevant pages naturally (e.g., "As explained in the 401k Overview...")
4. If comparing options, present both sides fairly
5. Include important caveats or conditions
6. Suggest consulting HR or a financial advisor for personalized advice if appropriate
7. Keep the tone friendly and helpful

IMPORTANT: At the end of your answer, list the source pages used in this format:
**Sources:**
- [Page Title](URL)
- [Page Title](URL)

ANSWER:"""

                    # Generate answer
                    response = st.session_state.searcher.client.chat.completions.create(
                        model=st.session_state.searcher.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at explaining retirement planning clearly and helpfully."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Add answer to response
                    response_parts.append(f"\n\n{answer}")
                    
                    # Show source cards if enabled
                    if show_sources:
                        sources_html = "<div style='margin-top: 1rem;'>"
                        for source in source_links:
                            sources_html += f"""
                            <div class="source-card">
                                <div class="source-title">📄 {source['title']}</div>
                                <div class="source-similarity">Similarity: {source['similarity']:.3f}</div>
                                <a href="{source['url']}" target="_blank" style="font-size: 0.8rem; color: #58a6ff;">
                                    {source['url'][:60]}...
                                </a>
                            </div>
                            """
                        sources_html += "</div>"
                        response_parts.append(sources_html)
                
                else:
                    # Just show top results without generating answer
                    results_html = "<div style='margin-top: 1rem;'><strong>Top Results:</strong></div>"
                    for i, result in enumerate(results['aggregated_results'][:5], 1):
                        results_html += f"""
                        <div class="source-card">
                            <div class="source-title">{i}. {result['label']}</div>
                            <div class="source-similarity">Similarity: {result['max_similarity']:.3f}</div>
                            <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem;">
                                Matched {len(result['matched_queries'])} sub-queries
                            </div>
                            <a href="{result['url']}" target="_blank" style="font-size: 0.8rem; color: #58a6ff;">
                                Visit page →
                            </a>
                        </div>
                        """
                    response_parts.append(results_html)
                
                # Combine response
                full_response = "".join(response_parts)
                
                # Add assistant message to history
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': timestamp
                })
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.85rem; padding: 1rem;">
    <strong style="color: #58a6ff;">UC Retirement Assistant</strong> | 
    Powered by Knowledge Graph RAG | 
    myucretirement.com
</div>
""", unsafe_allow_html=True)
