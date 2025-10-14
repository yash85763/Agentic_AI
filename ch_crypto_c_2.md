Here are the specific changes needed:

## Change 1: Remove API Key Input (Sidebar section)

**Find this in sidebar (around line 200-210):**
```python
# API Key
api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    help="Leave empty to use environment variable"
)
```

**Remove it completely** and update the initialize button section:

**Find (around line 250):**
```python
if st.button("🚀 Initialize System", use_container_width=True):
    with st.spinner("Initializing knowledge graph search..."):
        try:
            st.session_state.searcher = GraphSimilaritySearch(
                graph_json_path=graph_path,
                openai_api_key=api_key if api_key else None
            )
```

**Replace with:**
```python
if st.button("🚀 Initialize System", use_container_width=True):
    with st.spinner("Initializing knowledge graph search..."):
        try:
            st.session_state.searcher = GraphSimilaritySearch(
                graph_json_path=graph_path,
                openai_api_key=None  # Always use environment variable
            )
```

## Change 2: Fix Query Analysis Display

**Find this section (around line 450-475):**
```python
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
```

**Replace with:**
```python
# Show decomposition if enabled (render immediately, not in chat)
if show_decomposition:
    decomp = results['decomposition']
    with st.expander("🧩 Query Analysis", expanded=True):
        st.markdown(f"**Reconstructed Query:** {decomp['reconstructed_query']}")
        st.markdown(f"**Intent:** {decomp['intent']}")
        st.markdown("**Key Concepts:** " + ", ".join(f"`{c}`" for c in decomp.get('key_concepts', [])))
        st.markdown("**Sub-queries:**")
        for i, sq in enumerate(decomp['sub_queries'], 1):
            st.markdown(f"  {i}. {sq}")
```

## Change 3: Fix Source Links in Answer


### New update in query processing block
```python
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
            
            # Get top pages for sources
            top_pages = results['aggregated_results'][:5]
            
            # Build source information
            source_links = []
            for i, page in enumerate(top_pages, 1):
                source_links.append({
                    'number': i,
                    'title': page['label'],
                    'url': page['url'],
                    'similarity': page['max_similarity'],
                    'matched_queries': len(page['matched_queries'])
                })
            
            # Generate answer if enabled
            if generate_answer:
                # Build context from pages
                context_parts = []
                source_list_for_prompt = []
                
                for i, page in enumerate(top_pages, 1):
                    full_node = st.session_state.searcher.get_node_details(page['url'])
                    if full_node:
                        text = full_node.get('text', '')[:1500]
                        context_parts.append(f"[SOURCE {i}] {page['label']}:\n{text}")
                        source_list_for_prompt.append(f"[{i}] {page['label']}")
                
                context = "\n\n".join(context_parts)
                sources_for_prompt = "\n".join(source_list_for_prompt)
                
                # Improved prompt with explicit source citation requirement
                prompt = f"""You are a retirement planning expert helping users understand their UC retirement benefits.

CHAT HISTORY (for context):
{history_context if history_context else "No prior conversation"}

CURRENT QUESTION: "{user_query}"

AVAILABLE SOURCES:
{sources_for_prompt}

DETAILED INFORMATION FROM SOURCES:

{context}

TASK:
Provide a clear, comprehensive answer to the user's question based ONLY on the information above.

CRITICAL REQUIREMENTS:
1. You MUST cite sources in your answer using the format: (Source 1), (Source 2), etc.
2. When you mention information, immediately cite which source it comes from
3. If multiple sources say the same thing, cite all of them: (Sources 1, 3)
4. Answer conversationally but always include citations
5. Use specific numbers and details from the sources
6. If comparing options, present both sides fairly
7. Include important caveats or conditions

CITATION EXAMPLES:
- "The 401k contribution limit for 2024 is $23,000 (Source 1)."
- "Both Roth and Traditional contributions are available (Sources 2, 3)."
- "According to the Benefits Overview (Source 1), you can..."

ANSWER (remember to cite sources throughout):"""

                # Generate answer
                response = st.session_state.searcher.client.chat.completions.create(
                    model=st.session_state.searcher.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining retirement planning clearly with proper source citations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                
                answer = response.choices[0].message.content
                
                # Add answer to response
                response_parts.append(f"<div style='margin: 1rem 0; line-height: 1.6;'>{answer}</div>")
            
            else:
                # Just show brief summary without full answer
                response_parts.append(f"""
                <div style='margin: 1rem 0;'>
                    Found <strong style='color: #58a6ff;'>{len(results['aggregated_results'])}</strong> relevant pages 
                    based on your query. See sources below for details.
                </div>
                """)
            
            # ALWAYS show source cards (regardless of checkbox)
            sources_html = """
            <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #30363d;'>
                <div style='color: #58a6ff; font-weight: 600; margin-bottom: 1rem; font-size: 1.1rem;'>
                    📚 Sources Used:
                </div>
            """
            
            for source in source_links:
                # Create clean URL for display
                display_url = source['url']
                if len(display_url) > 70:
                    display_url = display_url[:67] + "..."
                
                sources_html += f"""
                <div class="source-card" style="margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <div class="source-title">
                                <span style="color: #39d353; margin-right: 0.5rem;">[{source['number']}]</span>
                                {source['title']}
                            </div>
                            <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.3rem;">
                                <span class="source-similarity">
                                    Relevance: {source['similarity']:.1%}
                                </span>
                                <span style="margin: 0 0.5rem;">•</span>
                                <span>Matched {source['matched_queries']} sub-queries</span>
                            </div>
                        </div>
                        <div style="margin-left: 1rem;">
                            <a href="{source['url']}" target="_blank" 
                               style="display: inline-block; padding: 0.4rem 0.8rem; 
                                      background: rgba(88, 166, 255, 0.1); 
                                      border: 1px solid #58a6ff; border-radius: 6px;
                                      color: #58a6ff; text-decoration: none; font-size: 0.8rem;
                                      transition: all 0.2s;">
                                Visit Page →
                            </a>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.5rem; 
                                font-family: 'Fira Code', monospace; opacity: 0.7;">
                        🔗 {display_url}
                    </div>
                </div>
                """
            
            sources_html += "</div>"
            response_parts.append(sources_html)
            
            # Optional: Show all results in expandable section
            if len(results['aggregated_results']) > 5:
                more_results_html = f"""
                <details style="margin-top: 1rem; cursor: pointer;">
                    <summary style="color: #58a6ff; padding: 0.5rem; 
                                    background: rgba(88, 166, 255, 0.05); 
                                    border-radius: 6px; margin-bottom: 0.5rem;">
                        📋 Show all {len(results['aggregated_results'])} results
                    </summary>
                    <div style="padding: 0.5rem;">
                """
                
                for i, result in enumerate(results['aggregated_results'][5:], 6):
                    more_results_html += f"""
                    <div style="padding: 0.5rem; margin: 0.3rem 0; 
                                background: rgba(22, 27, 34, 0.4); border-radius: 4px;">
                        <div style="color: #c9d1d9; font-size: 0.9rem;">
                            <span style="color: #8b949e;">[{i}]</span> {result['label']}
                        </div>
                        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem;">
                            Relevance: {result['max_similarity']:.1%} • 
                            <a href="{result['url']}" target="_blank" style="color: #58a6ff;">
                                View page
                            </a>
                        </div>
                    </div>
                    """
                
                more_results_html += """
                    </div>
                </details>
                """
                response_parts.append(more_results_html)
            
            # Combine response
            full_response = "".join(response_parts)
            
            # Add assistant message to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': timestamp,
                'sources': source_links  # Store sources separately for future reference
            })
            
            # Rerun to display new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
```

## Key Improvements:

### 1. **Source Cards Always Visible**
- Source cards are now ALWAYS displayed regardless of checkbox
- Removed dependency on `show_sources` checkbox
- Made them a permanent part of the response

### 2. **Better Source Citation in LLM**
- LLM is explicitly required to cite sources as (Source 1), (Source 2), etc.
- Sources are numbered [1], [2], [3] for easy reference
- Prompt emphasizes citation throughout the answer

### 3. **Enhanced Source Card Design**
[1] 401k Contribution Limits
Relevance: 92.3% • Matched 3 sub-queries
[Visit Page →]
🔗 https://myucretirement.com/...


**Find this section (around line 520-560):**
```python
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
```

**Replace with:**
```python
# Generate answer
response = st.session_state.searcher.client.chat.completions.create(
    model=st.session_state.searcher.llm_model,
    messages=[
        {"role": "system", "content": "You are an expert at explaining retirement planning clearly and helpfully. Do NOT include source links in your answer - they will be added automatically."},
        {"role": "user", "content": prompt.replace("IMPORTANT: At the end of your answer, list the source pages used in this format:\n**Sources:**\n- [Page Title](URL)\n- [Page Title](URL)", "")}
    ],
    temperature=0.5
)

answer = response.choices[0].message.content

# Remove any source section the LLM might have added
if "**Sources:**" in answer or "Sources:" in answer:
    answer = answer.split("**Sources:**")[0].split("Sources:")[0].strip()

# Add answer to response
response_parts.append(f"\n\n{answer}")

# Add proper source links
if source_links:
    sources_section = "\n\n**📚 Sources:**\n"
    for source in source_links:
        sources_section += f"- [{source['title']}]({source['url']})\n"
    response_parts.append(sources_section)
```

**Also update the prompt to remove source instruction. Find (around line 500):**
```python
IMPORTANT: At the end of your answer, list the source pages used in this format:
**Sources:**
- [Page Title](URL)
- [Page Title](URL)

ANSWER:"""
```

**Replace with:**
```python
ANSWER:"""
```

## Summary of Changes:

1. ✅ **Removed API key input** - now only uses environment variable
2. ✅ **Fixed query analysis** - uses `st.expander()` instead of HTML in chat
3. ✅ **Fixed source links** - manually appends accurate links from `source_links` array instead of relying on LLM to generate them

These changes ensure:
- API key is always from environment
- Query decomposition displays properly in an expander
- Source links are accurate (taken from actual search results, not LLM-generated)
