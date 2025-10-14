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
