HippoRAG Explained: RAG as Associative Memory

1. Why HippoRAG exists

Most Retrieval-Augmented Generation systems retrieve information like this:

user query -> query embedding -> nearest chunk embeddings -> top-k chunks -> LLM answer

This works well when the answer is contained in one or two chunks that look semantically similar to the query.

But many real questions are not like that. Many questions require associative retrieval:

Question mentions A.
A is connected to B.
B is connected to C.
The answer is in a passage about C.

A normal vector retriever may miss C because C may not be lexically or semantically close enough to the original query. HippoRAG tries to solve this by adding a graph-based memory layer on top of RAG.

The core idea is:

Instead of treating documents as isolated chunks, convert them into an associative graph of concepts, entities, relations, and passages. At query time, start from the query-related nodes and let relevance spread through the graph.

That spreading mechanism is Personalized PageRank, usually abbreviated as PPR.

⸻

2. One-sentence definition

HippoRAG is a RAG framework that builds a knowledge graph from documents and uses Personalized PageRank over that graph to retrieve connected evidence, especially for multi-hop and associative questions.

It combines three major ideas:

1. LLM extraction — use an LLM to extract facts/triples from text.
2. Knowledge graph indexing — store extracted facts as connected graph nodes and edges.
3. Personalized PageRank retrieval — start from query-related nodes and rank nearby connected nodes/passages.

⸻

3. The memory analogy

HippoRAG is inspired by how human long-term memory is believed to work.

A simplified analogy:

Human memory idea	HippoRAG component	Meaning
Neocortex	LLM and document text	Rich language understanding and factual content
Hippocampus	Knowledge graph + PPR	Fast associative index over memories
Memory cue	User query	The thing that activates a memory
Spreading association	Personalized PageRank	Relevance flows through related concepts
Recalled memory	Retrieved passages	Evidence given back to the LLM

In normal RAG, retrieval is mostly based on similarity.

In HippoRAG, retrieval is based on similarity plus graph association.

⸻

4. The problem with vanilla vector RAG

Imagine the corpus contains two separate passages:

Passage 1:
Erik Hort was born in Montebello.
Passage 2:
Montebello is located in Rockland County.

The user asks:

What county is Erik Hort's birthplace part of?

The answer requires a chain:

Erik Hort -> birthplace -> Montebello -> located in -> Rockland County

A vector retriever may easily retrieve Passage 1 because it directly mentions Erik Hort and birthplace-like information.

But Passage 2 may be missed because it does not mention Erik Hort. It only talks about Montebello and Rockland County.

So the retriever needs to do something more like:

Start at Erik Hort.
Find Montebello.
Then find facts connected to Montebello.
Retrieve the passage saying Montebello is in Rockland County.

This is what HippoRAG’s graph-based retrieval helps with.

⸻

5. How HippoRAG builds its graph

HippoRAG has an offline indexing phase.

5.1 Input documents

You start with documents:

PDFs
web pages
reports
papers
contracts
company documents
regulatory filings
notes

These documents are chunked into passages.

5.2 Extract facts from passages

For each passage, an LLM extracts facts as triples:

(subject, relation, object)

Example:

("Erik Hort", "was born in", "Montebello")
("Montebello", "is located in", "Rockland County")
("Rockland County", "is located in", "New York")

This style is often called Open Information Extraction, or OpenIE.

It is called “open” because relation types are not predefined. The LLM can extract natural-language relations such as:

founded by
headquartered in
acquired
serves customers in
provides software for
was born in
is located in

5.3 Convert triples into a graph

Each entity or phrase becomes a node.

Each relation becomes an edge.

For example:

Erik Hort --was born in--> Montebello
Montebello --is located in--> Rockland County
Rockland County --is located in--> New York

A graph version:

graph LR
    A[Erik Hort] -->|was born in| B[Montebello]
    B -->|is located in| C[Rockland County]
    C -->|is located in| D[New York]

5.4 Connect graph nodes back to passages

The graph is useful only if it can point back to original evidence.

So HippoRAG needs mappings like:

Node: Erik Hort
  appears in: Passage 1
Node: Montebello
  appears in: Passage 1, Passage 2
Node: Rockland County
  appears in: Passage 2

This matters because the LLM should not answer from graph labels alone. It should answer from retrieved evidence passages.

⸻

6. What retrieval looks like in HippoRAG

At query time, HippoRAG roughly does this:

1. User asks a question.
2. System identifies query-relevant nodes or triples.
3. Those nodes become seed nodes.
4. Personalized PageRank starts from those seed nodes.
5. Relevance spreads through the graph.
6. High-scoring nodes identify high-value passages.
7. Those passages are passed to the LLM.
8. LLM generates the final answer using evidence.

The key part is step 4 and 5.

That is the PPR part.

⸻

7. Personalized PageRank from first principles

7.1 PageRank without the math

Classic PageRank was originally used to rank web pages.

The idea was:

A page is important if many important pages link to it.

So if many strong pages point to Page X, Page X becomes important.

But HippoRAG is not trying to rank globally popular web pages.

HippoRAG asks a different question:

Given this specific user query, which graph nodes are important relative to the query?

That is why it uses Personalized PageRank, not plain global PageRank.

⸻

7.2 Normal PageRank vs Personalized PageRank

Normal PageRank

Normal PageRank asks:

Which nodes are globally important in the whole graph?

Example:

Wikipedia
Google
New York
United States

These nodes may become highly ranked because many things point to them.

But for RAG, global importance is not enough.

If the user asks about a small company, you do not want the graph walk to always drift toward huge generic nodes like “United States,” “software,” or “business.”

Personalized PageRank

Personalized PageRank asks:

Which nodes are important relative to my starting nodes?

In HippoRAG, the starting nodes come from the query.

Example query:

What county is Erik Hort's birthplace part of?

Seed node:

Erik Hort

PPR then ranks nodes near Erik Hort:

Erik Hort
Montebello
Rockland County
New York

So PPR is a query-conditioned graph ranking algorithm.

⸻

7.3 PPR as spreading attention

The easiest mental model:

PPR puts relevance on the query nodes and lets that relevance flow through graph edges.

Suppose we start with 100 relevance points on Erik Hort.

Erik Hort:        100
Montebello:         0
Rockland County:    0
New York:           0

The graph has this path:

Erik Hort -> Montebello -> Rockland County -> New York

After one step, some relevance flows to Montebello:

Erik Hort:         15
Montebello:        85
Rockland County:    0
New York:           0

After another step, some relevance flows from Montebello to Rockland County, while some jumps back to Erik Hort:

Erik Hort:         15
Montebello:        13
Rockland County:   72
New York:           0

After more steps, relevance keeps spreading but also keeps restarting back to Erik Hort.

Eventually the scores stabilize.

The high-scoring nodes are interpreted as relevant to the query.

⸻

7.4 The restart trick

PPR has a restart mechanism.

At each step, the random walker does one of two things:

Option 1: Continue walking to a neighboring node.
Option 2: Jump back to the query seed nodes.

A common setting is approximately:

85% chance: follow a graph edge
15% chance: restart at the query seeds

This restart is extremely important.

Without restart, the walk may drift too far:

Erik Hort
-> Montebello
-> Rockland County
-> New York
-> United States
-> North America
-> Earth

That is not useful retrieval.

The restart says:

Explore connected facts, but do not forget the original question.

So PPR balances two forces:

exploration: move through graph edges
anchoring: return to query nodes

⸻

7.5 The PPR equation

The standard Personalized PageRank update can be written like this:

r = alpha * s + (1 - alpha) * P^T * r

Where:

Symbol	Meaning
r	Final relevance score vector over graph nodes
s	Seed vector from the query
P	Transition matrix of the graph
P^T	Transposed transition matrix, used to collect score from incoming neighbors
alpha	Restart probability
1 - alpha	Probability of walking through graph edges

In plain English:

final node score =
    restart score from the query
    +
    score received from neighboring nodes

A node gets a high score when:

1. It is directly related to the query, or
2. It is connected to nodes that are related to the query, or
3. Many relevant paths from the query lead to it.

⸻

7.6 What is the seed vector?

The seed vector says where the walk starts.

Suppose the graph has four nodes:

0: Erik Hort
1: Montebello
2: Rockland County
3: New York

For the query:

What county is Erik Hort's birthplace part of?

The system may choose Erik Hort as the seed.

Then the seed vector is:

s = [1.0, 0.0, 0.0, 0.0]

Meaning:

Start all query relevance at Erik Hort.

If the query maps to multiple nodes, the seed vector may distribute probability:

s = [0.7, 0.3, 0.0, 0.0]

Meaning:

70% starts at Erik Hort.
30% starts at another query-relevant node.

⸻

7.7 What is the transition matrix?

The transition matrix describes where relevance can flow.

Suppose the graph is:

Erik Hort -> Montebello
Montebello -> Rockland County
Rockland County -> New York

Then from Erik Hort, the walker can go to Montebello.

From Montebello, the walker can go to Rockland County.

From Rockland County, the walker can go to New York.

If a node has multiple neighbors, its probability is split among them.

Example:

Montebello -> Rockland County
Montebello -> New York

Then Montebello may distribute relevance like:

50% to Rockland County
50% to New York

In real systems, edges may be weighted. A stronger edge can receive more probability.

⸻

7.8 Tiny numeric example

Consider this graph:

graph LR
    A[Erik Hort] --> B[Montebello]
    B --> C[Rockland County]
    C --> D[New York]

Seed:

Erik Hort = 1.0

Restart probability:

alpha = 0.15

Walk probability:

1 - alpha = 0.85

At a very simplified level:

Iteration	Erik Hort	Montebello	Rockland County	New York
0	1.00	0.00	0.00	0.00
1	0.15	0.85	0.00	0.00
2	0.15	0.13	0.72	0.00
3	0.15	0.13	0.11	0.61

This table is simplified because real PPR usually handles dangling nodes, bidirectional edges, self-loops, normalization, and convergence. But the intuition is correct:

query relevance starts at seed nodes
then moves outward through graph connections
but keeps restarting at the query

⸻

7.9 Why PPR helps multi-hop retrieval

A multi-hop question usually requires this pattern:

Query mentions A.
Need to find B related to A.
Then need C related to B.
Answer comes from C.

Vanilla vector search often struggles because C may not look similar to the query.

PPR helps because it does not require C to be directly similar to the query. C only needs to be reachable through meaningful graph connections.

Example:

Query: What county is Erik Hort's birthplace part of?
Seed:
Erik Hort
Graph spread:
Erik Hort -> Montebello -> Rockland County
Retrieved evidence:
Passage about Erik Hort and Montebello
Passage about Montebello and Rockland County

The LLM now receives both pieces of evidence.

⸻

8. PPR in HippoRAG specifically

In HippoRAG, PPR is used as a retrieval ranking mechanism.

It does not generate the final answer.

It does not reason in natural language.

It ranks graph nodes and associated passages.

A simplified HippoRAG retrieval flow:

User query
   ↓
Query-to-node / query-to-triple linking
   ↓
Seed vector over graph nodes
   ↓
Personalized PageRank
   ↓
Ranked graph nodes
   ↓
Associated passages
   ↓
Top-k evidence passages
   ↓
LLM answer

So PPR answers this internal question:

Which connected memories should be recalled for this query?

⸻

8.1 Why not just ask the LLM to do multi-hop reasoning?

You could use an agentic loop:

retrieve -> reason -> retrieve again -> reason -> retrieve again

That can work.

But it has problems:

1. It requires multiple LLM calls.
2. It is slower.
3. It is more expensive.
4. It can drift if the LLM chooses poor follow-up queries.
5. It may fail if the first retrieval step misses the key bridge entity.

HippoRAG tries to make multi-hop retrieval more efficient by doing graph activation in one retrieval stage.

Instead of asking the LLM to repeatedly decide where to search next, the graph algorithm spreads relevance automatically.

⸻

8.2 Why PPR instead of shortest path?

You might ask:

Why not simply find paths in the graph?

Shortest path is useful when you know the source and target.

But in retrieval, you usually do not know the target.

The system only knows the query.

PPR is useful because it does not need the target. It simply asks:

Starting from the query nodes, which nodes become most relevant after graph propagation?

This is better for discovery.

⸻

8.3 Why PPR instead of simple one-hop expansion?

One-hop expansion retrieves immediate neighbors only.

For example:

Erik Hort -> Montebello

But many questions require two or three hops:

Erik Hort -> Montebello -> Rockland County

PPR allows controlled multi-hop expansion.

It can go beyond one hop while still staying anchored to the query.

⸻

9. HippoRAG 2: what changed

HippoRAG 2 keeps the central idea of graph-based memory and PPR, but improves the architecture.

The original HippoRAG was heavily phrase/entity-centric. That means retrieval depended strongly on extracted graph nodes and their connections.

HippoRAG 2 adds stronger passage integration and more effective online LLM use.

The high-level improvements are:

1. Passage nodes are integrated more deeply into the graph.
2. Query-to-triple matching improves retrieval beyond simple entity matching.
3. LLM-based recognition/filtering helps remove irrelevant triples before running graph retrieval.
4. The system is designed to perform better across factual, sense-making, and associative memory tasks.

The important takeaway:

HippoRAG 1 showed that PPR over a KG can support associative retrieval. HippoRAG 2 improves how passages and query-time filtering are integrated so the system is not only good at association but also better at direct factual retrieval and broader sense-making.

⸻

10. Comparison with other retrieval approaches

10.1 Vanilla RAG

query -> vector search -> chunks -> answer

Good for:

direct semantic lookup
simple factual questions
questions where answer is in one chunk

Weak for:

multi-hop questions
cross-document reasoning
bridge-entity discovery

10.2 Agentic RAG

query -> retrieve -> LLM decides next search -> retrieve again -> answer

Good for:

complex tasks
adaptive search
planning-heavy workflows

Weak for:

latency
cost
retrieval drift
harder reproducibility

10.3 GraphRAG

GraphRAG often uses graph construction, community detection, and summaries to answer broader questions.

Good for:

corpus-level summarization
community-level questions
global themes across a dataset

Weak for:

some precise multi-hop retrieval cases
cases where summaries lose specific evidence

10.4 HippoRAG

documents -> triples -> graph -> PPR retrieval -> evidence passages -> answer

Good for:

associative retrieval
multi-hop fact connection
cross-document evidence discovery
memory-like recall

Weak for:

noisy triple extraction
bad entity linking
large graph maintenance
ambiguous relation semantics

⸻

11. Why HippoRAG matters for company profiling agents

For your company profiler use case, HippoRAG is relevant because company facts are often scattered across many PDFs and websites.

Example:

Document A:
Company X acquired Company Y.
Document B:
Company Y offers revenue cycle management software.
Document C:
Revenue cycle management software is sold to hospitals and clinics.
Profile field:
Target customer segment.

A direct vector search for “Company X target customer segment” may not retrieve all three documents.

A graph can connect:

Company X -> acquired -> Company Y
Company Y -> offers -> revenue cycle management software
revenue cycle management software -> used by -> hospitals and clinics

Then PPR can retrieve the connected evidence chain.

This is especially helpful for fields that require inference, such as:

business model
customer segment
product category
market served
acquisition-derived capabilities
partner ecosystem
regulatory exposure
supply chain role

⸻

12. Implementation sketch

A simplified implementation could look like this.

12.1 Offline indexing

def build_hipporag_index(documents):
    passages = chunk_documents(documents)
    graph = Graph()
    passage_store = {}
    for passage in passages:
        passage_id = save_passage(passage)
        passage_store[passage_id] = passage
        triples = llm_extract_openie_triples(passage.text)
        for subject, relation, object_ in triples:
            s_node = normalize_entity(subject)
            o_node = normalize_entity(object_)
            graph.add_node(s_node)
            graph.add_node(o_node)
            graph.add_edge(s_node, o_node, label=relation)
            graph.link_node_to_passage(s_node, passage_id)
            graph.link_node_to_passage(o_node, passage_id)
    graph = add_synonym_or_similarity_edges(graph)
    return graph, passage_store

12.2 Online retrieval

def retrieve_with_hipporag(query, graph, passage_store, top_k=10):
    query_nodes = link_query_to_graph_nodes(query, graph)
    seed_vector = build_seed_vector(query_nodes, graph)
    ppr_scores = personalized_pagerank(
        graph=graph,
        seed_vector=seed_vector,
        restart_probability=0.15,
    )
    ranked_nodes = sort_nodes_by_score(ppr_scores)
    candidate_passages = collect_passages_from_nodes(
        ranked_nodes,
        graph,
        passage_store,
    )
    reranked_passages = rerank_passages(query, candidate_passages)
    return reranked_passages[:top_k]

12.3 PPR pseudocode

def personalized_pagerank(graph, seed_vector, restart_probability=0.15, iterations=30):
    # r is the current relevance distribution over nodes
    r = seed_vector.copy()
    # P is the transition matrix of the graph
    P = build_transition_matrix(graph)
    alpha = restart_probability
    for _ in range(iterations):
        r = alpha * seed_vector + (1 - alpha) * P.T @ r
    return r

In plain English:

Start with query relevance.
Repeatedly spread some relevance through graph edges.
Repeatedly inject some relevance back into the query nodes.
Stop when scores stabilize or after a fixed number of iterations.

⸻

13. Practical engineering considerations

13.1 Triple extraction quality

Bad triples create bad graph paths.

Example of a bad extraction:

("Company X", "is", "market")

This is vague and not useful.

Better extraction:

("Company X", "provides fraud detection software for", "banks")

For production systems, you need:

triple validation
relation normalization
entity canonicalization
source citations
confidence scores

13.2 Entity resolution

These may refer to the same company:

IBM
International Business Machines
IBM Corp.
International Business Machines Corporation

If you do not merge them, the graph fragments.

Fragmented graph means PPR cannot flow properly.

13.3 Generic node problem

Some nodes are too generic:

company
software
product
United States
business
customer

These can become noisy hubs.

Solutions:

downweight generic nodes
remove stop-entities
use relation-aware weighting
use type filters
limit max degree
apply passage reranking after PPR

13.4 Edge weighting

Not all edges should be equal.

Possible edge weights:

confidence of triple extraction
frequency of relation
semantic similarity between query and triple
recency of source document
authority of source document
field-specific relevance

For company profiling, a recent 10-K may deserve more weight than an old marketing blog.

13.5 Retrieval should still return passages

Do not let the graph become the final answer source by itself.

The graph is an index.

The evidence should still come from original passages.

Good answer generation should use:

retrieved passage text
source document
page number / URL
extracted triples
confidence score

⸻

14. A concrete mental model

Think of normal RAG like this:

Search a library by matching your question to book paragraphs.

Think of HippoRAG like this:

Search a library by starting from concepts in your question, following the web of related concepts, and then pulling the books/pages attached to the brightest concepts.

PPR is the “brightness spreading” algorithm.

It says:

The query lights up some nodes.
The light flows through edges.
The restart keeps the light near the query.
The brightest connected nodes point to useful evidence.

⸻

15. The cleanest summary

HippoRAG changes retrieval from this:

Find chunks that look like the query.

To this:

Find concepts mentioned by the query.
Spread relevance through a knowledge graph.
Retrieve passages attached to the most relevant connected concepts.

The most important part is Personalized PageRank:

PPR = query-anchored relevance propagation over a graph.

It is useful because many answers are not directly similar to the query. They are connected to the query through intermediate facts.

That is why HippoRAG is powerful for multi-document, multi-hop, profile-building, research, legal, regulatory, and company-intelligence workflows.

⸻

16. Key terms

Term	Meaning
RAG	Retrieval-Augmented Generation; retrieve evidence, then generate answer
Chunk	A passage of text from a document
Triple	A fact represented as subject-relation-object
OpenIE	Open Information Extraction; extracting triples without a fixed schema
Knowledge graph	Nodes and edges representing entities, concepts, and relations
Seed node	A graph node linked to the user query
PageRank	Algorithm that ranks nodes based on graph links
Personalized PageRank	PageRank biased toward specific seed nodes
Restart probability	Probability of jumping back to query seed nodes
Transition matrix	Matrix describing how probability flows across graph edges
Associative retrieval	Retrieval through related concepts rather than only direct similarity
Multi-hop retrieval	Retrieval requiring multiple connected facts

⸻

17. References to study next

1. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models — original HippoRAG paper.
2. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models — HippoRAG 2 paper.
3. Efficient Algorithms for Personalized PageRank Computation: A Survey — useful for understanding PPR deeply from an algorithmic perspective.
4. Google PageRank / random surfer model — helpful background for the original PageRank intuition.