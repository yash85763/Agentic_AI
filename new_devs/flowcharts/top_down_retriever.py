import re
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Iterable
from utils.embed import embed


class TopDownRetriever:
    """
    Two-stage scoring: dense similarity + hierarchy-aware reranking.
    - Prefers generic/global families when the query is generic.
    - Boosts series-specific families only when the query explicitly mentions them.
    - Light depth penalty (shallower nodes surface first for generic queries).
    - Optional lexical keyword boost.
    Also returns a compact subgraph for each top-k result (ancestors/siblings/children).
    """

    # Roots considered "generic/global" (tune as needed)
    GENERIC_FAMILIES = (
        "chart",               # chart-level options
        "title",               # title.*
        "subtitle",            # subtitle.*
        "tooltip",             # tooltip.*
        "series",              # series.* (methods/options)
        "plotOptions.series",  # generic series family
        "global",              # Highcharts.setOptions / global options (if present).
    )

    # Common Highcharts series namespaces (expand if needed)
    SERIES_TYPES = {
        "line", "spline", "area", "areaspline", "column", "bar", "scatter", "bubble",
        "heatmap", "funnel", "pie", "treemap", "sunburst", "packedbubble", "boxplot",
        "waterfall", "candlestick", "ohlc", "polygon", "gauge", "solidgauge", "vector",
        "streamgraph", "wordcloud", "bellcurve", "pareto", "xrange", "timeline",
    }

    # Optional lightweight lexical signals
    KEY_TERMS = {
        "animation", "animate", "update", "updates", "updating",
        "title", "subtitle", "caption", "credits", "tooltip",
        "font", "font size", "label", "data label", "data labels",
    }

    def __init__(
        self,
        graph: nx.DiGraph,
        embeddings: Dict[str, np.ndarray],
        *,
        weights: Optional[Dict[str, float]] = None,
        depth_free: int = 2,              # depth level exempt from penalty
        assume_normalized: bool = False,  # set True if embeddings are already L2-normalized
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.depth_free = max(0, depth_free)
        self.assume_normalized = assume_normalized

        # Default weights for reranking heuristics
        self.w = {
            "generic_boost": 0.10,     # boost generic families when query is generic
            "specific_boost": 0.12,    # boost specific series family when query mentions it
            "specific_penalty": 0.06,  # penalty for specific families when query is generic
            "depth_penalty": 0.01,     # per extra depth level beyond depth_free
            "keyword_boost": 0.03,     # small lexical signal
        }
        if weights:
            self.w.update(weights)

        # Pre-normalize embeddings if needed
        if not self.assume_normalized:
            for nid, vec in list(self.embeddings.items()):
                self.embeddings[nid] = self._normalize(vec)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec if n == 0 else vec / n

    @staticmethod
    def _cosine(q: np.ndarray, v: np.ndarray) -> float:
        # both vectors are normalized -> dot product equals cosine
        return float(np.dot(q, v))

    @staticmethod
    def _depth(path: str) -> int:
        return path.count(".")

    @staticmethod
    def _family(path: str) -> str:
        """
        Family prefix:
          - 'plotOptions.series.animation' -> 'plotOptions.series'
          - 'plotOptions.heatmap.animation' -> 'plotOptions.heatmap'
          - 'tooltip.animation.duration' -> 'tooltip'
          - 'chart.animation' -> 'chart'
        """
        parts = path.split(".")
        if len(parts) >= 2 and parts[0] == "plotOptions":
            return ".".join(parts[:2])
        return parts[0]

    @staticmethod
    def _contains_series_type(path: str) -> Tuple[bool, str]:
        tokens = set(path.split("."))
        for t in TopDownRetriever.SERIES_TYPES:
            if t in tokens:
                return True, t
        return False, ""

    @staticmethod
    def _mentioned_series_types(query: str) -> List[str]:
        q = query.lower()
        return [t for t in TopDownRetriever.SERIES_TYPES if t in q]

    @staticmethod
    def _has_keywords(query: str, *texts: Iterable[str]) -> bool:
        q = query.lower()
        lower_texts = " ".join((t or "").lower() for t in texts)
        return any(k in q and k in lower_texts for k in TopDownRetriever.KEY_TERMS)

    # ---------- subgraph enrichment ----------

    def build_subgraph(self, property_path: str, depth: int = 2) -> Optional[Dict[str, Any]]:
        """
        Returns a small neighborhood around the node:
        - ancestors: walk up (child -> parent edges) up to `depth` hops
        - descendants: walk down (predecessors) up to `depth` hops
        - siblings: other children of immediate parent(s)
        """
        if property_path not in self.graph:
            return None

        center = self.graph.nodes[property_path]

        # Ancestors (parents upward)
        ancestors: List[str] = []
        frontier = [property_path]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                parents = list(self.graph.successors(node))
                ancestors.extend(parents)
                next_frontier.extend(parents)
            if not next_frontier:
                break
            frontier = next_frontier

        # Descendants (children downward)
        descendants: List[str] = []
        frontier = [property_path]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                kids = list(self.graph.predecessors(node))
                descendants.extend(kids)
                next_frontier.extend(kids)
            if not next_frontier:
                break
            frontier = next_frontier

        # Siblings (share immediate parent)
        siblings: List[str] = []
        immediate_parents = list(self.graph.successors(property_path))
        for p in immediate_parents:
            sibs = [n for n in self.graph.predecessors(p) if n != property_path]
            siblings.extend(sibs)

        return {
            "center": {
                "path": property_path,
                "name": center.get("name"),
                "description": center.get("description", ""),
                "type": center.get("type"),
                "defaultValue": center.get("defaultValue"),
                "since": center.get("since"),
                "samples": center.get("samples", []),
            },
            "ancestors": ancestors,
            "siblings": siblings,
            "descendants": descendants,
        }

    # ---------- main search ----------

    def search(
        self,
        user_query: str,
        embed_fn,              # callable: str -> List[float] or np.ndarray
        top_k: int = 3,
        subgraph_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Returns top-k results with:
          - final score (reranked),
          - base cosine,
          - small debug block,
          - subgraph neighborhood.
        """
        results: List[Dict[str, Any]] = []

        # Quick guard: empty graph/embeddings
        if not self.embeddings or self.graph.number_of_nodes() == 0:
            return results

        # Embed query safely
        try:
            q_vec = np.array(embed_fn(user_query))
            print(f"[search]    - q_vec: {q_vec}")
        except Exception as e:
            # If embedding fails, return empty results rather than None
            print(f"[search] Query embedding failed: {e}")
            return results

        q = self._normalize(q_vec)
        print(f"[search]    - q_vec.normalize: {q}")

        # Stage A: Base dense similarities
        base_scores: List[Tuple[str, float]] = []
        for nid, vec in self.embeddings.items():
            try:
                base_scores.append((nid, self._cosine(q, vec)))
            except Exception as e:
                # Skip malformed vectors
                print(f"[search] Cosine failed for node {nid}: {e}")

        # Determine query intent re: chart type
        mentioned_types = set(self._mentioned_series_types(user_query))
        query_is_generic = (len(mentioned_types) == 0)
        print(f"[search]    - mentioned types and is query generic: {mentioned_types} | {query_is_generic}")

        # Stage B: Top-down reranking using hierarchical priors
        reranked: List[Tuple[str, float, float, Dict[str, float], Dict[str, Any]]] = []

        for nid, base in base_scores:
            nd = self.graph.nodes[nid]
            name = nd.get("name", "")
            desc = nd.get("description", "")
            path = nid

            fam = self._family(path)
            is_generic_family = (fam in self.GENERIC_FAMILIES)
            has_series_type, type_token = self._contains_series_type(path)
            depth = self._depth(path)

            boost = 0.0
            penalty = 0.0
            reasons = {}

            # 1) Prefer generic/global families for generic queries
            if query_is_generic and is_generic_family:
                boost += self.w["generic_boost"]
                reasons["generic_boost"] = self.w["generic_boost"]

            # 2) Penalize specific series types if the query is generic
            if query_is_generic and has_series_type and not is_generic_family:
                penalty += self.w["specific_penalty"]
                reasons["specific_penalty"] = self.w["specific_penalty"]

            # 3) If query mentions a specific type, boost matching paths
            if not query_is_generic and has_series_type and type_token in mentioned_types:
                boost += self.w["specific_boost"]
                reasons["specific_boost"] = self.w["specific_boost"]

            # 4) Light depth penalty beyond the free threshold
            if depth > self.depth_free:
                extra = (depth - self.depth_free) * self.w["depth_penalty"]
                penalty += extra
                reasons["depth_penalty"] = extra

            # 5) Optional lexical keyword boost (tiny)
            if self._has_keywords(user_query, name, desc, path):
                boost += self.w["keyword_boost"]
                reasons["keyword_boost"] = self.w["keyword_boost"]

            final = base * (1.0 + boost) - penalty
            meta = {
                "family": fam,
                "depth": depth,
                "is_generic_family": is_generic_family,
                "has_series_type": has_series_type,
                "type_token": type_token,
                "query_is_generic": query_is_generic
            }
            reranked.append((nid, final, base, reasons, meta))

        # Sort by final score & take top-k
        reranked.sort(key=lambda x: x[1], reverse=True)
        top = reranked[:top_k]

        # Build results + subgraphs
        results: List[Dict[str, Any]] = []
        for nid, final, base, reasons, meta in top:
            nd = self.graph.nodes[nid]
            parents = list(self.graph.successors(nid))       # parent(s)
            children = list(self.graph.predecessors(nid))    # child(ren)

            subgraph = self.build_subgraph(nid, depth=subgraph_depth)

            results.append({
                "path": nid,
                "name": nd.get("name"),
                "description": nd.get("description", ""),
                "type": nd.get("type"),
                "defaultValue": nd.get("defaultValue"),
                "since": nd.get("since"),
                "samples": nd.get("samples", []),
                "ancestors": parents,
                "children": children,
                "relevance": float(final),
                "debug": {
                    "base": float(base),
                    "reasons": reasons,
                    **meta
                },
                "subgraph": subgraph
            })
        return results


import pickle
from utils.embed import embed

# 1) Load graph + embeddings produced by your builder
with open("highcharts_graph.gpickle", "rb") as f:
    data = pickle.load(f)

# print(data)
graph = data["graph"]
embeddings = data.get('embeddings', {})  # node_id -> np.ndarray

# print(embeddings)

# 2) Instantiate the retriever (defaults are sane; weights are tunable)
retriever = TopDownRetriever(
    graph,
    embeddings,
    weights={
        # optional: adjust if you want stronger global preference
        "generic_boost": 0.12,
        "specific_boost": 0.12,
        "specific_penalty": 0.06,
        "depth_penalty": 0.1,
        "keyword_boost": 0.03,
    },
    depth_free=2,               # no penalty up to depth 2
    assume_normalized=False     # set True if you already saved normalized embeddings
)

# 3) Query
query = "set dotted grid lines"
results = retriever.search(query, embed_fn=embed, top_k=2, subgraph_depth=2)

# print(np.array(embed(query)))
# print(results)

# 4) Print sample
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r['path']} (relevance: {r['relevance']:.3f})")
    print(f"    Type: {', '.join(r['type']) if r['type'] else 'N/A'}")
    print(f"    Default: {r['defaultValue']}")
    print(f"    Description: {r['description']}...")

    sg = r["subgraph"]
    if sg:
        print("    -- Subgraph --")
        print(f"    Ancestors: {sg['ancestors'][:6]}{' ...' if len(sg['ancestors']) > 6 else ''}")
        print(f"    Siblings : {sg['siblings'][:6]}{' ...' if len(sg['siblings']) > 6 else ''}")
        print(f"    Children : {sg['descendants'][:6]}{' ...' if len(sg['descendants']) > 6 else ''}")