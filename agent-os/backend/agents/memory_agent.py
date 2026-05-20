"""
MemoryAgent - Persists learnings from each pipeline run.

Writes to:
- memory/schema-cache.json (auto, deterministic)
- memory/column-mappings.json (auto, deterministic)
- memory/corrections.md (only when explicit corrections present)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

MEMORY_MODEL = os.getenv("MEMORY_MODEL", "ollama/llama3.3:8b")
AGENT_FS_ROOT = os.getenv("AGENT_FS_ROOT", "./agent-config")


class MemoryAgent(BaseAgent):
    """Persists schema, column mappings, and corrections after each run."""

    DEFAULT_MODEL: str = MEMORY_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any] = None,
        langfuse_handler: Any = None,
        model: str = None,
        fs_root: str = None,
    ) -> None:
        super().__init__(
            model=model or MEMORY_MODEL,
            cognitive_context=cognitive_context or {},
            langfuse_handler=langfuse_handler,
        )
        self.fs_root = Path(fs_root or AGENT_FS_ROOT)
        self.memory_dir = self.fs_root / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe(name="memory.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Persist learnings from this pipeline run.

        Expected state:
            - file_manifests: list of ingested file manifests
            - semantic_maps / understanding: column-to-semantic resolutions
            - corrections: optional list of explicit corrections
            - job_id, redis_client
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")

        self._emit_event(
            "agent_start",
            {"agent": "memory"},
            job_id,
            redis_client,
        )

        updated: List[str] = []

        manifests = state.get("file_manifests", [])
        if manifests:
            self.update_schema_cache(manifests)
            updated.append("schema-cache.json")

        semantic_maps = state.get("semantic_maps") or state.get("understanding", {})
        if semantic_maps:
            self.update_column_mappings(semantic_maps)
            updated.append("column-mappings.json")

        corrections = state.get("corrections", [])
        for c in corrections:
            self.log_correction(
                original=c.get("original", ""),
                correction=c.get("correction", ""),
                context=c.get("context", ""),
            )
            updated.append("corrections.md")

        state["memory_updated"] = updated

        self._emit_event(
            "agent_complete",
            {"agent": "memory", "files_updated": updated},
            job_id,
            redis_client,
        )
        return state

    # ------------------------------------------------------------------
    # Schema cache
    # ------------------------------------------------------------------

    def update_schema_cache(self, file_manifests: List[Dict[str, Any]]) -> None:
        """Merge new schemas into memory/schema-cache.json."""
        path = self.memory_dir / "schema-cache.json"
        existing = self._load_json(path) or {
            "_version": 1,
            "_description": "Auto-updated by Memory Agent after each pipeline run",
            "schemas": {},
        }

        schemas = existing.setdefault("schemas", {})
        for manifest in file_manifests:
            file_key = self._schema_key(manifest)
            if not file_key:
                continue
            schemas[file_key] = {
                "columns": manifest.get("columns", []),
                "dtypes": manifest.get("dtypes", {}),
                "row_count": manifest.get("row_count", 0),
                "sheets": manifest.get("sheets", []),
                "last_seen": datetime.utcnow().isoformat() + "Z",
                "filename": manifest.get("filename"),
            }

        existing["_last_updated"] = datetime.utcnow().isoformat() + "Z"
        self._save_json(path, existing)
        logger.info("Updated schema cache with %d files", len(file_manifests))

    def _schema_key(self, manifest: Dict[str, Any]) -> str:
        """Derive a stable key for a file manifest.

        Uses filename stem + column-set hash so that the same logical file
        across months matches the same cache entry.
        """
        import hashlib
        filename = manifest.get("filename", "")
        stem = Path(filename).stem if filename else ""

        cols = sorted(manifest.get("columns", []))
        cols_repr = "|".join(c.lower() for c in cols)
        cols_hash = hashlib.md5(cols_repr.encode()).hexdigest()[:8]

        if stem:
            # Normalise periodic suffixes (2024-01, Q1, etc.)
            stem_norm = self._normalize_stem(stem)
            return f"{stem_norm}::{cols_hash}"
        return f"unknown::{cols_hash}"

    def _normalize_stem(self, stem: str) -> str:
        """Strip date/period suffixes from a filename stem."""
        import re
        for pattern in (
            r"[_\-\s]\d{4}[_\-]\d{2}([_\-]\d{2})?$",  # YYYY-MM or YYYY-MM-DD suffix
            r"[_\-\s]Q[1-4][_\-\s]?\d{0,4}$",  # Q1 2024 suffix
            r"[_\-\s]\d{4}$",  # year suffix
            r"[_\-\s]v\d+$",  # version suffix
        ):
            stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
        return stem.strip()

    # ------------------------------------------------------------------
    # Column mappings
    # ------------------------------------------------------------------

    def update_column_mappings(self, semantic_maps: Dict[str, Any]) -> None:
        """Merge new column-name → canonical-name mappings."""
        path = self.memory_dir / "column-mappings.json"
        existing = self._load_json(path) or {
            "_version": 1,
            "_description": "Maps raw column names to canonical names",
            "mappings": {},
        }

        mappings = existing.setdefault("mappings", {})

        # semantic_maps can be either {raw: canonical} or {file_id: {raw: canonical}}
        flat = self._flatten_semantic_maps(semantic_maps)
        for raw, canonical in flat.items():
            if not raw or not canonical:
                continue
            raw_norm = raw.strip().lower()
            if not canonical.strip():
                continue
            # Only overwrite if no prior mapping or the prior mapping was identical
            mappings[raw_norm] = canonical.strip()

        existing["_last_updated"] = datetime.utcnow().isoformat() + "Z"
        self._save_json(path, existing)
        logger.info("Updated column mappings with %d entries", len(flat))

    def _flatten_semantic_maps(self, maps: Any) -> Dict[str, str]:
        """Normalise possibly nested semantic_maps to a flat {raw: canonical} dict."""
        if not isinstance(maps, dict):
            return {}

        # Already flat {raw: canonical}?
        if all(isinstance(v, str) for v in maps.values()):
            return {k: v for k, v in maps.items() if isinstance(k, str)}

        flat: Dict[str, str] = {}
        for outer_v in maps.values():
            if isinstance(outer_v, dict):
                for raw, canonical in outer_v.items():
                    if isinstance(raw, str) and isinstance(canonical, str):
                        flat[raw] = canonical
        return flat

    # ------------------------------------------------------------------
    # Corrections log
    # ------------------------------------------------------------------

    def log_correction(self, original: str, correction: str, context: str = "") -> None:
        """Append a correction entry to memory/corrections.md."""
        path = self.memory_dir / "corrections.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else "# Agent Corrections Log\n\n"

        entry = f"""
### [{datetime.utcnow().strftime("%Y-%m-%d")}] Auto-logged correction
- **Original**: {original}
- **Correct behavior**: {correction}
"""
        if context:
            entry += f"- **Context**: {context}\n"

        path.write_text(existing + entry, encoding="utf-8")
        logger.info("Logged correction to %s", path)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> Dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return None

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
