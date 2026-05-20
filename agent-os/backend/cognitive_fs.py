"""
CognitiveFSLoader – reads/writes the agent-config directory tree and
assembles a structured system prompt for LLM agents.

Directory layout expected under <agent_config_root>/:
    soul.md
    knowledge/
        *.md
    skills/
        *.md
    memory/
        corrections.md
        schema-cache.json
        column-mappings.json
    pipelines/
        *.yaml | *.json | *.md
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveFSLoader:
    """
    Reads the agent-config filesystem and assembles context for LLM agents.

    Parameters
    ----------
    agent_config_root:
        Absolute path to the ``agent-config/`` directory.
    """

    def __init__(self, agent_config_root: str | Path) -> None:
        self.root = Path(agent_config_root).resolve()
        if not self.root.exists():
            logger.warning("agent-config root does not exist: %s", self.root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self, rel_path: str) -> str | None:
        """Return file content as string, or None if missing."""
        full = self.root / rel_path
        try:
            return full.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.debug("CognitiveFS: file not found – %s", full)
            return None
        except Exception as exc:
            logger.warning("CognitiveFS: could not read %s: %s", full, exc)
            return None

    def _read_json(self, rel_path: str) -> dict[str, Any]:
        """Return parsed JSON dict, or empty dict if missing/invalid."""
        raw = self._read_file(rel_path)
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("CognitiveFS: invalid JSON in %s: %s", rel_path, exc)
            return {}

    def _glob_md(self, directory: str) -> dict[str, str]:
        """Return {filename: content} for every *.md in *directory*."""
        dir_path = self.root / directory
        result: dict[str, str] = {}
        if not dir_path.is_dir():
            return result
        for md_file in sorted(dir_path.glob("*.md")):
            content = self._read_file(str(md_file.relative_to(self.root)))
            if content is not None:
                result[md_file.name] = content
        return result

    @staticmethod
    def _keyword_match(task: str, text: str) -> bool:
        """
        Return True if any meaningful word (≥4 chars) from *task* appears in
        *text* (case-insensitive).
        """
        words = {w.lower() for w in re.findall(r"\w{4,}", task)}
        text_lower = text.lower()
        return any(w in text_lower for w in words)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_context(self, task: str) -> dict[str, Any]:
        """
        Read all relevant agent-config files for *task* and return them
        as a structured dictionary.

        Keys returned
        -------------
        soul          – content of soul.md (str | None)
        knowledge     – {filename: content} for all knowledge/*.md
        skills        – {filename: content} for *matching* skills/*.md
        corrections   – content of memory/corrections.md (str | None)
        schema_cache  – parsed memory/schema-cache.json (dict)
        column_mappings – parsed memory/column-mappings.json (dict)
        task          – the original task string
        """
        soul = self._read_file("soul.md")
        knowledge = self._glob_md("knowledge")

        # Load only skills whose content is relevant to the task
        all_skills = self._glob_md("skills")
        relevant_skills: dict[str, str] = {}
        for fname, content in all_skills.items():
            # Match against filename (strip .md) or content body
            searchable = fname.replace("-", " ").replace("_", " ") + " " + content
            if self._keyword_match(task, searchable):
                relevant_skills[fname] = content

        corrections = self._read_file("memory/corrections.md")
        schema_cache = self._read_json("memory/schema-cache.json")
        column_mappings = self._read_json("memory/column-mappings.json")

        return {
            "soul": soul,
            "knowledge": knowledge,
            "skills": relevant_skills,
            "corrections": corrections,
            "schema_cache": schema_cache,
            "column_mappings": column_mappings,
            "task": task,
        }

    def assemble_system_prompt(self, context: dict[str, Any]) -> str:
        """
        Combine all loaded context into a single structured system prompt.

        Parameters
        ----------
        context:
            Dictionary returned by :meth:`load_context`.
        """
        sections: list[str] = []

        # --- Soul / identity -------------------------------------------------
        if context.get("soul"):
            sections.append("# Agent Identity\n\n" + context["soul"].strip())

        # --- Knowledge base --------------------------------------------------
        if context.get("knowledge"):
            kb_parts = []
            for fname, content in context["knowledge"].items():
                title = fname.replace("-", " ").replace("_", " ").replace(".md", "").title()
                kb_parts.append(f"## {title}\n\n{content.strip()}")
            sections.append("# Knowledge Base\n\n" + "\n\n---\n\n".join(kb_parts))

        # --- Relevant skills -------------------------------------------------
        if context.get("skills"):
            skill_parts = []
            for fname, content in context["skills"].items():
                title = fname.replace("-", " ").replace("_", " ").replace(".md", "").title()
                skill_parts.append(f"## {title}\n\n{content.strip()}")
            sections.append("# Available Skills\n\n" + "\n\n---\n\n".join(skill_parts))

        # --- Memory: corrections ---------------------------------------------
        if context.get("corrections"):
            sections.append(
                "# Past Corrections & Lessons Learned\n\n"
                + context["corrections"].strip()
            )

        # --- Memory: schema cache --------------------------------------------
        if context.get("schema_cache"):
            sections.append(
                "# Schema Cache\n\n```json\n"
                + json.dumps(context["schema_cache"], indent=2)
                + "\n```"
            )

        # --- Memory: column mappings -----------------------------------------
        if context.get("column_mappings"):
            sections.append(
                "# Column Mappings\n\n```json\n"
                + json.dumps(context["column_mappings"], indent=2)
                + "\n```"
            )

        # --- Current task ----------------------------------------------------
        if context.get("task"):
            sections.append("# Current Task\n\n" + context["task"].strip())

        return "\n\n" + "\n\n---\n\n".join(sections) + "\n"

    def update_memory(self, key: str, value: dict[str, Any]) -> None:
        """
        Merge *value* into a JSON memory file.

        Parameters
        ----------
        key:
            One of ``"schema_cache"`` or ``"column_mappings"``.
        value:
            Dict to merge into the existing file.
        """
        key_to_path: dict[str, str] = {
            "schema_cache": "memory/schema-cache.json",
            "column_mappings": "memory/column-mappings.json",
        }
        if key not in key_to_path:
            raise ValueError(
                f"Unknown memory key '{key}'. Valid keys: {list(key_to_path)}"
            )

        rel_path = key_to_path[key]
        existing = self._read_json(rel_path)
        existing.update(value)

        full_path = self.root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("CognitiveFS: updated %s", rel_path)

    def get_file(self, path: str) -> str:
        """
        Read any file inside agent-config by its *relative* path.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the resolved path escapes the agent-config root (path traversal).
        """
        full = (self.root / path).resolve()
        if not str(full).startswith(str(self.root)):
            raise ValueError(f"Path traversal attempt blocked: {path!r}")
        if not full.exists():
            raise FileNotFoundError(f"File not found in agent-config: {path!r}")
        return full.read_text(encoding="utf-8")

    def put_file(self, path: str, content: str) -> None:
        """
        Write *content* to *path* (relative to agent-config root).

        Creates parent directories as needed.

        Raises
        ------
        ValueError
            If the resolved path escapes the agent-config root.
        """
        full = (self.root / path).resolve()
        if not str(full).startswith(str(self.root)):
            raise ValueError(f"Path traversal attempt blocked: {path!r}")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        logger.info("CognitiveFS: wrote %s (%d bytes)", path, len(content))

    def list_files(self) -> dict[str, Any]:
        """
        Return a nested dict tree of every file in agent-config.

        Leaf values are::

            {"path": "<rel_path>", "size": <int>, "modified_at": "<iso>"}

        Returns
        -------
        dict with keys "root" and "tree".
        """
        tree: dict[str, Any] = {}

        if not self.root.is_dir():
            return {"root": str(self.root), "tree": tree}

        for file_path in sorted(self.root.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.root)
            parts = rel.parts

            # Walk/build nested dicts for directory components
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})

            stat = file_path.stat()
            node[parts[-1]] = {
                "path": rel.as_posix(),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

        return {"root": str(self.root), "tree": tree}
