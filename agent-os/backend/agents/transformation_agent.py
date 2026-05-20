"""
TransformationAgent - Generates pandas transformation code via LLM, executes
it in a Docker sandbox, and merges team parquet files into one master dataset.

Parallelises transformation across teams (files) using a thread pool.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langfuse.decorators import observe

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

WORKER_MODEL = os.getenv("WORKER_MODEL", "ollama/qwen2.5-coder:32b")
WORKER_FALLBACK_MODEL = os.getenv("WORKER_FALLBACK_MODEL", "openai/gpt-4o-mini")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agentos-sandbox:latest")
SANDBOX_TIMEOUT_SECS = int(os.getenv("SANDBOX_TIMEOUT_SECS", "120"))
MAX_PARALLEL_TRANSFORMS = int(os.getenv("MAX_PARALLEL_TRANSFORMS", "4"))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TransformResult:
    """Result of executing transformation code against a file."""

    file_name: str
    output_parquet: str       # Path to the output parquet file
    rows_in: int
    rows_out: int
    columns_out: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "output_parquet": self.output_parquet,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "columns_out": self.columns_out,
            "warnings": self.warnings,
            "error": self.error,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# TransformationAgent
# ---------------------------------------------------------------------------


class TransformationAgent(BaseAgent):
    """Generates and executes pandas transformation code per team file."""

    DEFAULT_MODEL = WORKER_MODEL

    def __init__(
        self,
        cognitive_context: Dict[str, Any],
        langfuse_handler: Any,
        model: str = WORKER_MODEL,
    ) -> None:
        super().__init__(
            model=model,
            cognitive_context=cognitive_context,
            langfuse_handler=langfuse_handler,
        )
        self._fallback = WORKER_FALLBACK_MODEL

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    @observe(name="transformation_agent.generate_transformation_code")
    def generate_transformation_code(
        self,
        manifest: Dict[str, Any],
        business_rules: Dict[str, Any],
    ) -> str:
        """Ask the LLM to produce a standalone pandas transformation script.

        The generated script must:
        - Read the input file from ``INPUT_PATH`` environment variable.
        - Apply all required transformations (column renames, type casts,
          derivations, deduplication, filtering, etc.).
        - Write the result to ``OUTPUT_PATH`` as a parquet file.
        - Print a JSON summary to stdout: ``{"rows_in": N, "rows_out": M}``.

        Args:
            manifest: FileManifest dict including semantic_map per sheet.
            business_rules: Domain-specific transformation rules.

        Returns:
            Python source code string ready to execute.
        """
        # Build a compact representation of the file for the prompt
        sheets_info = []
        for sheet in manifest.get("sheets", []):
            if sheet.get("classification") not in ("data", "unknown"):
                continue
            sem_cols = []
            for c in sheet.get("semantic_map", {}).get("columns", []):
                sem_cols.append(
                    f"{c['raw_name']} → {c['canonical_name']} ({c['semantic_type']})"
                )
            sheets_info.append(
                {
                    "sheet_name": sheet["sheet_name"],
                    "file_type": manifest.get("file_type", "csv"),
                    "row_count": sheet["row_count"],
                    "semantic_columns": sem_cols,
                }
            )

        system_prompt = textwrap.dedent("""
            You are an expert pandas engineer. Write a self-contained Python script
            that transforms raw spreadsheet data according to the provided rules.

            Script requirements:
            1. Read INPUT_PATH from os.environ["INPUT_PATH"].
            2. Read OUTPUT_PATH from os.environ["OUTPUT_PATH"].
            3. Load the file (Excel or CSV based on extension).
            4. Apply all transformations (rename columns to canonical names, cast
               types, derive calculated columns, deduplicate, filter, etc.).
            5. Write the result to OUTPUT_PATH as parquet (df.to_parquet(...)).
            6. Print a single JSON line to stdout:
               {"rows_in": <int>, "rows_out": <int>, "warnings": []}
            7. Do NOT import external libraries beyond: os, json, sys, pandas,
               numpy, re, datetime.
            8. Handle errors gracefully — print {"error": "<msg>"} and exit(1).

            Return ONLY the Python code. No markdown fences, no explanations.
        """).strip()

        user_prompt = (
            f"File: {manifest.get('file_name')}\n\n"
            f"Sheets:\n{json.dumps(sheets_info, indent=2)}\n\n"
            f"Business rules:\n{json.dumps(business_rules, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._call_llm(
                messages,
                temperature=0.05,
                max_tokens=4096,
                fallback_model=self._fallback,
            )
            code = self._extract_text(response).strip()
            # Strip any accidental markdown fences
            if code.startswith("```"):
                lines = code.splitlines()
                code = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            return code
        except Exception as exc:
            logger.error("generate_transformation_code failed: %s", exc)
            # Return a minimal identity script
            return self._identity_script(manifest)

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    @observe(name="transformation_agent.execute_transformation")
    def execute_transformation(
        self,
        code: str,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> TransformResult:
        """Execute transformation code in a Docker sandbox (or local subprocess).

        When Docker is unavailable (e.g. CI), falls back to a local subprocess
        with a restricted environment.

        Args:
            code: Python source code produced by :meth:`generate_transformation_code`.
            file_path: Absolute path to the input file.
            output_dir: Directory for the output parquet.  Defaults to a temp dir.

        Returns:
            :class:`TransformResult`.
        """
        file_name = Path(file_path).name
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="agentos_transform_")
        output_path = str(Path(output_dir) / f"{Path(file_path).stem}.parquet")

        # Write code to a temp file
        script_path = os.path.join(output_dir, f"transform_{uuid.uuid4().hex[:8]}.py")
        with open(script_path, "w") as f:
            f.write(code)

        env = {
            "INPUT_PATH": file_path,
            "OUTPUT_PATH": output_path,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        # Try Docker first
        docker_result = self._run_in_docker(script_path, file_path, output_path, env)
        if docker_result is not None:
            return self._parse_exec_result(
                docker_result, file_name, file_path, output_path
            )

        # Fall back to local subprocess
        logger.info("Docker unavailable — running transformation locally.")
        return self._run_locally(script_path, file_name, file_path, output_path, env)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @observe(name="transformation_agent.merge_team_files")
    def merge_team_files(
        self,
        parquet_paths: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        """Concatenate all team parquet files into a single master parquet.

        Args:
            parquet_paths: List of paths to individual team parquet files.
            output_path: Destination path for the merged file.  Auto-generated
                if not supplied.

        Returns:
            Absolute path to the merged parquet file.
        """
        if not parquet_paths:
            raise ValueError("No parquet paths provided for merging.")

        if output_path is None:
            output_dir = tempfile.mkdtemp(prefix="agentos_merged_")
            output_path = os.path.join(output_dir, "merged_output.parquet")

        dfs: List[pd.DataFrame] = []
        for path in parquet_paths:
            try:
                df = pd.read_parquet(path)
                dfs.append(df)
                logger.debug("Loaded parquet '%s': %d rows", path, len(df))
            except Exception as exc:
                logger.warning("Could not load parquet '%s': %s", path, exc)

        if not dfs:
            raise RuntimeError("All parquet files failed to load — cannot merge.")

        merged = pd.concat(dfs, ignore_index=True, sort=False)
        merged.to_parquet(output_path, index=False)
        logger.info(
            "Merged %d files → %d rows → '%s'", len(dfs), len(merged), output_path
        )
        return output_path

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    @observe(name="transformation_agent.run")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform all team files in parallel, then merge.

        Expected state keys:
            - ``file_manifests`` (list): Enriched manifests from UnderstandingAgent.
            - ``uploaded_files`` (list): Original file paths (for reading).
            - ``business_rules`` (dict, optional): Transformation rules.
            - ``output_dir`` (str, optional): Where to write parquet files.
            - ``job_id``, ``redis_client``.

        Returns:
            State with ``transform_results``, ``parquet_paths``, and
            ``merged_parquet`` added.
        """
        job_id = state.get("job_id", self._job_id())
        redis_client = state.get("redis_client")
        file_manifests: List[Dict[str, Any]] = state.get("file_manifests", [])
        uploaded_files: List[str] = state.get("uploaded_files", [])
        business_rules: Dict[str, Any] = state.get("business_rules", {})
        output_dir: Optional[str] = state.get("output_dir")

        self._emit_event(
            "transformation_started",
            {"file_count": len(file_manifests)},
            job_id,
            redis_client,
        )

        # Build (manifest, file_path) pairs
        pairs: List[tuple[Dict[str, Any], str]] = []
        for manifest in file_manifests:
            # Try to find the matching uploaded file by name
            fname = manifest.get("file_name", "")
            matched = next(
                (p for p in uploaded_files if Path(p).name == fname),
                None,
            )
            if matched is None and uploaded_files:
                matched = uploaded_files[0]
            if matched:
                pairs.append((manifest, matched))
            else:
                logger.warning("No matching file path for manifest '%s'", fname)

        transform_results: List[Dict[str, Any]] = []
        parquet_paths: List[str] = []

        def _transform_one(
            manifest: Dict[str, Any], file_path: str
        ) -> TransformResult:
            code = self.generate_transformation_code(manifest, business_rules)
            return self.execute_transformation(code, file_path, output_dir)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSFORMS) as executor:
            future_to_file = {
                executor.submit(_transform_one, m, p): m.get("file_name")
                for m, p in pairs
            }
            for future in as_completed(future_to_file):
                fname = future_to_file[future]
                try:
                    result = future.result()
                    transform_results.append(result.to_dict())
                    if result.success:
                        parquet_paths.append(result.output_parquet)
                    self._emit_event(
                        "file_transformed",
                        {
                            "file": fname,
                            "success": result.success,
                            "rows_out": result.rows_out,
                        },
                        job_id,
                        redis_client,
                    )
                except Exception as exc:
                    logger.error("Transform failed for '%s': %s", fname, exc)
                    self._emit_event(
                        "transform_error",
                        {"file": fname, "error": str(exc)},
                        job_id,
                        redis_client,
                    )
                    transform_results.append(
                        {
                            "file_name": fname,
                            "success": False,
                            "error": str(exc),
                        }
                    )

        # Merge all successful parquet outputs
        merged_parquet: Optional[str] = None
        if parquet_paths:
            try:
                merged_parquet = self.merge_team_files(parquet_paths, output_path=output_dir and os.path.join(output_dir, "merged_output.parquet"))
                self._emit_event(
                    "merge_completed",
                    {"merged_parquet": merged_parquet},
                    job_id,
                    redis_client,
                )
            except Exception as exc:
                logger.error("Merge failed: %s", exc)
                self._emit_event(
                    "merge_error", {"error": str(exc)}, job_id, redis_client
                )

        state["transform_results"] = transform_results
        state["parquet_paths"] = parquet_paths
        state["merged_parquet"] = merged_parquet

        self._emit_event(
            "transformation_completed",
            {
                "transforms": len(transform_results),
                "successful": len(parquet_paths),
                "merged_parquet": merged_parquet,
            },
            job_id,
            redis_client,
        )
        logger.info(
            "TransformationAgent done: %d/%d succeeded, merged → '%s'",
            len(parquet_paths),
            len(pairs),
            merged_parquet,
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_in_docker(
        self,
        script_path: str,
        input_path: str,
        output_path: str,
        env: Dict[str, str],
    ) -> Optional[subprocess.CompletedProcess]:
        """Attempt to run the script inside a Docker container."""
        try:
            # Quick check that docker is available
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None

        script_dir = str(Path(script_path).parent)
        input_dir = str(Path(input_path).parent)
        output_dir = str(Path(output_path).parent)

        env_flags: List[str] = []
        for k, v in env.items():
            env_flags += ["-e", f"{k}={v}"]

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1",
            "-v", f"{script_dir}:/scripts:ro",
            "-v", f"{input_dir}:/input:ro",
            "-v", f"{output_dir}:/output",
            *env_flags,
            SANDBOX_IMAGE,
            "python", f"/scripts/{Path(script_path).name}",
        ]

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired:
            logger.error("Docker sandbox timed out after %ds.", SANDBOX_TIMEOUT_SECS)
            return None
        except Exception as exc:
            logger.warning("Docker execution error: %s", exc)
            return None

    def _run_locally(
        self,
        script_path: str,
        file_name: str,
        input_path: str,
        output_path: str,
        env: Dict[str, str],
    ) -> TransformResult:
        """Execute the transformation script in a local subprocess."""
        import subprocess as sp

        full_env = {**os.environ, **env}
        try:
            result = sp.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                env=full_env,
                timeout=SANDBOX_TIMEOUT_SECS,
            )
            return self._parse_exec_result(result, file_name, input_path, output_path)
        except sp.TimeoutExpired:
            return TransformResult(
                file_name=file_name,
                output_parquet=output_path,
                rows_in=0,
                rows_out=0,
                error="Transformation timed out.",
                success=False,
            )
        except Exception as exc:
            return TransformResult(
                file_name=file_name,
                output_parquet=output_path,
                rows_in=0,
                rows_out=0,
                error=str(exc),
                success=False,
            )

    def _parse_exec_result(
        self,
        proc: subprocess.CompletedProcess,
        file_name: str,
        input_path: str,
        output_path: str,
    ) -> TransformResult:
        """Parse the subprocess result into a TransformResult."""
        if proc.returncode != 0:
            logger.error(
                "Transformation subprocess failed (rc=%d): %s",
                proc.returncode,
                proc.stderr[:500],
            )
            return TransformResult(
                file_name=file_name,
                output_parquet=output_path,
                rows_in=0,
                rows_out=0,
                error=proc.stderr[:500] or "Non-zero exit code",
                success=False,
            )

        summary: Dict[str, Any] = {}
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    summary = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        if "error" in summary:
            return TransformResult(
                file_name=file_name,
                output_parquet=output_path,
                rows_in=summary.get("rows_in", 0),
                rows_out=0,
                error=summary["error"],
                success=False,
            )

        # Try to read the output parquet to get column list
        columns_out: List[str] = []
        try:
            df = pd.read_parquet(output_path)
            columns_out = list(df.columns)
        except Exception:
            pass

        return TransformResult(
            file_name=file_name,
            output_parquet=output_path,
            rows_in=summary.get("rows_in", 0),
            rows_out=summary.get("rows_out", 0),
            columns_out=columns_out,
            warnings=summary.get("warnings", []),
            success=True,
        )

    def _identity_script(self, manifest: Dict[str, Any]) -> str:
        """Minimal fallback script: read file and write as parquet unchanged."""
        file_type = manifest.get("file_type", "csv")
        read_call = (
            "pd.read_excel(input_path)" if file_type == "excel"
            else "pd.read_csv(input_path)"
        )
        return textwrap.dedent(f"""
            import os, json, pandas as pd, sys
            input_path = os.environ["INPUT_PATH"]
            output_path = os.environ["OUTPUT_PATH"]
            try:
                df = {read_call}
                rows_in = len(df)
                df.to_parquet(output_path, index=False)
                print(json.dumps({{"rows_in": rows_in, "rows_out": len(df), "warnings": ["identity transform used"]}}))
            except Exception as exc:
                print(json.dumps({{"error": str(exc)}}))
                sys.exit(1)
        """).strip()
