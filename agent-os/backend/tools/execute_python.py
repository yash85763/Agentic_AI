"""
Docker sandbox executor for safe Python code execution.

Security hardening (per architectural review, Section 2.1 + 7):

  NO HOST BIND MOUNTS
    Input files are copied into the container via TAR archive, never
    bind-mounted. This eliminates the host filesystem exposure that comes
    with rootful Docker bind mounts.

  CUSTOM SECCOMP PROFILE
    sandbox-seccomp.json blocks dangerous syscalls: ptrace, keyctl,
    personality, mbind, migrate_pages, and kernel-module operations.
    Falls back gracefully if the profile file is not found (logs a warning).

  AST-BASED CODE VALIDATION
    Every code string is parsed and walked with Python's ast module before
    a container is created. Code that imports disallowed modules, calls
    exec()/eval(), or attempts direct socket access is rejected immediately
    without spinning up a container.

  POST-EXECUTION CONTRACT CHECKS
    run_contract_checks() verifies financial aggregation integrity:
      - sum of output ≈ sum of input for the amount column (±$0.01)
      - row count plausibility
      - no extreme outliers (>3 σ) without flags
    Returns a contract_status suitable for ConfidenceBundle.contract_status.

  NO NEW PRIVILEGES / ALL CAPS DROPPED
    security_opt=["no-new-privileges:true"] prevents setuid escalation.
    cap_drop=["ALL"] removes all Linux capabilities.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import docker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CodeRejectedError(Exception):
    """Raised when AST validation rejects code before execution."""


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Execute Python code in an isolated Docker container.

    See module docstring for the full list of security controls.
    """

    DEFAULT_IMAGE = "agentos-sandbox:latest"
    MAX_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        max_workers: int = 8,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
    ):
        self.client = docker.from_env()
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._policy_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Security: policy + AST validation
    # ------------------------------------------------------------------

    def _load_policy(self) -> dict:
        """Load sandbox_policy.yaml; fall back to hardcoded defaults."""
        if self._policy_cache is not None:
            return self._policy_cache

        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "sandbox_policy.yaml"),
            "/app/backend/sandbox_policy.yaml",
            "sandbox_policy.yaml",
        ]
        for p in search_paths:
            if os.path.exists(p):
                try:
                    import yaml
                    with open(p) as f:
                        self._policy_cache = yaml.safe_load(f) or {}
                    logger.debug("Loaded sandbox policy from %s", p)
                    return self._policy_cache
                except Exception as exc:
                    logger.warning("Failed to load sandbox_policy.yaml at %s: %s", p, exc)

        self._policy_cache = {
            "approved_modules": [
                "pandas", "numpy", "scipy", "statsmodels", "sklearn",
                "openpyxl", "xlrd", "pyarrow", "duckdb",
                "json", "csv", "os", "pathlib", "math", "statistics",
                "decimal", "datetime", "collections", "itertools",
                "functools", "typing", "io", "re", "string", "sys",
                "dateutil", "holidays", "pytz", "copy", "operator",
                "struct", "hashlib", "base64", "uuid", "random",
            ],
            "blocked_builtins": ["exec", "eval", "compile", "__import__"],
            "contract_check": {
                "sum_tolerance_absolute": 0.01,
                "outlier_std_threshold": 3.0,
                "row_ratio_min": 0.01,
                "row_ratio_max": 100.0,
            },
        }
        return self._policy_cache

    def _validate_code_ast(self, code: str) -> tuple[bool, str]:
        """Validate code against security policy using AST analysis.

        Returns:
            (is_valid, rejection_reason_or_empty_string)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        policy = self._load_policy()
        approved: set[str] = set(policy.get("approved_modules", []))
        blocked_builtins: set[str] = set(
            policy.get("blocked_builtins", ["exec", "eval", "compile"])
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if approved and root not in approved:
                        return False, (
                            f"Blocked import: '{alias.name}' is not in approved_modules. "
                            f"Add it to sandbox_policy.yaml to allow it."
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if approved and root not in approved:
                        return False, (
                            f"Blocked import: 'from {node.module}' is not in approved_modules."
                        )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in blocked_builtins:
                    return False, f"Blocked builtin: '{node.func.id}()' is not permitted."

            # Block direct socket attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in ("socket", "AF_INET", "AF_INET6", "SOCK_STREAM"):
                    return False, "Blocked: direct network socket access is not permitted."

        return True, ""

    def _get_seccomp_profile(self) -> str:
        """Return the seccomp profile path, or 'unconfined' if not found.

        In production, mount sandbox-seccomp.json to /app/sandbox-seccomp.json
        in the backend container so Docker can reference it.
        """
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "sandbox-seccomp.json"),
            "/app/sandbox-seccomp.json",
            "sandbox-seccomp.json",
        ]
        for p in search_paths:
            if os.path.exists(p):
                return os.path.abspath(p)

        logger.warning(
            "sandbox-seccomp.json not found — running without custom seccomp. "
            "Mount it to /app/sandbox-seccomp.json for full syscall hardening."
        )
        return "unconfined"

    # ------------------------------------------------------------------
    # TAR helpers
    # ------------------------------------------------------------------

    def _write_code_to_tar(self, code: str) -> bytes:
        """Pack Python code string into a TAR archive as user_code.py."""
        buf = io.BytesIO()
        encoded = code.encode("utf-8")
        info = tarfile.TarInfo(name="user_code.py")
        info.size = len(encoded)
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.addfile(info, io.BytesIO(encoded))
        buf.seek(0)
        return buf.read()

    def _copy_input_files_to_container(
        self, container, input_files: dict[str, str]
    ) -> None:
        """Copy input files into the container via TAR archive.

        Files are written to the container's overlay layer at /sandbox/inputs/
        before start. Docker-level writes bypass read_only=True; the container
        process reads them as a read-only overlay mount.

        No host directories are bind-mounted — this eliminates the host
        filesystem exposure class of vulnerability.
        """
        if not input_files:
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for filename, host_path in input_files.items():
                abs_path = os.path.abspath(host_path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"Input file not found: {abs_path!r}")
                arcname = os.path.basename(filename)
                tar.add(abs_path, arcname=arcname)

        buf.seek(0)
        container.put_archive("/sandbox/inputs", buf.read())

    def _collect_output_files(self, container) -> dict[str, bytes]:
        """Extract files written to /sandbox/outputs inside the container."""
        output_files: dict[str, bytes] = {}
        try:
            bits, _ = container.get_archive("/sandbox/outputs")
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            output_files[os.path.basename(member.name)] = f.read()
        except (docker.errors.NotFound, Exception):
            pass
        return output_files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        input_files: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute Python code in an isolated Docker container.

        Parameters
        ----------
        code : str
            Python source to execute. Validated against sandbox_policy.yaml
            via AST analysis before any container is created.
        input_files : dict | None
            {container_filename: host_path}. Copied via TAR — no bind mounts.
        timeout : int
            Wall-clock seconds (capped at MAX_TIMEOUT_SECONDS).

        Returns
        -------
        dict with keys:
            stdout, stderr, exit_code, duration_ms, output_files,
            timed_out, ast_rejected, rejection_reason
        """
        timeout = min(timeout, self.MAX_TIMEOUT_SECONDS)

        # AST validation — reject before creating any container
        is_valid, rejection_reason = self._validate_code_ast(code)
        if not is_valid:
            logger.warning(
                "Code rejected by AST validator: %s", rejection_reason
            )
            return {
                "stdout": "",
                "stderr": f"Code rejected by security policy: {rejection_reason}",
                "exit_code": -2,
                "duration_ms": 0,
                "output_files": {},
                "timed_out": False,
                "ast_rejected": True,
                "rejection_reason": rejection_reason,
            }

        seccomp_profile = self._get_seccomp_profile()
        security_opts = ["no-new-privileges:true"]
        if seccomp_profile != "unconfined":
            security_opts.append(f"seccomp={seccomp_profile}")

        container = None
        start_ts = time.monotonic()
        timed_out = False

        try:
            # Create container — NO host bind mounts
            container = self.client.containers.create(
                image=self.image,
                command=["python", "/sandbox/user_code.py"],
                network_disabled=True,
                mem_limit=self.mem_limit,
                nano_cpus=self.nano_cpus,
                working_dir="/sandbox",
                read_only=True,
                tmpfs={
                    "/tmp": "size=64m",
                    # /sandbox/outputs is writable tmpfs for code output
                    "/sandbox/outputs": "size=256m",
                    # /sandbox/inputs is NOT tmpfs — files copied via put_archive
                    # persist in the container overlay layer (readable, not writable
                    # by container process due to read_only=True)
                },
                user="1001",  # sandbox user (uid 1001) — matches sandbox Dockerfile
                security_opt=security_opts,
                cap_drop=["ALL"],
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
            )

            # Copy code and input files before container starts
            container.put_archive("/sandbox", self._write_code_to_tar(code))
            if input_files:
                self._copy_input_files_to_container(container, input_files)

            # Start and wait
            container.start()
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                timed_out = True
                exit_code = -1
                try:
                    container.kill()
                except Exception:
                    pass

            # Collect logs
            raw_logs = container.logs(stdout=True, stderr=True, stream=False)
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            try:
                for frame_type, frame_data in self._demux_logs(container):
                    if frame_type == 1:
                        stdout_chunks.append(
                            frame_data.decode("utf-8", errors="replace")
                        )
                    else:
                        stderr_chunks.append(
                            frame_data.decode("utf-8", errors="replace")
                        )
            except Exception:
                stdout_chunks = [raw_logs.decode("utf-8", errors="replace")]

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            if not stdout and not stderr and raw_logs:
                stdout = raw_logs.decode("utf-8", errors="replace")

            output_files = self._collect_output_files(container)

        except docker.errors.ImageNotFound:
            logger.error(
                "Sandbox image %r not found — build it with: "
                "docker build -t agentos-sandbox:latest ./sandbox-image/",
                self.image,
            )
            raise
        except Exception:
            logger.exception("Unexpected error during sandbox execution")
            raise
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        duration_ms = int((time.monotonic() - start_ts) * 1000)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "output_files": output_files,
            "timed_out": timed_out,
            "ast_rejected": False,
            "rejection_reason": None,
        }

    def run_contract_checks(
        self,
        input_files: dict[str, str],
        output_file_bytes: dict[str, bytes],
        amount_column: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run financial contract checks on input vs output parquet data.

        Checks (per Section 7, Step 3):
          1. Sum conservation: output sum ≈ input sum for amount_column (±$0.01)
          2. Row count plausibility (output/input ratio in [0.01, 100])
          3. No extreme outliers (>3σ) in output numeric columns

        Runs on the host using pandas — not user-generated code.

        Returns:
            {"contract_status": "pass"|"fail"|"skipped", "checks": [...]}
        """
        policy = self._load_policy()
        cc = policy.get("contract_check", {})
        tolerance = float(cc.get("sum_tolerance_absolute", 0.01))
        std_threshold = float(cc.get("outlier_std_threshold", 3.0))
        row_ratio_min = float(cc.get("row_ratio_min", 0.01))
        row_ratio_max = float(cc.get("row_ratio_max", 100.0))

        try:
            import io as _io
            import pandas as pd

            # Find first parquet in inputs and outputs
            in_pq = next(
                (p for p in input_files if p.endswith(".parquet")), None
            )
            out_pq = next(
                (n for n in output_file_bytes if n.endswith(".parquet")), None
            )

            if not in_pq or not out_pq:
                return {
                    "contract_status": "skipped",
                    "reason": "no parquet files found for contract checks",
                }

            df_in = pd.read_parquet(input_files[in_pq])
            df_out = pd.read_parquet(_io.BytesIO(output_file_bytes[out_pq]))

        except Exception as exc:
            logger.warning("Contract check setup failed: %s", exc)
            return {"contract_status": "skipped", "error": str(exc)}

        checks: list[dict] = []
        overall = "pass"

        # Check 1: Sum conservation
        if amount_column and amount_column in df_in.columns and amount_column in df_out.columns:
            in_sum = float(df_in[amount_column].sum())
            out_sum = float(df_out[amount_column].sum())
            diff = abs(in_sum - out_sum)
            status = "pass" if diff <= tolerance else "fail"
            if status == "fail":
                overall = "fail"
            checks.append({
                "check": "sum_conservation",
                "column": amount_column,
                "status": status,
                "input_sum": in_sum,
                "output_sum": out_sum,
                "diff": diff,
                "tolerance": tolerance,
            })

        # Check 2: Row count plausibility
        in_rows = len(df_in)
        out_rows = len(df_out)
        ratio = out_rows / max(in_rows, 1)
        row_status = "pass" if row_ratio_min <= ratio <= row_ratio_max else "fail"
        if row_status == "fail":
            overall = "fail"
        checks.append({
            "check": "row_count_plausible",
            "status": row_status,
            "input_rows": in_rows,
            "output_rows": out_rows,
            "ratio": round(ratio, 4),
        })

        # Check 3: Outlier detection
        try:
            import numpy as np
            outlier_flags: list[str] = []
            for col in df_out.select_dtypes(include=[float, int]).columns:
                mean = df_out[col].mean()
                std = df_out[col].std()
                if std > 0:
                    n_outliers = int(((df_out[col] - mean).abs() > std_threshold * std).sum())
                    if n_outliers > 0:
                        outlier_flags.append(f"{col}: {n_outliers} values > {std_threshold}σ")
            checks.append({
                "check": "outlier_detection",
                "status": "warn" if outlier_flags else "pass",
                "flags": outlier_flags,
            })
        except Exception as exc:
            checks.append({
                "check": "outlier_detection",
                "status": "skipped",
                "error": str(exc),
            })

        return {"contract_status": overall, "checks": checks}

    def _demux_logs(self, container):
        """Yield (stream_type, data) from Docker multiplexed log stream."""
        raw = container.logs(stdout=True, stderr=True, stream=False, demux=False)
        buf = io.BytesIO(raw)
        while True:
            header = buf.read(8)
            if len(header) < 8:
                break
            stream_type = header[0]
            length = int.from_bytes(header[4:8], "big")
            data = buf.read(length)
            yield stream_type, data

    async def execute_async(
        self,
        code: str,
        input_files: dict | None = None,
        timeout: int = 30,
    ) -> dict:
        """Async wrapper around execute() for non-blocking use in agents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.execute,
            code,
            input_files,
            timeout,
        )

    def close(self):
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton + convenience coroutine
# ---------------------------------------------------------------------------

_default_executor: SandboxExecutor | None = None


def get_executor() -> SandboxExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = SandboxExecutor()
    return _default_executor


async def execute_python(
    code: str,
    input_files: dict | None = None,
    timeout: int = 30,
) -> dict:
    """Convenience coroutine used by agent nodes."""
    return await get_executor().execute_async(code, input_files, timeout)
