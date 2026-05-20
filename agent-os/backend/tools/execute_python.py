"""
Docker sandbox executor for safe Python code execution.
Runs code in isolated containers with resource limits and no network access.
"""

import docker
import asyncio
import json
import tempfile
import os
import time
import tarfile
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """
    Execute Python code in an isolated Docker container with strict security controls:
      - Network disabled
      - Memory capped at 512 MB
      - CPU capped at 1 vCPU
      - Read-only filesystem (except /tmp)
      - Non-root user inside the container
    """

    # Docker image that must be pre-built and available on the host
    DEFAULT_IMAGE = "agent-sandbox:latest"

    # Reasonable upper bound to prevent runaway containers
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_code_to_tar(self, code: str) -> bytes:
        """Pack the user code into a TAR archive so it can be copied into the container."""
        buf = io.BytesIO()
        info = tarfile.TarInfo(name="user_code.py")
        encoded = code.encode("utf-8")
        info.size = len(encoded)
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.addfile(info, io.BytesIO(encoded))
        buf.seek(0)
        return buf.read()

    def _build_volumes(self, input_files: dict[str, str] | None) -> dict:
        """
        Build the Docker volume-mount dict from {filename: host_path}.
        All input mounts are read-only.
        """
        volumes: dict = {}
        if not input_files:
            return volumes
        for filename, host_path in input_files.items():
            abs_path = os.path.abspath(host_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Input file not found: {abs_path!r}")
            # Mount under /sandbox/inputs/<filename>
            container_path = f"/sandbox/inputs/{os.path.basename(filename)}"
            volumes[abs_path] = {"bind": container_path, "mode": "ro"}
        return volumes

    def _collect_output_files(self, container) -> dict[str, bytes]:
        """
        Extract any files the code wrote to /sandbox/outputs inside the container.
        Returns {filename: bytes}.
        """
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
                            name = os.path.basename(member.name)
                            output_files[name] = f.read()
        except (docker.errors.NotFound, Exception):
            # /sandbox/outputs may not exist if the code didn't write anything
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
        """
        Execute Python code in an isolated Docker container.

        Parameters
        ----------
        code:
            The Python source code to run.
        input_files:
            Mapping of {container_filename: host_path}.  Each file is
            mounted read-only under /sandbox/inputs/.
        timeout:
            Wall-clock timeout in seconds (capped at MAX_TIMEOUT_SECONDS).

        Returns
        -------
        {
            "stdout":        str,
            "stderr":        str,
            "exit_code":     int,
            "duration_ms":   int,
            "output_files":  {filename: bytes},
            "timed_out":     bool,
        }
        """
        timeout = min(timeout, self.MAX_TIMEOUT_SECONDS)
        volumes = self._build_volumes(input_files)

        container = None
        start_ts = time.monotonic()
        timed_out = False

        try:
            # ----------------------------------------------------------
            # Create the container (do NOT start yet so we can copy code)
            # ----------------------------------------------------------
            container = self.client.containers.create(
                image=self.image,
                command=["python", "/sandbox/user_code.py"],
                network_disabled=True,
                mem_limit=self.mem_limit,
                nano_cpus=self.nano_cpus,
                volumes=volumes,
                working_dir="/sandbox",
                # Read-only root FS; /tmp and /sandbox/outputs are tmpfs
                read_only=True,
                tmpfs={
                    "/tmp": "size=64m",
                    "/sandbox/outputs": "size=256m",
                    "/sandbox/inputs": "size=256m",
                },
                user="nobody",
                # Prevent privilege escalation
                security_opt=["no-new-privileges:true"],
                # Drop ALL capabilities
                cap_drop=["ALL"],
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
            )

            # Copy the code archive into the container
            container.put_archive("/sandbox", self._write_code_to_tar(code))

            # ----------------------------------------------------------
            # Start and wait
            # ----------------------------------------------------------
            container.start()
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                # Timeout or Docker error — kill the container
                timed_out = True
                exit_code = -1
                try:
                    container.kill()
                except Exception:
                    pass

            # ----------------------------------------------------------
            # Collect logs
            # ----------------------------------------------------------
            raw_logs = container.logs(stdout=True, stderr=True, stream=False)
            # Docker multiplexes stdout/stderr; decode the stream
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            try:
                for frame_type, frame_data in self._demux_logs(container):
                    if frame_type == 1:
                        stdout_chunks.append(frame_data.decode("utf-8", errors="replace"))
                    else:
                        stderr_chunks.append(frame_data.decode("utf-8", errors="replace"))
            except Exception:
                # Fallback: treat all output as stdout
                stdout_chunks = [raw_logs.decode("utf-8", errors="replace")]

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

            # If TTY is disabled logs() returns multiplexed stream; some
            # Docker SDK versions return plain bytes — handle both.
            if not stdout and not stderr and raw_logs:
                stdout = raw_logs.decode("utf-8", errors="replace")

            output_files = self._collect_output_files(container)

        except docker.errors.ImageNotFound:
            logger.error("Sandbox image %r not found — build it first.", self.image)
            raise
        except Exception as exc:
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
        }

    def _demux_logs(self, container):
        """
        Yield (stream_type, data) tuples from a Docker multiplexed log stream.
        stream_type 1 = stdout, 2 = stderr.
        """
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
        """
        Async wrapper around :meth:`execute`.

        Runs the blocking Docker call in the thread pool so that the
        event loop is not blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.execute,
            code,
            input_files,
            timeout,
        )

    def close(self):
        """Shut down the thread pool gracefully."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised)
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
    """
    Convenience coroutine for use by agent nodes.

    Returns the same dict as :meth:`SandboxExecutor.execute`.
    """
    return await get_executor().execute_async(code, input_files, timeout)
