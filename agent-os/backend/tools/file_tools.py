"""
MinIO file-storage tools.

All agent nodes that need to persist or retrieve files should use the
FileTools class (or the module-level singleton returned by get_file_tools()).
"""

import logging
import os
from datetime import timedelta
from typing import Any

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket name constants
# ---------------------------------------------------------------------------

RAW_FILES = "raw-files"
PROCESSED_FILES = "processed-files"
REPORTS = "reports"

_ALL_BUCKETS = (RAW_FILES, PROCESSED_FILES, REPORTS)


# ---------------------------------------------------------------------------
# FileTools
# ---------------------------------------------------------------------------


class FileTools:
    """
    Thin wrapper around the MinIO client that provides the operations
    needed by agent nodes.

    Parameters
    ----------
    endpoint:
        MinIO server host:port (default: read from MINIO_ENDPOINT env var,
        falling back to "localhost:9000").
    access_key / secret_key:
        Credentials (default: MINIO_ACCESS_KEY / MINIO_SECRET_KEY env vars).
    secure:
        Use TLS.  Defaults to False for local/dev; set MINIO_SECURE=true
        in production.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
    ):
        self._endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        _access = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        _secret = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        _secure = secure if secure is not None else (
            os.getenv("MINIO_SECURE", "false").lower() == "true"
        )

        self.client = Minio(
            self._endpoint,
            access_key=_access,
            secret_key=_secret,
            secure=_secure,
        )

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create *bucket_name* if it does not already exist."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info("Created bucket: %s", bucket_name)
            else:
                logger.debug("Bucket already exists: %s", bucket_name)
        except S3Error as exc:
            logger.error("Failed to ensure bucket %r: %s", bucket_name, exc)
            raise

    def ensure_default_buckets(self) -> None:
        """Create all standard buckets (RAW_FILES, PROCESSED_FILES, REPORTS)."""
        for bucket in _ALL_BUCKETS:
            self.ensure_bucket(bucket)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        bucket: str,
        object_name: str,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload a local file to MinIO.

        Returns
        -------
        str
            The object URL, e.g. ``http://localhost:9000/raw-files/foo.csv``.
        """
        self.ensure_bucket(bucket)
        abs_path = os.path.abspath(local_path)
        file_size = os.path.getsize(abs_path)

        extra_headers = {}
        if metadata:
            # MinIO SDK accepts metadata as extra_headers with "x-amz-meta-" prefix
            for k, v in metadata.items():
                extra_headers[f"x-amz-meta-{k}"] = v

        try:
            self.client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=abs_path,
                content_type=content_type,
                metadata=extra_headers or None,
            )
            logger.info(
                "Uploaded %r -> %s/%s (%d bytes)",
                local_path, bucket, object_name, file_size,
            )
        except S3Error as exc:
            logger.error(
                "Upload failed for %r -> %s/%s: %s",
                local_path, bucket, object_name, exc,
            )
            raise

        scheme = "https" if self.client._base_url.is_https else "http"
        return f"{scheme}://{self._endpoint}/{bucket}/{object_name}"

    def upload_bytes(
        self,
        data: bytes,
        bucket: str,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes (e.g. an in-memory report) to MinIO."""
        import io as _io

        self.ensure_bucket(bucket)
        buf = _io.BytesIO(data)
        try:
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=buf,
                length=len(data),
                content_type=content_type,
            )
        except S3Error as exc:
            logger.error("upload_bytes failed for %s/%s: %s", bucket, object_name, exc)
            raise

        scheme = "https" if self.client._base_url.is_https else "http"
        return f"{scheme}://{self._endpoint}/{bucket}/{object_name}"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(self, bucket: str, object_name: str) -> bytes:
        """
        Download an object and return its content as bytes.

        Raises ``minio.error.S3Error`` if the object does not exist.
        """
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            logger.debug("Downloaded %s/%s (%d bytes)", bucket, object_name, len(data))
            return data
        except S3Error as exc:
            logger.error("Download failed for %s/%s: %s", bucket, object_name, exc)
            raise
        finally:
            try:
                response.close()
                response.release_conn()
            except Exception:
                pass

    def download_to_file(
        self, bucket: str, object_name: str, dest_path: str
    ) -> str:
        """Download an object to a local path. Returns *dest_path*."""
        try:
            self.client.fget_object(bucket, object_name, dest_path)
            logger.debug("Downloaded %s/%s -> %r", bucket, object_name, dest_path)
            return dest_path
        except S3Error as exc:
            logger.error(
                "download_to_file failed for %s/%s: %s", bucket, object_name, exc
            )
            raise

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_files(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List objects in *bucket* under *prefix*.

        Returns
        -------
        list of dicts with keys:
            name, size, last_modified, etag, is_dir
        """
        self.ensure_bucket(bucket)
        results: list[dict[str, Any]] = []
        try:
            objects = self.client.list_objects(
                bucket, prefix=prefix, recursive=recursive
            )
            for obj in objects:
                results.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "etag": obj.etag,
                        "is_dir": obj.is_dir,
                    }
                )
        except S3Error as exc:
            logger.error("list_files failed for %s/%s*: %s", bucket, prefix, exc)
            raise
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_file(self, bucket: str, object_name: str) -> None:
        """Delete a single object. No-op if it does not exist."""
        try:
            self.client.remove_object(bucket, object_name)
            logger.info("Deleted %s/%s", bucket, object_name)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                logger.warning(
                    "delete_file: object %s/%s does not exist", bucket, object_name
                )
                return
            logger.error("delete_file failed for %s/%s: %s", bucket, object_name, exc)
            raise

    def delete_files_with_prefix(self, bucket: str, prefix: str) -> int:
        """
        Delete all objects whose name starts with *prefix*.

        Returns the number of objects deleted.
        """
        objects = self.list_files(bucket, prefix=prefix)
        count = 0
        for obj in objects:
            if not obj["is_dir"]:
                self.delete_file(bucket, obj["name"])
                count += 1
        return count

    # ------------------------------------------------------------------
    # Pre-signed URLs
    # ------------------------------------------------------------------

    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: int = 3600,
        method: str = "GET",
    ) -> str:
        """
        Generate a pre-signed URL for *object_name*.

        Parameters
        ----------
        expires:
            Expiry in seconds (default 1 hour).
        method:
            HTTP method: "GET" (download) or "PUT" (upload).
        """
        try:
            expiry = timedelta(seconds=expires)
            if method.upper() == "PUT":
                url = self.client.presigned_put_object(
                    bucket, object_name, expires=expiry
                )
            else:
                url = self.client.presigned_get_object(
                    bucket, object_name, expires=expiry
                )
            logger.debug(
                "Presigned %s URL for %s/%s (expires %ds)",
                method, bucket, object_name, expires,
            )
            return url
        except S3Error as exc:
            logger.error(
                "get_presigned_url failed for %s/%s: %s", bucket, object_name, exc
            )
            raise

    # ------------------------------------------------------------------
    # Convenience: stat
    # ------------------------------------------------------------------

    def stat_file(self, bucket: str, object_name: str) -> dict[str, Any]:
        """
        Return metadata for a single object.

        Returns
        -------
        {name, size, last_modified, etag, content_type, metadata}
        """
        try:
            stat = self.client.stat_object(bucket, object_name)
            return {
                "name": stat.object_name,
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": dict(stat.metadata or {}),
            }
        except S3Error as exc:
            logger.error("stat_file failed for %s/%s: %s", bucket, object_name, exc)
            raise

    def object_exists(self, bucket: str, object_name: str) -> bool:
        """Return True if the object exists (stat it)."""
        try:
            self.stat_file(bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return False
            raise


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_file_tools: FileTools | None = None


def get_file_tools() -> FileTools:
    """Return the process-wide FileTools singleton."""
    global _default_file_tools
    if _default_file_tools is None:
        _default_file_tools = FileTools()
    return _default_file_tools
