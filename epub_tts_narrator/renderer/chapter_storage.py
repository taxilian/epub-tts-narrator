"""Independent compressed artifact downloads and the shared storage key layout."""

from __future__ import annotations

from .chapter_protocol import ARTIFACT_NAMES, require_hash
from .common import file_hash
from .diagnostics import exception_detail


def key_for(settings, job_id, sha256, name):
    """Storage key layout shared by publish and external cleanup tools."""
    prefix = settings.get("prefix", "epub-tts-narrator").strip("/")
    return "/".join(part for part in (prefix, job_id, sha256, name) if part)


class S3ArtifactPublisher:
    """Publish immutable artifacts before a job becomes complete.

    Credentials and region are explicit file settings; no ambient AWS identity
    or default bucket is used. Download URLs can be renewed from persisted keys.
    """

    def __init__(self, settings, *, client=None):
        self.settings = dict(settings)
        self.bucket = settings.get("bucket", "")
        if not self.bucket:
            raise ValueError("chapter_storage.bucket is required")
        if client is None:
            if not settings.get("access_key_id") or not settings.get("secret_access_key") or not settings.get("region"):
                raise ValueError("Chapter storage needs explicit region and credentials in the TOML file")
            try:
                import boto3
                import botocore.session
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError("Object storage requires the chapter-server optional dependencies") from exc
            # Suppress shared config/credential files, including ambient AWS_PROFILE.
            session = botocore.session.Session(session_vars={
                "profile": (None, None, None, None),
                "config_file": (None, None, "/dev/null", None),
                "credentials_file": (None, None, "/dev/null", None),
            })
            session.set_credentials(settings["access_key_id"], settings["secret_access_key"])
            client = boto3.Session(botocore_session=session).client(
                "s3", region_name=settings["region"], endpoint_url=settings.get("endpoint_url") or None,
                verify=True,
                aws_access_key_id=settings["access_key_id"], aws_secret_access_key=settings["secret_access_key"],
                config=Config(signature_version="s3v4", connect_timeout=15, read_timeout=60,
                              retries={"max_attempts": 3, "mode": "standard"},
                              s3={"addressing_style": "path"}, proxies={},
                              ignore_configured_endpoint_urls=True,
                              use_dualstack_endpoint=False, use_fips_endpoint=False))
        self.client = client

    def publish(self, job_id, name, path, sha256):
        require_hash(job_id, "Job ID")
        require_hash(sha256, "Artifact hash")
        if name not in ARTIFACT_NAMES:
            raise ValueError("Only compressed chapter audio and JSON may be published")
        key = key_for(self.settings, job_id, sha256, name)
        record = {"key": key, "bucket": self.bucket, "endpoint_url": self.settings.get("endpoint_url", ""),
                  "region": self.settings.get("region", ""), "sha256": sha256, "size": path.stat().st_size}
        try:
            if file_hash(path) != sha256:
                raise ValueError("Artifact changed before upload")
            self.client.upload_file(str(path), self.bucket, key,
                                    ExtraArgs={"Metadata": {"sha256": sha256},
                                               "ContentType": "application/json" if name.endswith(".json")
                                               else "audio/flac" if name.endswith(".flac") else "audio/mp4"})
            stored = self.client.head_object(Bucket=self.bucket, Key=key)
            if (stored.get("ContentLength") != record["size"] or stored.get("Metadata", {}).get("sha256") != sha256
                    or file_hash(path) != sha256):
                raise ValueError("Artifact upload failed its size/hash metadata check")
        except Exception as exc:
            # SDK errors can include private endpoint or credential information;
            # report only the safe type and provider status/code, never raw text.
            raise RuntimeError(f"Unable to publish chapter artifacts ({exception_detail(exc)}); "
                               "saved audio is retained for retry") from None
        return record

    def url(self, record):
        if any(record.get(key) != self.settings.get(key, "") for key in ("bucket", "endpoint_url", "region")):
            raise ValueError("Saved chapter artifacts belong to a different storage configuration")
        try:
            return self.client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": record["key"]},
                ExpiresIn=self.settings.get("download_expiry_seconds", 3600))
        except Exception as exc:
            raise RuntimeError(f"Unable to create a chapter download URL ({exception_detail(exc)})") from None
