"""No-network checks for durable downloads and paid-compute lifecycle calls."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.chapter_storage import S3ArtifactPublisher, stop_runpod
from epub_tts_narrator.renderer.common import file_hash


class ChapterStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sync.json"
        self.path.write_text('{"precision":"paragraph"}')
        self.settings = {"bucket": "fixture", "endpoint_url": "https://objects.example.invalid",
                         "region": "test-1", "prefix": "books", "download_expiry_seconds": 600}
        self.client = Mock()
        self.publisher = S3ArtifactPublisher(self.settings, client=self.client)

    def test_upload_is_addressed_by_hash_and_verified_before_download(self):
        sha = file_hash(self.path)
        self.client.head_object.return_value = {"ContentLength": self.path.stat().st_size,
                                                 "Metadata": {"sha256": sha}}
        record = self.publisher.publish("a" * 64, "sync.json", self.path, sha)
        self.assertIn(sha, record["key"])
        self.assertEqual(record["size"], self.path.stat().st_size)
        self.publisher.url(record)
        self.client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "fixture", "Key": record["key"]}, ExpiresIn=600)
        self.assertNotIn("access_key", record)

    def test_incomplete_upload_never_returns_an_artifact_record(self):
        self.client.head_object.return_value = {"ContentLength": 0, "Metadata": {}}
        with self.assertRaisesRegex(RuntimeError, "saved audio is retained"):
            self.publisher.publish("a" * 64, "sync.json", self.path, file_hash(self.path))
        self.assertTrue(self.path.is_file())

    def test_wav_cannot_be_uploaded(self):
        with self.assertRaisesRegex(ValueError, "compressed"):
            self.publisher.publish("a" * 64, "chapter.wav", self.path, file_hash(self.path))
        self.client.upload_file.assert_not_called()

    def test_storage_config_change_does_not_sign_wrong_object(self):
        with self.assertRaisesRegex(ValueError, "different storage"):
            self.publisher.url({"bucket": "other", "endpoint_url": self.settings["endpoint_url"], "region": "test-1"})

    def test_sdk_errors_do_not_leak_credentials(self):
        self.client.upload_file.side_effect = RuntimeError("private-credential-value")
        with self.assertRaises(RuntimeError) as raised:
            self.publisher.publish("a" * 64, "sync.json", self.path, file_hash(self.path))
        self.assertNotIn("private-credential-value", str(raised.exception))

    def test_publish_failure_identifies_sdk_type_and_provider_codes_only(self):
        class ClientError(Exception):
            pass
        failure = ClientError("raw text with fixture-ceph-secret embedded")
        failure.response = {"Error": {"Code": "AccessDenied", "Message": "raw text"},
                            "ResponseMetadata": {"HTTPStatusCode": 403}}
        self.client.upload_file.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self.publisher.publish("a" * 64, "sync.json", self.path, file_hash(self.path))
        message = str(raised.exception)
        self.assertIn("ClientError", message)
        self.assertIn("AccessDenied", message)
        self.assertIn("403", message)
        self.assertIn("saved audio is retained", message)
        self.assertNotIn("raw text", message)
        self.assertNotIn("fixture-ceph-secret", message)

    def test_url_failure_identifies_sdk_type_without_raw_text(self):
        self.client.generate_presigned_url.side_effect = KeyError("raw-key-material")
        record = {"key": "k", "bucket": "fixture", "endpoint_url": self.settings["endpoint_url"],
                  "region": "test-1"}
        with self.assertRaises(RuntimeError) as raised:
            self.publisher.url(record)
        self.assertIn("KeyError", str(raised.exception))
        self.assertNotIn("raw-key-material", str(raised.exception))

    def test_credentials_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            S3ArtifactPublisher(self.settings)

    def test_file_credentials_ignore_missing_ambient_profile(self):
        try:
            import boto3
        except ImportError:
            self.skipTest("optional boto3 dependency")
        with patch.dict(os.environ, {"AWS_PROFILE": "does-not-exist-fixture", "AWS_DEFAULT_REGION": "other-region"}):
            publisher = S3ArtifactPublisher(dict(self.settings, access_key_id="fixture-id", secret_access_key="fixture-secret"))
            self.assertEqual(publisher.client.meta.region_name, "test-1")
            self.assertEqual(publisher.client.meta.endpoint_url, self.settings["endpoint_url"])

    def test_stop_uses_actual_provider_lifecycle_not_process_exit(self):
        response = Mock(status=204)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch("epub_tts_narrator.renderer.chapter_storage.build_opener", return_value=opener):
            stop_runpod({"runpod_pod_id": "pod_fixture", "runpod_api_key": "fixture-key"})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://rest.runpod.io/v1/pods/pod_fixture/stop")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer fixture-key")

    def test_invalid_pod_id_never_sends_request(self):
        with patch("epub_tts_narrator.renderer.chapter_storage.build_opener") as opener:
            with self.assertRaises(ValueError):
                stop_runpod({"runpod_pod_id": "../../other", "runpod_api_key": "fixture"})
            opener.assert_not_called()

    def test_stop_failure_identifies_http_status_and_type(self):
        from urllib.error import HTTPError
        opener = Mock()
        opener.open.side_effect = HTTPError("https://rest.runpod.io/v1/pods/pod_fixture/stop",
                                            503, "unavailable", {}, None)
        with patch("epub_tts_narrator.renderer.chapter_storage.build_opener", return_value=opener):
            with self.assertRaises(RuntimeError) as raised:
                stop_runpod({"runpod_pod_id": "pod_fixture", "runpod_api_key": "fixture-key"})
        message = str(raised.exception)
        self.assertIn("HTTPError", message)
        self.assertIn("HTTP status 503", message)
        self.assertIn("may still be billed", message)
        self.assertNotIn("fixture-key", message)

    def test_stop_failure_identifies_network_error_type_without_raw_reason(self):
        from urllib.error import URLError
        opener = Mock()
        opener.open.side_effect = URLError("raw connection detail fixture-key")
        with patch("epub_tts_narrator.renderer.chapter_storage.build_opener", return_value=opener):
            with self.assertRaises(RuntimeError) as raised:
                stop_runpod({"runpod_pod_id": "pod_fixture", "runpod_api_key": "fixture-key"})
        message = str(raised.exception)
        self.assertIn("URLError", message)
        self.assertNotIn("raw connection detail", message)


if __name__ == "__main__":
    unittest.main()
