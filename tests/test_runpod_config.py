"""Deployment overlays preserve rendering intent without leaking unrelated keys."""

from copy import deepcopy
import os
from pathlib import Path
import stat
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.runpod_config import (
    build_client_config, build_remote_config, load_settings, toml_text, write_private_toml,
)


class RunpodConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.path = self.directory / "epub-audiobook.toml"
        self.base = {
            "engine": {"checkpoint": "models/local", "temperature": 0.8, "top_p": 0.85},
            "verification": {"model": "models/whisper", "reference_model": "models/second",
                             "recurrence_threshold": 4, "max_resolution_attempts": 10},
            "render": {"work_dir": "work", "output_dir": "audio", "epub": "input.epub",
                       "sentence_pause_ms": 315, "paragraph_pause_ms": 420,
                       "max_attempts": 5, "normalize_loudness": False},
            "chapter_server": {"runpod_api_key": "file-runpod-key"},
            "chapter_storage": {"bucket": "books", "endpoint_url": "https://storage.example.test",
                                "region": "default", "access_key_id": "ceph-key",
                                "secret_access_key": "ceph-secret"},
            "provider": {"model": "text-model", "api_key": "analysis-key",
                         "base_url": "https://analysis.example.test/v1"},
            "jev": {"enabled": True, "api_key": "jev-key", "threshold": 0.8},
            "gemini": {"api_key": "voice-design-key"},
            "elevenlabs": {"api_key": "other-voice-design-key"},
        }
        write_private_toml(self.path, self.base)

    def remote(self, settings=None):
        settings = settings or load_settings(self.path)
        return build_remote_config(settings["renderer_config"], settings["override_raw"],
                                   "pod123", "application-key", deployment=settings["deployment"])

    def test_overlay_changes_remote_only_and_preserves_base_file(self):
        before = self.path.read_bytes()
        override = {"render": {"sentence_pause_ms": 200},
                    "runpod": {"state_dir": "deployment", "max_hourly_price": 2.75}}
        write_private_toml(self.directory / "runpod.local.toml", override)
        settings = load_settings(self.path)
        remote = self.remote(settings)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(settings["renderer_config"]["render"]["sentence_pause_ms"], 315)
        self.assertEqual(remote["render"]["sentence_pause_ms"], 200)
        self.assertEqual(remote["render"]["paragraph_pause_ms"], 420)
        self.assertEqual(remote["render"]["max_attempts"], 5)
        self.assertFalse(remote["render"]["normalize_loudness"])
        self.assertEqual(settings["deployment"]["state_dir"], str(self.directory / "deployment"))
        self.assertEqual(settings["deployment"]["max_hourly_price"], 2.75)

    def test_remote_has_single_native_worker_and_cpu_verifier(self):
        remote = self.remote()
        worker, = remote["render_pool"]["workers"]
        self.assertEqual(worker["concurrency"], 1)
        self.assertEqual(worker["protocol"], "fish_native_drama3")
        self.assertEqual(worker["url"], "http://127.0.0.1:8080/v1/tts")
        self.assertEqual(remote["engine"]["checkpoint"], "/workspace/models/drama-3")
        self.assertEqual(remote["engine"]["temperature"], 0.8)
        self.assertEqual(remote["verification"]["type"], "faster_whisper")
        self.assertEqual(remote["verification"]["device"], "cpu")
        self.assertEqual(remote["verification"]["compute_type"], "int8")
        self.assertNotIn("max_inflight", remote["render_pool"])
        self.assertEqual(remote["verification"]["model"], "/workspace/models/whisper-ct2")
        self.assertEqual(remote["verification"]["reference_model"], "/workspace/models/whisper-ct2-reference")
        self.assertEqual(remote["verification"]["recurrence_threshold"], 4)
        self.assertEqual(remote["chapter_server"]["port"], 8082)

    def test_configured_reference_model_is_prepared_on_the_worker_and_absent_stays_unset(self):
        remote = self.remote()
        path = write_private_toml(self.directory / "worker.toml", remote)
        self.assertEqual(load_config(path)["verification"]["reference_model"],
                         "/workspace/models/whisper-ct2-reference")
        base = deepcopy(self.base)
        base["verification"].pop("reference_model")
        write_private_toml(self.path, base)
        remote = self.remote()
        self.assertIsNone(remote["verification"]["reference_model"])
        load_config(write_private_toml(self.directory / "worker.toml", remote))

    def test_batched_backend_uses_one_server_with_bounded_request_slots(self):
        write_private_toml(self.directory / "runpod.local.toml", {
            "runpod": {"tts_backend": "sglang", "tts_concurrency": 8}})
        remote = self.remote()
        worker, = remote["render_pool"]["workers"]
        self.assertEqual(worker["protocol"], "fish_sglang_drama3")
        self.assertEqual(worker["url"], "http://127.0.0.1:8080/v1/audio/speech")
        self.assertEqual(worker["concurrency"], 8)
        self.assertEqual(remote["render_pool"]["max_inflight"], 32)
        self.assertFalse(worker["multi_speaker"])
        self.assertFalse(worker["api_key"])
        self.assertEqual(remote["engine"]["checkpoint"], "/workspace/models/drama-3")
        loaded = load_config(write_private_toml(self.directory / "worker.toml", remote))
        self.assertEqual(loaded["render_pool"]["workers"][0]["concurrency"], 8)

    def test_remote_verification_device_and_compute_type_are_explicit_overrides(self):
        # The local machine's hardware settings never opt the remote into CUDA.
        base = deepcopy(self.base)
        base["verification"].update(device="cuda", compute_type="float32")
        write_private_toml(self.path, base)
        for override, expected in (
                ({}, ("cpu", "int8")),
                ({"device": "cuda"}, ("cuda", "int8_float16")),
                ({"device": "cuda", "compute_type": "float16"}, ("cuda", "float16")),
                ({"device": "cpu", "compute_type": "float32"}, ("cpu", "float32"))):
            with self.subTest(override=override):
                write_private_toml(self.directory / "runpod.local.toml", {"verification": override})
                remote = self.remote()
                settings = remote["verification"]
                self.assertEqual((settings["device"], settings["compute_type"]), expected)
                load_config(write_private_toml(self.directory / "worker.toml", remote))
        self.assertEqual(load_settings(self.path)["renderer_config"]["verification"]["compute_type"],
                         "float32")

    def test_invalid_remote_verification_is_rejected_before_upload(self):
        for override in ({"device": "mps"}, {"device": []}, {"device": True},
                         {"device": "cuda", "compute_type": "invented"}):
            with self.subTest(override=override):
                write_private_toml(self.directory / "runpod.local.toml", {"verification": override})
                with self.assertRaisesRegex(ValueError, "verification"):
                    self.remote()

    def test_invalid_batching_options_are_rejected_before_cloud_calls(self):
        for key, value in (("tts_backend", "other"), ("tts_backend", []),
                           ("tts_concurrency", True), ("tts_concurrency", 0),
                           ("tts_concurrency", 9), ("tts_concurrency", 1.5),
                           ("sglang_source", "/workspace/../etc/sglang")):
            with self.subTest(key=key, value=value):
                write_private_toml(self.directory / "runpod.local.toml", {"runpod": {key: value}})
                with self.assertRaisesRegex(ValueError, "runpod"):
                    load_settings(self.path)

    def test_deployment_service_port_controls_worker_without_changing_local_server(self):
        base = deepcopy(self.base)
        base["chapter_server"]["port"] = 8090
        base["runpod"] = {"service_port": 8091}
        write_private_toml(self.path, base)
        write_private_toml(self.directory / "runpod.local.toml", {
            "runpod": {"service_port": 8182}, "chapter_server": {"port": 8183}})
        settings = load_settings(self.path)
        self.assertEqual(settings["deployment"]["service_port"], 8182)
        self.assertEqual(settings["renderer_config"]["chapter_server"]["port"], 8090)
        self.assertEqual(self.remote(settings)["chapter_server"]["port"], 8182)

    def test_rejects_invalid_or_conflicting_service_ports(self):
        for value in (0, 65536, True, 8082.0, "8082", 22, 8080):
            with self.subTest(value=value):
                write_private_toml(self.directory / "runpod.local.toml", {"runpod": {"service_port": value}})
                with self.assertRaisesRegex(ValueError, "runpod.service_port"):
                    load_settings(self.path)

    def test_remote_copies_required_judges_and_ceph_but_no_voice_design_or_book_paths(self):
        remote = self.remote()
        self.assertEqual(remote["provider"]["api_key"], "analysis-key")
        self.assertEqual(remote["jev"]["api_key"], "jev-key")
        self.assertEqual(remote["chapter_storage"]["secret_access_key"], "ceph-secret")
        self.assertEqual(remote["chapter_server"]["runpod_api_key"], "file-runpod-key")
        self.assertEqual(remote["chapter_server"]["runpod_pod_id"], "pod123")
        self.assertEqual(remote["chapter_server"]["idle_action"], "runpod_stop")
        text = toml_text(remote)
        for excluded in ("voice-design-key", "other-voice-design-key", str(self.directory), "input.epub"):
            self.assertNotIn(excluded, text)
        for section in ("analysis", "extraction", "voice_design", "gemini", "elevenlabs", "voices"):
            self.assertNotIn(section, remote)

    def test_provider_omitted_only_when_both_recovery_features_disabled(self):
        settings = load_settings(self.path)
        settings["override_raw"] = {"verification": {
            "phonetic_fallback": "off", "recurring_resolution": "off"}}
        self.assertNotIn("provider", self.remote(settings))
        settings["override_raw"]["verification"]["recurring_resolution"] = "analysis"
        self.assertIn("provider", self.remote(settings))

    def test_worker_file_roundtrips_through_existing_renderer_validation(self):
        remote = self.remote()
        path = write_private_toml(self.directory / "worker.toml", remote)
        loaded = load_config(path)
        self.assertEqual(loaded["render_pool"]["workers"][0]["protocol"], "fish_native_drama3")
        self.assertEqual(loaded["render"]["sentence_pause_ms"], 315)
        self.assertEqual(loaded["chapter_server"]["api_key"], "application-key")

    def test_client_uses_remote_endpoint_without_editing_local_book_locations(self):
        settings = load_settings(self.path)
        local = deepcopy(settings["renderer_config"])
        client = build_client_config(local, "https://pod123-8082.proxy.runpod.net", "app-key")
        self.assertFalse(local["remote_chapters"]["url"])
        self.assertEqual(client["remote_chapters"]["url"], "https://pod123-8082.proxy.runpod.net")
        self.assertEqual(client["render"]["work_dir"], str(self.directory / "work"))
        self.assertEqual(client["render"]["output_dir"], str(self.directory / "audio"))

    def test_client_and_server_share_render_overrides_without_linux_paths_on_client(self):
        local = load_settings(self.path)["renderer_config"]
        override = {"render": {"sentence_pause_ms": 125, "max_paragraph_chars": 1400,
                                "chapter_numbers": [2], "chapter_format": "mp3",
                                "work_dir": "/workspace/work", "epub": "/workspace/book.epub",
                                "ffmpeg": "/workspace/bin/ffmpeg"}}
        client = build_client_config(local, "https://pod123-8082.proxy.runpod.net", "app-key", override)
        remote = build_remote_config(local, override, "pod123", "app-key")
        for key in ("sentence_pause_ms", "max_paragraph_chars", "chapter_format"):
            self.assertEqual(client["render"][key], remote["render"][key])
        self.assertEqual(client["render"]["chapter_numbers"], [2])
        self.assertEqual(client["render"]["chapter_format"], "m4a")
        for key in ("work_dir", "epub", "ffmpeg"):
            self.assertEqual(client["render"][key], local["render"][key])
        self.assertEqual(local["render"]["sentence_pause_ms"], 315)

    def test_config_path_origin_is_retained_with_deployment_section(self):
        base = deepcopy(self.base)
        base["runpod"] = {"state_dir": "state", "gpu_type_id": "NVIDIA H100 PCIe"}
        write_private_toml(self.path, base)
        other = self.directory / "overrides" / "worker.toml"
        write_private_toml(other, {"runpod": {"max_hourly_price": 2.50}})
        settings = load_settings(self.path, other)
        self.assertEqual(settings["deployment"]["state_dir"], str(self.directory / "state"))
        self.assertEqual(settings["renderer_config"]["engine"]["checkpoint"],
                         str(self.directory / "models/local"))
        self.assertEqual(settings["renderer_config"]["chapter_server"]["data_dir"],
                         str(self.directory / "chapter-server-data"))

    def test_missing_explicit_override_errors_but_default_is_optional(self):
        load_settings(self.path)
        with self.assertRaises(FileNotFoundError):
            load_settings(self.path, self.directory / "missing.toml")

    def test_no_environment_credential_fallback(self):
        base = deepcopy(self.base)
        base["chapter_server"]["runpod_api_key"] = ""
        write_private_toml(self.path, base)
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "ambient-secret"}):
            settings = load_settings(self.path)
        self.assertEqual(settings["runpod_api_key"], "")
        with self.assertRaisesRegex(ValueError, "Runpod idle shutdown requires"):
            self.remote(settings)

    def test_override_can_rotate_worker_credentials_without_changing_local_settings(self):
        write_private_toml(self.directory / "runpod.local.toml", {
            "chapter_server": {"runpod_api_key": "override-runpod-key"},
            "chapter_storage": {"secret_access_key": "override-ceph-key"}})
        settings = load_settings(self.path)
        self.assertEqual(settings["runpod_api_key"], "override-runpod-key")
        self.assertEqual(self.remote(settings)["chapter_storage"]["secret_access_key"], "override-ceph-key")
        self.assertEqual(settings["renderer_config"]["chapter_server"]["runpod_api_key"], "file-runpod-key")

    def test_rejects_unsafe_or_unknown_deployment_values_before_api_calls(self):
        for key, value in (("volume_size_gb", -1), ("max_hourly_price", float("inf")),
                           ("max_hourly_price", True), ("unknown", "value"),
                           ("remote_config", "/opt/config.toml"), ("data_dir", "/workspace/../tmp/data"),
                           ("git_ref", "--upload-pack=command"), ("ssh_user", "root; command"),
                           ("github_url", "https://credential@github.com/taxilian/repo.git")):
            with self.subTest(key=key, value=value):
                if isinstance(value, float) and not value < float("inf"):
                    (self.directory / "runpod.local.toml").write_text(f"[runpod]\n{key} = inf\n")
                else:
                    write_private_toml(self.directory / "runpod.local.toml", {"runpod": {key: value}})
                with self.assertRaises(ValueError):
                    load_settings(self.path)

    def test_rejects_unrelated_override_sections(self):
        write_private_toml(self.directory / "runpod.local.toml", {"gemini": {"api_key": "unneeded"}})
        with self.assertRaisesRegex(ValueError, "only deployment and remote rendering"):
            load_settings(self.path)

    def test_writer_handles_quotes_newlines_unicode_lists_and_empty_tables_privately(self):
        original = {"settings": {"value": 'quote " and backslash \\ and newline\n café',
                                 "nested": {"list": ["first", "second"], "empty": {}},
                                 "workers": [{"name": "one", "concurrency": 1}],
                                 "nothing": None}}
        path = self.directory / "private.toml"
        path.write_text("old")
        path.chmod(0o644)
        write_private_toml(path, original)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        parsed = tomllib.loads(path.read_text())
        expected = deepcopy(original)
        expected["settings"].pop("nothing")
        self.assertEqual(parsed, expected)

    def test_writer_does_not_follow_destination_symlink(self):
        original = self.directory / "original.toml"
        original.write_text("do not replace")
        destination = self.directory / "output.toml"
        destination.symlink_to(original)
        write_private_toml(destination, {"settings": {"enabled": True}})
        self.assertFalse(destination.is_symlink())
        self.assertEqual(original.read_text(), "do not replace")

    def test_invalid_native_sampling_fails_before_deployment(self):
        settings = load_settings(self.path)
        settings["override_raw"] = {"engine": {"top_k": 40}}
        with self.assertRaisesRegex(ValueError, "top_k must be 30"):
            self.remote(settings)


if __name__ == "__main__":
    unittest.main()
