"""Global worker settings are bounded, portable and safe to diagnose."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.config import load_config as load_analyzer_config
from epub_tts_narrator.renderer.config import load_config


def worker_table(name="remote", kind="fish_http", **settings):
    values = {"name": name, "type": kind, **settings}
    if kind == "fish_http":
        values.setdefault("url", "https://render.example/v1/tts")
    return "[[render_pool.workers]]\n" + "".join(
        f"{key}={json.dumps(value)}\n" for key, value in values.items())


class DistributedConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "global.toml"

    def write(self, extra="", *, checkpoint=True, verifier=True):
        self.path.write_text(
            ('[engine]\ncheckpoint="./drama-3"\n' if checkpoint else "")
            + ('[verification]\nmodel="./whisper"\n' if verifier else "")
            + extra)
        return self.path

    def test_unconfigured_pool_preserves_local_defaults_and_ignores_environment(self):
        self.write()
        with patch.dict("os.environ", {"FISH_API_KEY": "environment-secret",
                                       "RENDERER_URL": "https://wrong.example/v1/tts"}):
            config = load_config(self.path)
        self.assertEqual(config["render_pool"]["workers"], [])
        self.assertEqual(config["engine"]["checkpoint"], str(self.root / "drama-3"))
        self.assertEqual(config["server"]["host"], "127.0.0.1")
        self.assertEqual(config["server"]["api_key"], "")
        self.assertNotIn("environment-secret", str(config))

    def test_local_and_remote_defaults_are_normalized_without_changing_file(self):
        self.write(worker_table("mac", "local")
                   + worker_table("cloud", api_key="file-secret", concurrency=3))
        before = self.path.read_bytes()
        workers = load_config(self.path)["render_pool"]["workers"]
        self.assertEqual(workers[0], {"name": "mac", "type": "local", "concurrency": 1})
        self.assertEqual(workers[1]["concurrency"], 3)
        self.assertEqual(workers[1]["api_key"], "file-secret")
        self.assertEqual(workers[1]["model"], "drama-3")
        self.assertEqual(workers[1]["timeout_seconds"], 300.0)
        self.assertIs(workers[1]["multi_speaker"], True)
        self.assertEqual(self.path.read_bytes(), before)
        # Both commands read this same file, without analyzer-specific copies.
        load_analyzer_config(self.path)

    def test_remote_only_does_not_require_local_checkpoint(self):
        self.write(worker_table(), checkpoint=False)
        self.assertIsNone(load_config(self.path)["engine"]["checkpoint"])
        self.write(worker_table("mac", "local") + worker_table(), checkpoint=False)
        with self.assertRaisesRegex(ValueError, "engine.checkpoint"):
            load_config(self.path)

    def test_server_can_load_engine_only_config_without_whisper(self):
        self.write('[server]\nhost="127.0.0.1"\nport=8091\napi_key="file-key"\n', verifier=False)
        config = load_config(self.path, require_verifier=False)
        self.assertIsNone(config["verification"]["model"])
        self.assertEqual(config["server"]["port"], 8091)
        with self.assertRaisesRegex(ValueError, "verification.model"):
            load_config(self.path)

    def test_remote_only_validates_sampling_before_loading_any_worker(self):
        for key, value in (("temperature", 0), ("top_p", 1.1), ("top_k", True),
                           ("max_new_tokens", 4096), ("experts", "unknown")):
            with self.subTest(key=key):
                self.write('[engine]\n' + f'{key}={json.dumps(value)}\n' + worker_table(),
                           checkpoint=False)
                with self.assertRaisesRegex(ValueError, f"engine.{key}"):
                    load_config(self.path)

    def test_malformed_worker_fields_fail_cleanly(self):
        invalid = (
            '[render_pool]\nworkers="remote"\n',
            '[render_pool]\nworkers=[1]\n',
            worker_table(kind=[]),
            worker_table(model=[]),
            worker_table(concurrency=True),
            worker_table(concurrency=0),
            worker_table(concurrency=17),
            worker_table(multi_speaker="true"),
            worker_table(timeout_seconds=0),
            worker_table(unknown="private-value"),
            worker_table("bad name"),
        )
        for value in invalid:
            with self.subTest(value=value):
                self.write(value)
                with self.assertRaises(ValueError) as caught:
                    load_config(self.path)
                self.assertNotIn("private-value", str(caught.exception))

    def test_urls_and_header_credentials_are_validated_without_echoing_secrets(self):
        secret = "credential-fixture"
        urls = (
            f"https://user:{secret}@render.example/v1/tts",
            f"https://render.example/v1/tts?key={secret}",
            f"https://render.example/v1/tts#{secret}",
            f"https://render.example:{secret}/v1/tts",
            "https://render.example:70000/v1/tts",
            "https://render.example", "file:///v1/tts",
            "https://render.example/v1/tts\nInjected: header",
        )
        for url in urls:
            with self.subTest(url=url):
                self.write(worker_table(url=url))
                with self.assertRaisesRegex(ValueError, r"\.url") as caught:
                    load_config(self.path)
                self.assertNotIn(secret, str(caught.exception))
        for value in (f"{secret}\nInjected: header", f"{secret} suffix", 123):
            with self.subTest(api_key=value):
                self.write(worker_table(api_key=value))
                with self.assertRaisesRegex(ValueError, "api_key") as caught:
                    load_config(self.path)
                self.assertNotIn(secret, str(caught.exception))

    def test_single_local_slot_and_unique_names_are_required(self):
        invalid = (
            worker_table("mac", "local", concurrency=2),
            worker_table("mac", "local") + worker_table("mac2", "local"),
            worker_table("same", "local") + worker_table("same"),
        )
        for value in invalid:
            with self.subTest(value=value):
                self.write(value)
                with self.assertRaisesRegex(ValueError, "local renderer|unique"):
                    load_config(self.path)

    def test_total_concurrency_is_bounded_at_32(self):
        at_limit = worker_table("one", concurrency=16) + worker_table("two", concurrency=16)
        self.write(at_limit)
        self.assertEqual(sum(worker["concurrency"] for worker in
                             load_config(self.path)["render_pool"]["workers"]), 32)
        self.write(at_limit + worker_table("extra", "local"))
        with self.assertRaisesRegex(ValueError, "at most 32"):
            load_config(self.path)

    def test_worker_count_and_server_resource_settings_are_bounded(self):
        self.write("".join(worker_table(f"worker{number}") for number in range(17)))
        with self.assertRaisesRegex(ValueError, "at most 16"):
            load_config(self.path)
        for setting in ('port=true', 'port=65536', 'max_pending_requests=0',
                        'max_request_bytes=1', 'request_timeout_seconds=nan',
                        'host="bad host"', 'api_key="secret key"', 'unknown="private-value"'):
            with self.subTest(setting=setting):
                self.write("[server]\n" + setting + "\n")
                with self.assertRaises(ValueError) as caught:
                    load_config(self.path)
                self.assertNotIn("secret key", str(caught.exception))
                self.assertNotIn("private-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
