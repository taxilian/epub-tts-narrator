"""Server credential creation preserves global settings and survives restarts."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import stat
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.server_config import ensure_server_api_key


class ServerCredentialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "settings.toml"

    def test_preserves_comments_and_unrelated_settings_in_supported_toml_forms(self):
        cases = (
            '# model settings\n[engine]\ncheckpoint = "./models"\n',
            '# header\n[server] # local server\nhost = "localhost" # bind\n',
            '[server]\napi_key = "" # fill me\nport = 8090\n',
            "['server']\n'api_key' = '' # fill me\n",
            '["ser\\u0076er"]\n"api_key" = ""\n',
            '[server]\napi_key = """"""\n',
            "[server]\napi_key = ''''''\n",
            '[server]\napi_key = """\n"""\n',
            '[server]\napi_key = """\\\n  \\\n  """\n',
            'server.host = "localhost"\n',
            'server.api_key = "" # root dotted setting\n',
            'server = { host = "localhost" } # inline table\n',
            'server = {}\n',
            'server = {api_key = "", port = 8088}\n',
            '[server]\r\napi_key = "" # windows\r\n',
            '[server]\r\nhost = "localhost"\r\n',
            '[server]',
            '[server] # no final newline',
            '[engine]\ncheckpoint = "./models"',
            '[server.extra]\nunknown = "preserve me"\n',
            '[server]\n# api_key = ""\n',
        )
        for text in cases:
            with self.subTest(text=text):
                self.path.write_bytes(text.encode())
                before = tomllib.loads(text)
                key, created = ensure_server_api_key(self.path)
                after = self.path.read_bytes().decode()
                expected = deepcopy(before)
                expected.setdefault("server", {})["api_key"] = key
                self.assertEqual(tomllib.loads(after), expected)
                self.assertTrue(created)
                self.assertRegex(key, r"^[A-Za-z0-9_-]{43}$")
                for comment in (line[line.index("#"):] for line in text.splitlines() if "#" in line):
                    self.assertIn(comment, after)
                if "\r\n" in text:
                    self.assertNotIn("\n", after.replace("\r\n", ""))
                if 'api_key = "" # fill me' in text:
                    self.assertEqual(after, text.replace('api_key = ""', f'api_key = "{key}"'))

    def test_does_not_change_empty_keys_or_table_lookalikes_elsewhere(self):
        text = '''# api_key = ""
[provider]
api_key = "" # unrelated credential
description = """
[server]
api_key = ""
"""
[engine]
checkpoint = "./models"
'''
        self.path.write_text(text)
        key, _ = ensure_server_api_key(self.path)
        after = self.path.read_text()
        self.assertTrue(after.startswith(text))
        self.assertEqual(tomllib.loads(after)["server"]["api_key"], key)

    def test_generated_key_is_private_and_reused_without_rewriting(self):
        self.path.write_text('[server]\napi_key=""\n')
        self.path.chmod(0o644)
        first, created = ensure_server_api_key(self.path)
        self.assertTrue(created)
        before = self.path.read_bytes()
        stamp = self.path.stat()
        with patch("epub_tts_narrator.renderer.server_config.secrets.token_urlsafe") as generate:
            second, created = ensure_server_api_key(self.path)
        self.assertEqual(first, second)
        self.assertFalse(created)
        generate.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_mtime_ns, stamp.st_mtime_ns)
        self.assertEqual(stat.S_IMODE(stamp.st_mode), 0o600)

    def test_existing_key_needs_no_write_permission_or_lock(self):
        text = '[server]\napi_key = "already-configured" # keep this\n'
        self.path.write_text(text)
        self.path.chmod(0o444)
        with patch("epub_tts_narrator.renderer.server_config.os.open") as open_file:
            self.assertEqual(ensure_server_api_key(self.path), ("already-configured", False))
        open_file.assert_not_called()
        self.assertEqual(self.path.read_text(), text)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o444)

    def test_link_updates_target_not_symlink(self):
        target = self.root / "real.toml"
        target.write_text('[server]\napi_key = ""\n')
        self.path.symlink_to(target.name)
        key, _ = ensure_server_api_key(self.path)
        self.assertTrue(self.path.is_symlink())
        self.assertEqual(tomllib.loads(target.read_text())["server"]["api_key"], key)

    def test_failed_atomic_replace_keeps_original_and_removes_private_temporary(self):
        text = '[server]\napi_key = ""\n'
        self.path.write_text(text)
        with (patch("epub_tts_narrator.renderer.server_config.os.replace",
                    side_effect=PermissionError("Cannot replace config")),
              self.assertRaisesRegex(PermissionError, "Cannot replace config")):
            ensure_server_api_key(self.path)
        self.assertEqual(self.path.read_text(), text)
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {"settings.toml", ".settings.toml.server-key.lock"})
        self.assertEqual((self.root / ".settings.toml.server-key.lock").read_bytes(), b"")

    def test_concurrent_startups_share_one_generated_key(self):
        self.path.write_text('[server]\napi_key = ""\n')
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(ensure_server_api_key, [self.path] * 8))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(len({key for key, _ in results}), 1)
        self.assertEqual(tomllib.loads(self.path.read_text())["server"]["api_key"], results[0][0])

    def test_invalid_config_is_unchanged_and_error_does_not_echo_credentials(self):
        for text in ('[server]\napi_key = "sensitive-but-unterminated',
                     '[server]\napi_key = "invalid key"\n', '[server]\napi_key = 13\n'):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError) as error:
                    ensure_server_api_key(self.path)
                self.assertNotIn("sensitive", str(error.exception))
                self.assertNotIn("invalid key", str(error.exception))
                self.assertEqual(self.path.read_text(), text)


if __name__ == "__main__":
    unittest.main()
