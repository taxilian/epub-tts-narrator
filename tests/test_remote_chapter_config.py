"""Remote chapter mode uses one file and needs no local model configuration."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.config import load_config as analyzer_config
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.book_context import resolve_book_config


class RemoteChapterConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "global.toml"

    def test_remote_client_requires_neither_synthesis_nor_verification_model(self):
        self.path.write_text('[remote_chapters]\nurl="https://worker.example"\napi_key="file-key"\n')
        with patch.dict("os.environ", {"RUNPOD_API_KEY": "env-secret"}):
            config = load_config(self.path)
        self.assertIsNone(config["engine"]["checkpoint"])
        self.assertIsNone(config["verification"]["model"])
        self.assertEqual(config["remote_chapters"]["api_key"], "file-key")
        self.assertEqual(config["chapter_server"]["runpod_api_key"], "")
        analyzer_config(self.path)

    def test_invalid_urls_credentials_and_ranges_fail_without_echoing_values(self):
        for extra in ('url="https://user:secret@worker.example"', 'url="https://worker.example?key=secret"',
                      'timeout_seconds=0', 'poll_seconds=0', 'websocket="yes"', 'api_key="secret value"'):
            self.path.write_text('[engine]\ncheckpoint="./model"\n[verification]\nmodel="./asr"\n'
                                 + '[remote_chapters]\n' + extra + '\n')
            with self.subTest(extra=extra), self.assertRaises(ValueError) as caught:
                load_config(self.path)
            self.assertNotIn("secret", str(caught.exception))

    def test_idle_shutdown_requires_durable_data_and_download_storage(self):
        base = '[remote_chapters]\nurl="https://worker.example"\n[chapter_server]\nidle_action="runpod_stop"\n'
        self.path.write_text(base)
        with self.assertRaisesRegex(ValueError, "persistent_data"):
            load_config(self.path)
        self.path.write_text(base + 'persistent_data=true\nrunpod_pod_id="pod-fixture"\n'
                             + 'runpod_api_key="file-token"\n[chapter_storage]\nbucket="fixture-bucket"\n')
        config = load_config(self.path)
        self.assertEqual(config["chapter_server"]["idle_timeout_seconds"], 600)
        self.assertEqual(config["chapter_server"]["data_dir"], str(self.path.parent / "chapter-server-data"))

    def test_faster_whisper_cpu_settings_allowed_for_server(self):
        self.path.write_text('[engine]\ncheckpoint="./model"\n[verification]\n'
                             'model="./ct2-asr"\ntype="faster_whisper"\ndevice="cpu"\ncompute_type="int8"\ncpu_threads=8\n')
        config = load_config(self.path)
        self.assertEqual(config["verification"]["cpu_threads"], 8)
        self.assertEqual(config["verification"]["model"], str(self.path.parent / "ct2-asr"))

    def test_server_data_must_be_disjoint_from_models_and_global_writable_stores(self):
        base = ('[engine]\ncheckpoint="./model"\n'
                '[verification]\nmodel="./asr"\nreference_model="./secondary-asr"\n'
                '[voice_design]\ncheckpoint="./designer"\n'
                '[voices]\ndirectory="./voices"\n'
                '[render]\nwork_dir="./work"\noutput_dir="./output"\n')
        for other in ("model", "asr", "secondary-asr", "designer", "voices", "work", "output"):
            for server in (other, f"{other}/service"):
                self.path.write_text(base + f'[chapter_server]\ndata_dir="./{server}"\n')
                with self.subTest(server=server), self.assertRaisesRegex(ValueError, "chapter server data"):
                    load_config(self.path)
        self.path.write_text(base + '[chapter_server]\ndata_dir="."\n')
        with self.assertRaisesRegex(ValueError, "chapter server data"):
            load_config(self.path)
        self.assertEqual(sorted(path.name for path in self.path.parent.iterdir()), ["global.toml"])

    def test_server_data_cannot_contain_model_or_output_subdirectories(self):
        for section, key in (("engine", "checkpoint"), ("verification", "model"),
                             ("voices", "directory"), ("render", "work_dir"), ("render", "output_dir")):
            self.path.write_text('[remote_chapters]\nurl="https://worker.example"\n'
                                 '[chapter_server]\ndata_dir="./server"\n'
                                 f'[{section}]\n{key}="./server/nested"\n')
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "chapter server data"):
                load_config(self.path)

    def test_server_path_symlink_cannot_hide_global_output_overlap(self):
        output = self.path.parent / "output"
        output.mkdir()
        (self.path.parent / "server-link").symlink_to(output, target_is_directory=True)
        self.path.write_text('[remote_chapters]\nurl="https://worker.example"\n'
                             '[render]\noutput_dir="./output"\n'
                             '[chapter_server]\ndata_dir="./server-link"\n')
        with self.assertRaisesRegex(ValueError, "distinct"):
            load_config(self.path)
        self.assertEqual(list(output.iterdir()), [])

    def test_default_server_directory_preserves_legacy_global_store_defaults(self):
        self.path.write_text('[engine]\ncheckpoint="./model"\n[verification]\nmodel="./asr"\n'
                             '[voices]\nseries_id="legacy-series"\n')
        config = load_config(self.path)
        self.assertEqual(config["voices"]["directory"], str(self.path.parent / "voices"))
        self.assertEqual(config["render"]["work_dir"], str(self.path.parent / "render-work"))
        self.assertEqual(config["render"]["output_dir"], str(self.path.parent / "audiobook"))
        self.assertEqual(config["chapter_server"]["data_dir"], str(self.path.parent / "chapter-server-data"))
        analyzer_config(self.path)

    def test_per_book_paths_and_cli_overrides_cannot_replace_server_storage(self):
        self.path.write_text('[remote_chapters]\nurl="https://worker.example"\n'
                             '[chapter_server]\ndata_dir="./remote/server"\n')
        config = load_config(self.path)
        book = self.path.parent / "book"
        bundle = {"directory": str(book), "manifest": {}, "source": {"book": {"id": "fixture-book"}}}
        server = self.path.parent / "remote" / "server"
        for option in ("output", "work_dir"):
            for target in (server, server / "nested", server.parent):
                with self.subTest(option=option, target=target), self.assertRaisesRegex(ValueError, "chapter server data"):
                    resolve_book_config(bundle, config, **{option: target})
        for local_directory in ("voices", "render-work", "audiobook"):
            config["chapter_server"]["data_dir"] = str(book / local_directory)
            with self.subTest(local_directory=local_directory), self.assertRaisesRegex(ValueError, "chapter server data"):
                resolve_book_config(bundle, config)
        self.assertFalse(book.exists())
        self.assertFalse(server.exists())
