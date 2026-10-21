"""Two real HTTP transports through verification, chapter encoding, and resume."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.server import make_server
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import assemble, render
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


class HTTPFixtureEngine(FixtureEngine):
    temperature = 1.0
    top_p = 0.9
    top_k = 30
    max_new_tokens = 2048

    def synthesize_fish(self, text, voices, path, *, seed):
        self.speech_calls.append({"text": text, "references": voices})
        # Test tones register a transcript for the test recognizer. Production
        # verification receives audio only, and never receives this fixture map.
        text = re.sub(r"<\|speaker:\d+\|> ?|</?whisper>", "", text)
        text = re.sub(r"\[(nervous|sighs|laughs)\]", "", text)
        return self._write(text, path, seed)


@unittest.skipUnless(FFMPEG and FFPROBE and importlib.util.find_spec("msgpack")
                     and importlib.util.find_spec("soundfile"), "requires renderer dependencies")
class HTTPWorkflowTests(unittest.TestCase):
    def test_two_http_workers_render_verify_package_and_resume_without_servers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            export_bundle(read_json(EXAMPLE / "reviewed.json"), root / "bundle")
            bundle = prepare_bundle(load_bundle(root / "bundle"))
            config = deepcopy(DEFAULTS)
            config["engine"]["checkpoint"] = None  # Coordinator has no TTS model.
            config["verification"]["model"] = str(root / "unused-asr")
            config["voices"].update(directory=str(root / "voices"), series_id="http-fixture")
            config["render"].update(work_dir=str(root / "work"), output_dir=str(root / "audio"),
                                    ffmpeg=FFMPEG, ffprobe=FFPROBE)
            config["quality"] = validate_policy({})
            fixtures, servers = {}, []
            engines = [HTTPFixtureEngine(fixtures), HTTPFixtureEngine(fixtures)]
            store = VoiceStore(config["voices"]["directory"], "http-fixture")
            for profile in bundle["cast"]["voices"]:
                reference = root / (profile["id"] + ".wav")
                engines[0]._write("Saved reference words.", reference, 1)
                store.publish(profile["id"], profile, reference, "Saved reference words.",
                              RenderPool.voice_binding_identity, {"passed": True}, {})
            reference_hashes = {voice["id"]: store.load(voice["id"])["asset_hash"]
                                for voice in bundle["cast"]["voices"]}
            try:
                for index, engine in enumerate(engines):
                    server_config = deepcopy(config)
                    server_config["server"].update(port=0, api_key="test-server-key")
                    try:
                        server = make_server(server_config, engine)
                    except PermissionError:
                        self.skipTest("sandbox does not permit loopback sockets")
                    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
                    thread.start()
                    servers.append((server, thread))
                    config["render_pool"]["workers"].append({
                        "name": f"mac-{index}", "type": "fish_http", "concurrency": 1,
                        "url": f"http://127.0.0.1:{server.server_port}/v1/tts",
                        "model": "drama-3", "api_key": "test-server-key", "timeout_seconds": 10,
                    })
                pool = RenderPool(config)
                verifier = FixtureVerifier(fixtures)
                with patch("epub_tts_narrator.renderer.pool.create_engine",
                           side_effect=AssertionError("coordinator must not load TTS")):
                    first = render(bundle, config, pool, verifier, chapters_only=True,
                                   notify=lambda _: None)
                    self.assertEqual(first["status"], "chapters_complete")
                    self.assertTrue(all(engine.speech_calls for engine in engines))
                    self.assertFalse(any(engine.voice_calls for engine in engines))
                    accepted = {str(path): path.read_bytes()
                                for path in (root / "work" / "clips").glob("*/accepted.json")}
                    self.assertEqual(len(accepted), 6)
                    for server, thread in servers:
                        server.shutdown()
                        thread.join()
                        server.server_close()
                    servers.clear()
                    # Even with both workers offline, completed work must reuse
                    # its accepted bytes and assemble without HTTP or ASR.
                    with patch.object(verifier, "transcribe", side_effect=AssertionError("reverified")):
                        resumed = render(bundle, config, pool, verifier, chapters_only=True,
                                         notify=lambda _: None)
                        self.assertEqual(first["chapters"], resumed["chapters"])
                        self.assertEqual(assemble(bundle, config, notify=lambda _: None)["status"], "complete")
                    self.assertEqual(accepted, {str(path): path.read_bytes()
                                               for path in (root / "work" / "clips").glob("*/accepted.json")})
                    self.assertTrue(list((root / "audio").glob("*.m4b")))
                    self.assertTrue((root / "audio" / "sync.json").is_file())
                    self.assertEqual(reference_hashes, {voice["id"]: store.load(voice["id"])["asset_hash"]
                                                      for voice in bundle["cast"]["voices"]})
                pool.close()
            finally:
                for server, thread in servers:
                    server.shutdown()
                    thread.join()
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
