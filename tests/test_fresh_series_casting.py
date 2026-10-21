"""A fresh analysis must reach casting without a separate prompt/sample repair.

The model and audio are fixtures: this tests workflow plumbing and continuity,
not whether a generative model follows acoustic directions perceptually.
"""

from contextlib import closing
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from audiobook_renderer.bundle import load_bundle
from audiobook_renderer.casting import CastingSession
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy
from epub_audiobook.cli import export_bundle
from epub_audiobook.contracts import validate_document
from epub_audiobook.pipeline import analyze
from epub_audiobook.series import load_registry, save_registry
from test_contracts import valid_document
from test_pipeline import FakeClient
from test_renderer_voice_design_integration import (
    FFMPEG, FFPROBE, ReferenceDesigner, TranscriptVerifier,
)


DESCRIPTION = (
    "An adult woman with warm chest resonance and a natural low speaking register. "
    "A single clearly audible conversational voice."
)
EVIDENCE_QUOTE = "her voice always had a rough edge"
SCRIPT = (
    "“We should take the lantern with us before we leave, because the narrow path "
    "along the water will be difficult to see after sunset. Stay close, and keep "
    "your hand on the rail until we reach the bridge.”"
)


class FreshSeriesClient(FakeClient):
    """Supply a valid profile, with its last stable trait added in reconciliation."""

    def discover(self, payload):
        patch = super().discover(payload)
        if not patch["characters"]:
            return patch
        character = patch["characters"][0]
        evidence_unit = next(unit for unit in payload["target_units"]
                             if EVIDENCE_QUOTE in unit["text"])
        profile = patch["voices"][0]
        profile.update(description=DESCRIPTION, traits=[
            {"name": "timbre", "value": "a slight rasp", "basis": "text",
             "evidence": [{"block_id": evidence_unit["block_id"], "quote": EVIDENCE_QUOTE}]},
            {"name": "pitch", "value": "low alto", "basis": "casting", "evidence": []},
            {"name": "articulation", "value": "crisp consonants", "basis": "casting", "evidence": []},
        ])
        # Deliberately retain a separate narration profile. The renderer must
        # resolve this identity relationship before generating either role.
        narrator = deepcopy(next(role for role in payload["known_cast"]["roles"]
                                 if role["id"] == "narrator_main"))
        narrator.update(character_id=character["id"], description="Mara narrates her own journey.")
        patch["roles"].insert(0, narrator)
        prior = payload.get("series_cast", {}).get("characters", [])
        if prior:
            character["series_identity"] = {
                "character_id": prior[0]["id"], "basis": "context", "review_required": False,
                "reason": "Mara is again identified as the lighthouse keeper.",
                "evidence": [{"block_id": evidence_unit["block_id"],
                              "quote": "Mara, the lighthouse keeper"}],
            }
            # This new proposal must not replace the established series voice.
            profile["description"] = "A different proposed voice for this later book."
        return patch

    def reconcile(self, payload):
        patch = super().reconcile(payload)
        character = patch["characters"][0]
        profile = deepcopy(next(voice for voice in payload["known_cast"]["voices"]
                                if voice["id"] == character["voice_id"]))
        profile["traits"].append({"name": "pace", "value": "steady conversational cadence",
                                  "basis": "casting", "evidence": []})
        patch["voices"] = [profile]
        return patch


@unittest.skipUnless(FFMPEG, "FFmpeg is required for the fresh series casting test")
class FreshSeriesCastingTests(unittest.TestCase):
    def test_normal_analysis_generates_one_shared_voice_and_sequel_reuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            registry_path = root / "series.json"
            source = deepcopy(valid_document()["source"])
            source["blocks"][0]["text"] = "Mara said softly, " + SCRIPT
            source["blocks"][1]["text"] = (
                "Mara, the lighthouse keeper, knew that " + EVIDENCE_QUOTE + "."
            )
            registry = load_registry(registry_path)
            self.assertEqual(registry["books"], [])
            first_client = FreshSeriesClient()
            first = analyze(source, first_client, root / "cache-1", series_registry=registry,
                            identity_consensus=False)
            self.assertEqual(validate_document(first), [])
            self.assertTrue(first["readiness"]["ready"])
            self.assertEqual([call[1].get("mode", "attribute") for call in first_client.calls],
                             ["discover", "reconcile", "attribute"])
            save_registry(registry_path, registry, first)
            registry = load_registry(registry_path)
            self.assertEqual(len(registry["books"]), 1)
            export_bundle(first, root / "bundle-1")
            first_bundle = load_bundle(root / "bundle-1")

            character = first_bundle["cast"]["characters"][0]
            voice_id = character["voice_id"]
            narrator = next(role for role in first_bundle["cast"]["roles"]
                            if role["id"] == "narrator_main")
            self.assertEqual(narrator["character_id"], character["id"])
            self.assertNotEqual(narrator["voice_id"], voice_id)
            profile = next(voice for voice in first_bundle["cast"]["voices"]
                           if voice["id"] == voice_id)
            self.assertEqual(profile["description"], DESCRIPTION)
            self.assertEqual(profile["traits"][-1]["name"], "pace")
            self.assertEqual(profile["traits"][0]["evidence"][0]["quote"], EVIDENCE_QUOTE)
            dialogue = next(segment for segment in first["segments"] if segment["kind"] == "dialogue")
            self.assertEqual(dialogue["delivery"]["direction"], "softly")

            config = deepcopy(DEFAULTS)
            config["engine"]["checkpoint"] = str(root / "drama-model")
            config["verification"]["model"] = str(root / "asr-model")
            config["voice_design"]["type"] = "engine"
            config["voices"].update(directory=str(root / "voices"), series_id=registry["series_id"])
            config["render"].update(work_dir=str(root / "work-1"), output_dir=str(root / "output-1"),
                                    ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
            config["quality"] = validate_policy({})
            transcripts = {}
            engine = ReferenceDesigner(transcripts)
            verifier = TranscriptVerifier(transcripts)
            with closing(CastingSession(first_bundle, config, engine_factory=lambda _: engine,
                                        verifier_factory=lambda *args, **kwargs: verifier,
                                        notify=lambda _: None)) as session:
                self.assertFalse(session.store.directory.exists())
                # No prepare_samples(), update_prompt(), manual aliases, or
                # explicit acceptance occurs anywhere in this workflow.
                generated = session.generate_missing()
                self.assertEqual(len(generated), 1)
                self.assertEqual(len(engine.calls), 1)
                self.assertEqual(len(verifier.calls), 1)
                call = engine.calls[0]
                self.assertIn(DESCRIPTION, call["prompt"])
                for trait in profile["traits"]:
                    self.assertIn(trait["value"], call["prompt"])
                self.assertNotIn(EVIDENCE_QUOTE, call["prompt"])
                self.assertNotIn("softly", call["prompt"])
                self.assertEqual(call["text"], SCRIPT)
                self.assertNotEqual(call["text"], config["voices"]["reference_text"])
                selected = session.store.load(voice_id)
                selected_bytes = Path(selected["reference_audio"]).read_bytes()
                self.assertEqual(selected["creation"]["sample_source"]["kind"], "book_excerpt")
                self.assertEqual(selected["creation"]["sample_source"]["book_id"], source["book"]["id"])
                self.assertEqual(session.store.load(narrator["voice_id"])["asset_hash"], selected["asset_hash"])
                self.assertTrue(all(row["selected"] is not None for row in session.rows()))

            sequel_source = deepcopy(source)
            sequel_source["book"].update(id="book2", title="The Keeper Returns")
            sequel_source["blocks"][0]["text"] = (
                "Mara said softly, “When we return to the lighthouse tomorrow, I will "
                "show you the old maps that explain how the ships found their way home.”"
            )
            sequel = analyze(sequel_source, FreshSeriesClient(), root / "cache-2",
                             series_registry=registry, identity_consensus=False)
            self.assertEqual(validate_document(sequel), [])
            self.assertTrue(sequel["readiness"]["ready"])
            self.assertEqual(sequel["cast"]["characters"][0]["id"], character["id"])
            self.assertEqual(sequel["cast"]["characters"][0]["voice_id"], voice_id)
            inherited = next(voice for voice in sequel["cast"]["voices"] if voice["id"] == voice_id)
            self.assertEqual(inherited["description"], DESCRIPTION)
            self.assertEqual([(trait["name"], trait["value"]) for trait in inherited["traits"]],
                             [(trait["name"], trait["value"]) for trait in profile["traits"]])
            self.assertTrue(all(not trait["evidence"] for trait in inherited["traits"]))
            save_registry(registry_path, registry, sequel)
            self.assertEqual(len(load_registry(registry_path)["books"]), 2)
            export_bundle(sequel, root / "bundle-2")
            sequel_bundle = load_bundle(root / "bundle-2")
            config["render"].update(work_dir=str(root / "work-2"), output_dir=str(root / "output-2"))

            def no_model(*args, **kwargs):
                self.fail("A later book must reuse the selected reference without loading models")

            with closing(CastingSession(sequel_bundle, config, engine_factory=no_model,
                                        verifier_factory=no_model, designer_factory=no_model,
                                        notify=lambda _: None)) as session:
                self.assertEqual(session.generate_missing(), [])
                for row in session.rows():
                    self.assertEqual(row["canonical_voice_id"], voice_id)
                    self.assertEqual(row["selected"]["asset_hash"], selected["asset_hash"])
                    self.assertEqual(row["reference_text"], SCRIPT)
                self.assertEqual(Path(session.store.load(voice_id)["reference_audio"]).read_bytes(), selected_bytes)
            self.assertEqual(len(engine.calls), 1)
            self.assertEqual(len(verifier.calls), 1)


if __name__ == "__main__":
    unittest.main()
