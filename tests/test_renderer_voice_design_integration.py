"""Casting must honor descriptions and preserve one identity across character roles."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave

from audiobook_renderer.bundle import load_bundle
from audition_fixtures import prepare_bundle
from audiobook_renderer.casting import CastingSession
from audiobook_renderer.common import digest, read_json, write_json
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.voices import VoiceStore
from audiobook_renderer.voice_prompts import audiobook_voice_prompt, default_voice_prompt
from audiobook_renderer.workflow import RenderBlocked, ensure_voice, render
from epub_audiobook.cli import export_bundle


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def pcm_hash(path):
    with wave.open(str(path), "rb") as audio:
        return hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()


class ReferenceDesigner:
    identity = {"engine": "description_fixture", "model": "designed-v1"}
    capabilities = {"voice_design": True, "reference_voice": True}

    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.calls = []
        self.applied = True
        self.error = None
        self.closed = False

    def create_voice(self, prompt, text, path, *, seed):
        self.calls.append({"prompt": prompt, "text": text, "seed": seed})
        if self.error is not None:
            raise self.error
        self.write_audio(text, path, seed)
        return {"description_applied": self.applied}

    def write_audio(self, text, path, seed):
        rate = 44100
        frequency = 200 + seed % 3000
        samples = [round(4000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(rate)]
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack("<" + "h" * rate, *samples))
        self.transcripts[pcm_hash(path)] = text

    def close(self):
        self.closed = True


class ReferencePlaybackEngine(ReferenceDesigner):
    """The book engine can clone a reference but must never sample one randomly."""

    identity = {"engine": "drama_fixture", "model": "drama-reference-v1"}
    capabilities = {
        "reference_voice": True, "voice_design": False, "unconditioned_voice_creation": True,
    }

    def __init__(self, transcripts):
        super().__init__(transcripts)
        self.speech_calls = []

    def create_voice(self, prompt, text, path, *, seed):
        self.calls.append({"prompt": prompt, "text": text, "seed": seed})
        raise AssertionError("Random voice creation must not be used")

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.speech_calls.append({
            "text": text, "voice_id": voice["voice_id"],
            "asset_hash": voice["asset_hash"], "reference_audio": voice["reference_audio"],
        })
        self.write_audio(text, path, seed)
        return {}


class TranscriptVerifier:
    identity = {"backend": "fixture-asr", "version": 1}

    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.calls = []
        self.wrong = False

    def transcribe(self, path):
        self.calls.append(str(path))
        text = self.transcripts[pcm_hash(path)] if not self.wrong else "Unrelated speech."
        words = text.split()
        return {"text": text, "segments": [{
            "text": text, "start": 0.0, "end": 1.0,
            "avg_logprob": -0.1, "no_speech_prob": 0.001,
            "words": [{"word": word, "start": index / len(words),
                       "end": (index + 1) / len(words), "probability": 0.99}
                      for index, word in enumerate(words)],
        }]}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for voice-design integration tests")
class VoiceDesignIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = prepare_bundle(load_bundle(EXAMPLE / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voice_design"].update(
            type="qwen3_mlx", checkpoint=str(self.root / "design-model"))
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="shared-series")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe",
        )
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = ReferencePlaybackEngine(self.transcripts)
        self.designer = ReferenceDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.factory_calls = []
        self.voice_id = "voice_mara"
        self.alias_id = "voice_mara_narration"

    def new_session(self, *, bundle=None, config=None, designer_factory=None):
        def make_designer(settings):
            self.factory_calls.append(deepcopy(settings))
            return self.designer
        session = CastingSession(
            prepare_bundle(bundle or self.bundle), config or self.config,
            engine_factory=lambda _: self.engine,
            verifier_factory=lambda *args, **kwargs: self.verifier,
            designer_factory=designer_factory or make_designer, notify=lambda _: None,
        )
        self.addCleanup(session.close)
        return session

    def narration_bundle(self):
        document = read_json(EXAMPLE / "reviewed.json")
        # Put the alias first: startup must not generate it before discovering the
        # canonical character voice later in the cast.
        document["cast"]["voices"].insert(0, {
            "id": self.alias_id, "description": "Mara speaking with a measured narrative delivery.",
            "traits": [],
        })
        for role in document["cast"]["roles"]:
            if role["id"] == "narrator_main":
                role["voice_id"] = self.alias_id
        for segment in document["segments"]:
            if segment["role_id"] == "narrator_main":
                segment["voice_id"] = self.alias_id
        destination = self.root / "narration-bundle"
        export_bundle(document, destination)
        return prepare_bundle(load_bundle(destination))

    def test_configured_designer_receives_prompt_and_playback_engine_never_samples(self):
        session = self.new_session()
        prompt = "A young adult female voice, bright and clear, with a gentle cadence."
        text = "This is the voice I will use for the entire series."
        session.update_prompt(self.voice_id, prompt, text)
        selected = session.regenerate(self.voice_id)
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(self.designer.calls[0]["prompt"], audiobook_voice_prompt(prompt))
        self.assertEqual(self.designer.calls[0]["text"], text)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.factory_calls, [self.config["voice_design"]])
        self.assertTrue(selected["description_applied"])
        self.assertEqual(selected["created_with"], self.engine.identity)
        self.assertEqual(session.store.load(self.voice_id)["asset_hash"], selected["asset_hash"])

    def test_unconfigured_designer_blocks_both_casting_and_render_creation(self):
        config = deepcopy(self.config)
        config["voice_design"]["type"] = "engine"
        config["voice_design"]["checkpoint"] = None
        session = self.new_session(config=config)
        with self.assertRaises((ValueError, RuntimeError)):
            session.regenerate(self.voice_id)
        with self.assertRaises((ValueError, RuntimeError)):
            ensure_voice(session.store, self.voice_id, self.bundle["cast"]["voices"][0],
                         self.engine, self.verifier, config, config["render"]["work_dir"],
                         notify=lambda _: None)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.designer.calls, [])
        self.assertIsNone(session.store.load(self.voice_id))

    def test_casting_records_design_identity_without_changing_render_identity(self):
        identity = {**self.designer.identity, "scene_template_version": 7}
        self.designer.voice_design_identity = identity
        session = self.new_session()
        selected = session.regenerate(self.voice_id)
        self.assertEqual(selected["creation"]["designer"], identity)
        self.assertEqual(selected["created_with"], self.engine.identity)
        records = list((Path(self.config["render"]["work_dir"]) / "casting-candidates").glob("*/accepted.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(read_json(records[0])["request"]["designer"], identity)

    def test_complete_render_description_reaches_designer_without_trait_list(self):
        session = self.new_session()
        profile = deepcopy(self.bundle["cast"]["voices"][0])
        description = "A fictional creature with a harsh dry rasp, thin resonance, and precise consonants."
        profile.update(description=description, traits=[
            {"name": "texture", "value": "harsh dry rasp", "basis": "casting", "evidence": []},
            {"name": "resonance", "value": "thin resonance", "basis": "casting", "evidence": []},
        ])
        selected = ensure_voice(
            session.store, self.voice_id, profile, self.engine, self.verifier,
            self.config, self.config["render"]["work_dir"], designer=self.designer,
            notify=lambda _: None)
        expected = audiobook_voice_prompt(description)
        self.assertEqual(self.designer.calls[0]["prompt"], expected)
        self.assertEqual(selected["creation"]["requested_description"], expected)

    def test_missing_render_voice_uses_description_only_and_records_exact_prompt(self):
        session = self.new_session()
        profile = deepcopy(self.bundle["cast"]["voices"][0])
        profile.update(description="A reserved speaker.", traits=[
            {"name": "timbre", "value": "slightly husky",
             "basis": "casting", "evidence": [{"quote": "Do not speak this evidence."}]},
        ])
        expected = audiobook_voice_prompt("A reserved speaker.")
        selected = ensure_voice(
            session.store, self.voice_id, profile, self.engine, self.verifier,
            self.config, self.config["render"]["work_dir"], designer=self.designer,
            notify=lambda _: None)
        self.assertEqual(self.designer.calls[0]["prompt"], expected)
        self.assertEqual(selected["creation"]["requested_description"], expected)
        records = list((Path(self.config["render"]["work_dir"]) / "voice-candidates").glob("*/accepted.json"))
        self.assertEqual(len(records), 1)
        request = read_json(records[0])["request"]
        self.assertEqual(request["prompt"], expected)
        self.assertEqual(records[0].parent.name, digest(request))
        profile["traits"][0]["value"] = "a different suggestion in a later book"
        retained = ensure_voice(
            session.store, self.voice_id, profile, self.engine, self.verifier,
            self.config, self.config["render"]["work_dir"], designer=self.designer,
            notify=lambda _: None)
        self.assertEqual(retained["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_designer_failure_cannot_fall_back_to_random_casting(self):
        session = self.new_session()
        self.designer.error = RuntimeError("Designer failed to load its model")
        with self.assertRaises((ValueError, RuntimeError)):
            session.regenerate(self.voice_id)
        self.assertGreaterEqual(len(self.designer.calls), 1)
        self.assertEqual(self.engine.calls, [])
        self.assertIsNone(session.store.load(self.voice_id))
        self.assertEqual(session.store.list_candidates(self.voice_id), [])

    def test_description_not_applied_is_rejected_even_if_transcription_passes(self):
        session = self.new_session()
        for applied in (False, None, "true"):
            with self.subTest(description_applied=applied):
                self.designer.applied = applied
                with self.assertRaises((ValueError, RuntimeError)):
                    session.regenerate(self.voice_id)
                self.assertIsNone(session.store.load(self.voice_id))
                self.assertEqual(session.store.list_candidates(self.voice_id), [])
        self.assertEqual(self.engine.calls, [])

    def test_failed_replacement_preserves_selected_voice_and_immutable_audio(self):
        session = self.new_session()
        original = session.regenerate(self.voice_id)
        original_audio = Path(original["reference_audio"]).read_bytes()
        self.verifier.wrong = True
        with self.assertRaises(RenderBlocked):
            session.regenerate(self.voice_id)
        self.assertEqual(session.store.load(self.voice_id)["asset_hash"], original["asset_hash"])
        self.assertEqual(Path(original["reference_audio"]).read_bytes(), original_audio)
        self.assertEqual(len(session.store.list_candidates(self.voice_id)), 1)
        self.assertFalse((session.store.directory / ".lock").exists())

    def test_narration_and_dialogue_share_one_generated_asset(self):
        session = self.new_session(bundle=self.narration_bundle())
        session.generate_missing()
        rows = {row["voice_id"]: row for row in session.rows()}
        narration = rows[self.alias_id]
        dialogue = rows[self.voice_id]
        self.assertEqual(narration["canonical_voice_id"], self.voice_id)
        self.assertEqual(narration["selected"]["asset_hash"], dialogue["selected"]["asset_hash"])
        self.assertEqual(narration["selected"]["reference_audio"], dialogue["selected"]["reference_audio"])
        self.assertEqual(len(self.designer.calls), 2)  # Mara and Lena only.
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.designer.calls[0]["prompt"], default_voice_prompt(self.bundle["cast"]["voices"][0]))

    def test_narrator_prompt_and_regeneration_update_character_voice(self):
        session = self.new_session(bundle=self.narration_bundle())
        original = session.ensure_ready(self.voice_id)
        prompt = "A low, rough female voice, steady and assured."
        text = "I am still the same person when I tell the story."
        session.update_prompt(self.alias_id, prompt, text)
        rows = {row["voice_id"]: row for row in session.rows()}
        self.assertEqual(rows[self.voice_id]["prompt"], prompt)
        self.assertEqual(rows[self.voice_id]["reference_text"], text)
        replacement = session.regenerate(self.alias_id)
        self.assertNotEqual(replacement["asset_hash"], original["asset_hash"])
        for voice_id in (self.voice_id, self.alias_id):
            self.assertEqual(session.store.load(voice_id)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.designer.calls[-1]["prompt"], audiobook_voice_prompt(prompt))
        self.assertEqual(self.designer.calls[-1]["text"], text)
        self.assertTrue(Path(original["reference_audio"]).exists())

    def test_linkage_is_by_character_id_and_does_not_merge_matching_names(self):
        bundle = self.narration_bundle()
        bundle["cast"]["characters"][1]["name"] = "Mara"
        session = self.new_session(bundle=bundle)
        session.generate_missing()
        self.assertNotEqual(session.store.load(self.voice_id)["asset_hash"],
                            session.store.load("voice_lena")["asset_hash"])
        self.assertEqual(session.store.load(self.alias_id)["asset_hash"],
                         session.store.load(self.voice_id)["asset_hash"])
        self.assertEqual(len(self.designer.calls), 2)

    def test_parent_alias_survives_absent_role_but_shared_cache_is_not_book_policy(self):
        session = self.new_session(bundle=self.narration_bundle())
        selected = session.ensure_ready(self.alias_id)
        # Standalone legacy-store readers retain their persisted relationships.
        reopened = VoiceStore(self.config["voices"]["directory"], self.config["voices"]["series_id"])
        self.assertEqual(reopened.load(self.alias_id)["asset_hash"], selected["asset_hash"])
        # Later-book cast provides only the former narration identity, with no
        # original canonical profile/role. Its parent retains that identity
        # evidence; the child's own cache must not become the policy source.
        following = deepcopy(self.bundle)
        following["cast"] = {
            "voices": [{"id": self.alias_id, "description": "New book-specific wording.", "traits": []}],
            "characters": [],
            "roles": [{"id": "other_narrator", "kind": "narration",
                       "character_id": None, "voice_id": self.alias_id}],
        }
        no_parent = self.new_session(bundle=following)
        self.assertEqual(no_parent.store.canonical_id(self.alias_id), self.alias_id)
        self.assertIsNone(no_parent.store.load(self.alias_id))
        later = VoiceStore(self.root / "later-book" / "voices",
                           self.config["voices"]["series_id"], cast=following["cast"],
                           alias_loader=lambda: {}, parents=(session.store,))
        reused = later.load(self.alias_id)
        self.assertEqual(reused["asset_hash"], selected["asset_hash"])
        self.assertEqual(reused["reference_audio"], selected["reference_audio"])
        self.assertEqual(later.canonical_id(self.alias_id), self.voice_id)
        self.assertEqual(len(self.designer.calls), 1)
        self.assertFalse(later.directory.exists())

    def test_old_unselected_random_take_is_not_adopted_as_a_designed_voice(self):
        session = self.new_session()
        path = self.root / "old-random.wav"
        text = self.config["voices"]["reference_text"]
        self.designer.write_audio(text, path, seed=7)
        old = session.store.publish_candidate(
            self.voice_id, self.bundle["cast"]["voices"][0], path, text,
            self.engine.identity, {"passed": True},
            {"method": "casting_candidate", "description_applied": False},
            candidate_id="0123456789abcdef0123456789abcdef",
        )
        replacement = session.ensure_ready(self.voice_id)
        self.assertNotEqual(replacement["asset_hash"], old["asset_hash"])
        self.assertTrue(replacement["description_applied"])
        self.assertEqual(len(self.designer.calls), 1)
        self.assertTrue(Path(old["reference_audio"]).exists())

    def test_persisted_alias_cannot_silently_move_to_another_character(self):
        bundle = self.narration_bundle()
        session = self.new_session(bundle=bundle)
        selected = session.ensure_ready(self.alias_id)
        following = deepcopy(bundle)
        for role in following["cast"]["roles"]:
            if role["id"] == "narrator_main":
                role["character_id"] = "lena"
        with self.assertRaisesRegex(ValueError, "canonical identity|different character"):
            later = self.new_session(bundle=following)
            later.rows()
        self.assertEqual(session.store.load(self.alias_id)["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_alias_corruption_and_cycles_fail_before_voice_generation(self):
        session = self.new_session(bundle=self.narration_bundle())
        session.ensure_ready(self.alias_id)
        path = session.store.directory / "voice-aliases.json"
        record = read_json(path)
        record["aliases"][self.alias_id] = "voice_lena"
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "checksum"):
            session.rows()
        self.assertEqual(len(self.designer.calls), 1)
        with self.assertRaisesRegex(ValueError, "cycles"):
            VoiceStore(self.config["voices"]["directory"], "separate-series",
                       aliases={"a": "b", "b": "a"})

    def conflicting_later_bundle(self):
        document = read_json(EXAMPLE / "reviewed.json")
        document["cast"]["characters"][1]["voice_id"] = self.alias_id
        document["cast"]["voices"][1]["id"] = self.alias_id
        for role in document["cast"]["roles"]:
            if role["character_id"] == "lena":
                role["voice_id"] = self.alias_id
        for segment in document["segments"]:
            if segment["role_id"] == "lena_dialogue":
                segment["voice_id"] = self.alias_id
        destination = self.root / "conflicting-later-bundle"
        export_bundle(document, destination)
        return prepare_bundle(load_bundle(destination))

    def test_persisted_alias_cannot_merge_distinct_current_characters(self):
        original = self.new_session(bundle=self.narration_bundle())
        selected = original.ensure_ready(self.alias_id)
        later_bundle = self.conflicting_later_bundle()
        # The current cast has no shared voice IDs or inferred alias. The
        # conflict appears only after loading the earlier book's alias mapping.
        with self.assertRaises(ValueError):
            later = self.new_session(bundle=later_bundle)
            later.rows()
        self.assertEqual(original.store.load(self.voice_id)["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_alias_cannot_change_character_when_original_character_is_absent(self):
        original = self.new_session(bundle=self.narration_bundle())
        original.ensure_ready(self.alias_id)
        later_bundle = self.conflicting_later_bundle()
        later_bundle["cast"]["characters"] = [
            item for item in later_bundle["cast"]["characters"] if item["id"] == "lena"]
        later_bundle["cast"]["voices"] = [
            item for item in later_bundle["cast"]["voices"] if item["id"] == self.alias_id]
        later_bundle["cast"]["roles"] = [
            item for item in later_bundle["cast"]["roles"] if item["character_id"] == "lena"]
        with self.assertRaises(ValueError):
            later = self.new_session(bundle=later_bundle)
            later.ensure_ready(self.alias_id)
        self.assertEqual(len(self.designer.calls), 1)

    def test_explicit_shared_identity_is_remembered_without_later_config_override(self):
        self.book_aliases({"voice_lena": self.voice_id})
        original = self.new_session()
        selected = original.ensure_ready("voice_lena")
        self.assertEqual(original.store.load(self.voice_id)["asset_hash"], selected["asset_hash"])
        # The explicit mapping authorizes this pair of cast identities to share
        # one voice, and its provenance must survive a later process/book.
        later = self.new_session(
            designer_factory=lambda _: self.fail("An explicitly shared voice should be reused"))
        rows = {row["voice_id"]: row for row in later.rows()}
        self.assertEqual(rows["voice_lena"]["canonical_voice_id"], self.voice_id)
        self.assertEqual(later.ensure_ready("voice_lena")["asset_hash"], selected["asset_hash"])
        self.assertEqual(rows[self.voice_id]["selected"]["reference_audio"],
                         rows["voice_lena"]["selected"]["reference_audio"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_render_rejects_persisted_character_conflict_without_leaving_a_lock(self):
        original = self.new_session(bundle=self.narration_bundle())
        original.ensure_ready(self.alias_id)
        later_bundle = self.conflicting_later_bundle()
        with self.assertRaises(ValueError):
            render(later_bundle, self.config, self.engine, self.verifier, notify=lambda _: None)
        self.assertEqual(self.engine.speech_calls, [])
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.assertFalse((self.root / "output").exists())

    def as_legacy_alias_file(self, store):
        path = store.directory / "voice-aliases.json"
        payload = {
            "format": "audiobook-voice-aliases", "version": 1,
            "series_id": store.series_id, "aliases": read_json(path)["aliases"],
        }
        write_json(path, {**payload, "sha256": digest(payload)})
        return path

    def book_aliases(self, aliases):
        """Put mutable casting decisions in this test's own book directory."""
        destination = self.root / "alias-book"
        if not destination.exists():
            shutil.copytree(self.bundle["directory"], destination)
            self.bundle = prepare_bundle(load_bundle(destination))
        write_json(destination / "voice-aliases.json", {
            "format": "audiobook-book-voice-aliases", "version": 1,
            "series_id": self.config["voices"]["series_id"],
            "book_id": self.bundle["source"]["book"]["id"], "aliases": aliases,
        })

    def test_legacy_alias_migration_preserves_voice_and_records_current_ownership(self):
        bundle = self.narration_bundle()
        original = self.new_session(bundle=bundle)
        selected = original.ensure_ready(self.alias_id)
        path = self.as_legacy_alias_file(original.store)
        resumed = self.new_session(bundle=bundle)
        self.assertEqual(resumed.rows()[0]["selected"]["asset_hash"], selected["asset_hash"])
        self.assertEqual(read_json(path)["version"], 1)  # Browsing is read-only.
        self.assertEqual(resumed.ensure_ready(self.alias_id)["asset_hash"], selected["asset_hash"])
        migrated = read_json(path)
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["voice_owners"][self.alias_id], "mara")
        self.assertEqual(migrated["voice_owners"][self.voice_id], "mara")
        self.assertEqual(migrated["explicit_aliases"], {})
        self.assertEqual(len(self.designer.calls), 1)

    def test_legacy_cache_alias_does_not_merge_distinct_current_characters(self):
        original = self.new_session(bundle=self.narration_bundle())
        original.ensure_ready(self.alias_id)
        path = self.as_legacy_alias_file(original.store)
        later = self.new_session(bundle=self.conflicting_later_bundle())
        # A version-1 cache has no identity ownership evidence. Its stale edge
        # cannot authorize merging current characters or reuse the wrong audio.
        self.assertEqual(later.store.canonical_id(self.alias_id), self.alias_id)
        self.assertIsNone(later.store.load(self.alias_id))
        self.assertEqual(read_json(path)["version"], 1)
        self.assertEqual(len(self.designer.calls), 1)

    def test_legacy_explicit_alias_migration_retains_configured_exception(self):
        aliases = {"voice_lena": self.voice_id}
        self.book_aliases(aliases)
        original = self.new_session()
        selected = original.ensure_ready("voice_lena")
        path = self.as_legacy_alias_file(original.store)
        resumed = self.new_session()
        self.assertEqual(resumed.ensure_ready("voice_lena")["asset_hash"], selected["asset_hash"])
        migrated = read_json(path)
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["explicit_aliases"], aliases)
        following = self.new_session()
        self.assertEqual(following.ensure_ready("voice_lena")["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    @unittest.skipUnless(FFPROBE, "FFprobe is required for complete renderer validation")
    def test_render_uses_identical_reference_for_narration_and_dialogue(self):
        bundle = self.narration_bundle()
        session = self.new_session(bundle=bundle)
        session.generate_missing()
        generation_count = len(self.designer.calls)
        result = render(bundle, self.config, self.engine, self.verifier, notify=lambda _: None)
        self.assertEqual(result["voice_assets"][self.alias_id], result["voice_assets"][self.voice_id])
        self.assertEqual(len(self.designer.calls), generation_count)
        shared = result["voice_assets"][self.voice_id]
        calls = [call for call in self.engine.speech_calls if call["asset_hash"] == shared]
        self.assertGreater(len(calls), 2)
        self.assertEqual({call["voice_id"] for call in calls}, {self.voice_id})
        self.assertEqual(len({call["reference_audio"] for call in calls}), 1)
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
