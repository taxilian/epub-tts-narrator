"""Heard variants: learning how the recognizer hears unusual words, fail-closed.

Recognizers, synthesis and model boundaries are fakes; the variant store,
alignment, plausibility rules, real ``validate_audio`` and ``verified_take`` stay
real. Fixture words prove mechanics, not live recognizer behavior.
"""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer import heard_variants
from epub_tts_narrator.renderer.heard_variants import FILENAME, apply, effective_variants, recognizer_key
from epub_tts_narrator.renderer.quality import compare_words
from epub_tts_narrator.renderer.retry_budget import SpeechRetryBudget
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, verified_take
from test_renderer_reference_consensus import transcript as timed
from test_renderer_workflow import write_tone


def audio(n):
    return f"{n:064x}"


def failed(expected, heard, **extra):
    return {"passed": False, "purpose": "speech", "reasons": ["text_mismatch"], "expected_text": expected,
            "transcript": heard, "metrics": compare_words(expected, heard), "words": [], **extra}


class Recognizer:
    def __init__(self, backend="whisper-fixture", heard=()):
        self.identity = {"backend": backend, "model_path": "/machine/local/path"}
        self.heard = list(heard)
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        value = self.heard.pop(0)
        if isinstance(value, BaseException):
            raise value
        return timed(value)


def make_book(directory, book_id="book-one"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.json").write_text(json.dumps({"book": {"id": book_id}}))
    return directory


class HeardVariantRuleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.book = make_book(Path(temporary.name) / "book")
        self.verifier = Recognizer()

    def take(self, expected, heard, number, verifier=None, **extra):
        return apply(failed(expected, heard, **extra), directory=self.book, verifier=verifier or self.verifier,
                     request_hash=audio(1000 + number), audio_sha256=audio(number))

    def test_a_variant_heard_in_three_independent_takes_is_accepted_and_audited(self):
        report, outcome = self.take("Kvothe walked home.", "Quote walked home.", 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["heard_variant_observations"], [{"expected": ["kvothe"], "heard": ["quote"], "takes": 1}])
        # Re-verifying the same audio (a resumed check) is not another take.
        report, _ = self.take("Kvothe walked home.", "Quote walked home.", 1)
        self.assertEqual(report["heard_variant_observations"][0]["takes"], 1)
        report, _ = self.take("Kvothe sat down.", "Quote sat down.", 2)
        self.assertFalse(report["passed"])
        self.assertEqual(report["heard_variant_observations"][0]["takes"], 2)
        report, outcome = self.take("Then Kvothe laughed.", "Then quote laughed.", 3)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["previous_reasons"], ["text_mismatch"])
        self.assertEqual(report["text_confirmation_method"], "heard_variant")
        self.assertGreater(report["metrics"]["wer"], 0)  # Never presented as a verbatim match.
        row = report["heard_variants"]["variants"][0]
        self.assertEqual((row["expected"], row["heard"], row["basis"], row["takes"]),
                         (["kvothe"], ["quote"], "repeated", 3))
        self.assertEqual({item["audio_sha256"] for item in row["sightings"]}, {audio(1), audio(2), audio(3)})
        self.assertEqual(report["heard_variants"]["recognizer"], recognizer_key(self.verifier))
        json.dumps(report, allow_nan=False)
        # A fourth take with the trusted variant is accepted as well.
        report, _ = self.take("Kvothe's lute.", "Quote's lute.", 4)
        self.assertFalse(report["passed"])  # The possessive is its own word, learned separately.
        report, _ = self.take("Kvothe ran.", "Quote ran.", 5)
        self.assertTrue(report["passed"])

    def test_unrelated_or_unsafe_differences_never_learn_or_accept(self):
        cases = {
            "implausible": ("Kvothe walked home.", "Banana walked home.", {}),
            "missing word": ("Kvothe walked home.", "Kvothe walked.", {}),
            "extra word": ("Kvothe walked home.", "Kvothe walked home again.", {}),
            "missing name": ("Kvothe walked home.", "Walked home.", {}),
            "number": ("Kvothe walked seven miles.", "Kvothe walked heaven miles.", {}),
            "negation": ("Kvothe did not go.", "Kvothe did knot go.", {}),
            "function word": ("Kvothe saw the inn.", "Kvothe saw a inn.", {}),
            "too many differences": ("Kvothe met Denna at Imre.", "Quote met dinner at emery.", {}),
            "acoustic failure": ("Kvothe walked home.", "Quote walked home.",
                                 {"reasons": ["text_mismatch", "audio_too_quiet"]}),
            "history echo": ("Kvothe walked home.", "Quote walked home.", {"history_echo": {"turn": 1}}),
            "pending confirmation": ("Kvothe walked home.", "Quote walked home.", {"resolution_pending": True}),
        }
        for label, (expected, heard, extra) in cases.items():
            with self.subTest(label):
                for number in range(1, 5):
                    report, outcome = self.take(expected, heard, number, **extra)
                    self.assertFalse(report["passed"])
                    self.assertIsNone(outcome)
                    self.assertNotIn("heard_variant_observations", report)
                self.assertFalse((self.book / FILENAME).exists())

    def test_a_trusted_variant_never_excuses_another_missing_word_word_or_recognizer(self):
        for number in range(1, 4):
            self.take("Kvothe walked home.", "Quote walked home.", number)
        report, outcome = self.take("Kvothe walked home.", "Quote walked.", 10)
        self.assertFalse(report["passed"])
        self.assertIsNone(outcome)
        report, _ = self.take("Denna walked home.", "Quote walked home.", 11)
        self.assertFalse(report["passed"])  # Bound to its own word.
        other = Recognizer(backend="another-recognizer")
        report, _ = self.take("Kvothe walked home.", "Quote walked home.", 12, verifier=other)
        self.assertFalse(report["passed"])  # Bound to the recognizer that heard it.
        self.assertEqual(report["heard_variant_observations"][0]["takes"], 1)
        # The machine-local model path is not part of the recognizer identity.
        moved = Recognizer()
        moved.identity["model_path"] = "/another/machine"
        report, _ = self.take("Kvothe walked home.", "Quote walked home.", 13, verifier=moved)
        self.assertTrue(report["passed"])

    def test_a_sound_review_approval_trusts_a_variant_at_once(self):
        expected, heard = "Aelith smiled at them.", "Alleys smiled at them."
        reviewed = {"passed": True, "reasons": [], "purpose": "speech", "expected_text": expected,
                    "transcript": heard, "verified_transcript": heard, "text_confirmation_method": "phonetic_primary"}
        report, outcome = apply(reviewed, directory=self.book, verifier=self.verifier,
                                request_hash=audio(9), audio_sha256=audio(9))
        self.assertIs(report, reviewed)
        self.assertEqual(outcome["source"], "review")
        report, _ = self.take("Aelith left.", "Alleys left.", 1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["heard_variants"]["variants"][0]["basis"], "review")

    def test_another_books_record_or_a_corrupt_file_is_refused(self):
        (self.book / FILENAME).write_text(json.dumps({"format": heard_variants.FORMAT, "version": 1,
                                                       "book_id": "someone-else", "recognizers": {}}))
        with self.assertRaisesRegex(ValueError, "different book"):
            self.take("Kvothe walked home.", "Quote walked home.", 1)
        (self.book / FILENAME).write_text("{not json")
        with self.assertRaises(ValueError):
            effective_variants(self.book, recognizer_key(self.verifier))


class HeardVariantInheritanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_a_continued_book_inherits_variants_but_writes_only_its_own(self):
        from test_name_pronunciations import SeriesInheritanceTests
        first = SeriesInheritanceTests.book(self, "first")
        second = SeriesInheritanceTests.book(self, "second", previous=first)
        verifier = Recognizer()
        for number in range(1, 3):
            apply(failed("Kvothe walked home.", "Quote walked home."), directory=first, verifier=verifier,
                  request_hash=None, audio_sha256=audio(number))
        before = (first / FILENAME).read_bytes()
        report, _ = apply(failed("Kvothe ran.", "Quote ran."), directory=second, verifier=verifier,
                          request_hash=None, audio_sha256=audio(3))
        self.assertTrue(report["passed"], report)  # Two sightings in book one plus one here.
        self.assertEqual((first / FILENAME).read_bytes(), before)
        local = json.loads((second / FILENAME).read_text())
        [variant] = [v for words in local["recognizers"].values() for v in words["kvothe"].values()]
        self.assertEqual([row["audio_sha256"] for row in variant["sightings"]], [audio(3)])
        # The earlier book never sees the later book's sighting.
        report, _ = apply(failed("Kvothe ran.", "Quote ran."), directory=first, verifier=verifier,
                          request_hash=None, audio_sha256=audio(4))
        self.assertTrue(report["passed"])  # Its own third take.
        self.assertEqual(len(effective_variants(first, recognizer_key(verifier))[("kvothe", "quote")]["sightings"]), 3)


class VerifiedTakeHeardVariantTests(unittest.TestCase):
    TEXT = "Kvothe walked home."

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.book = make_book(self.root / "book")
        self.config = {"quality": {}, "verification": {}, "_heard_variants_directory": str(self.book),
                       "render": {"max_attempts": 2, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
                                  "keep_intermediate_audio": True}}
        self.messages = []
        patcher = patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                        side_effect=lambda raw, wav, **_: Path(wav).write_bytes(Path(raw).read_bytes()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def generate(self, path, seed):
        # Distinct takes are distinct audio, as independent generations are.
        write_tone(path, frequency=200 + seed % 500)
        return {"seed": seed}

    def run_take(self, verifier, request):
        return verified_take(self.TEXT, self.generate, self.root / "clips" / "kvothe", request, verifier,
                             self.config, notify=self.messages.append, retry_budget=SpeechRetryBudget())

    def test_learning_survives_an_interrupted_run_and_the_third_take_is_accepted(self):
        request = {"kind": "speech", "chunk": {"chunk_id": "c1"}, "fixture": "heard-variants"}
        heard = "Quote walked home."
        verifier = Recognizer(heard=[heard, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_take(verifier, request)  # Take 2 is saved but its recognition was interrupted.
        sightings = effective_variants(self.book, recognizer_key(verifier))[("kvothe", "quote")]["sightings"]
        self.assertEqual(len(sightings), 1)
        verifier.heard = [heard, heard]
        wav, accepted = self.run_take(verifier, request)
        self.assertEqual(verifier.calls, 4)  # The saved take 2 is recognized, then take 3 is generated.
        report = accepted["report"]
        self.assertTrue(report["passed"])
        self.assertEqual(report["text_confirmation_method"], "heard_variant")
        self.assertEqual(report["heard_variants"]["variants"][0]["takes"], 3)
        self.assertEqual(accepted["attempt"], 3)
        self.assertTrue(any("not as a verbatim match" in line for line in self.messages), self.messages)
        self.assertTrue(any("heard this way in 1 of 3 takes" in line for line in self.messages), self.messages)
        saved = json.loads((Path(wav).parent / "accepted.json").read_text())
        self.assertEqual(saved["report"]["text_confirmation_method"], "heard_variant")
        # A rerun reuses the accepted bytes without recognizing them again.
        again, _ = self.run_take(verifier, request)
        self.assertEqual(Path(again), Path(wav))
        self.assertEqual(verifier.calls, 4)

    def test_turning_learning_off_keeps_ordinary_verification(self):
        self.config["_heard_variants_directory"] = None
        verifier = Recognizer(heard=["Quote walked home."] * 6)
        request = {"kind": "speech", "chunk": {"chunk_id": "c1"}, "fixture": "off"}
        for _ in range(3):
            with self.assertRaises(SpeechVerificationFailed):
                self.run_take(verifier, request)
        self.assertFalse((self.book / FILENAME).exists())


class PronounceNameHeardVariantTests(unittest.TestCase):
    def setUp(self):
        from test_name_pronunciations import InteractiveToolTests
        from test_renderer_pronunciation_review import HAS_FFMPEG
        if not HAS_FFMPEG:
            self.skipTest("ffmpeg/ffprobe required")
        InteractiveToolTests.setUp(self)

    def run_tool(self, commands, heard):
        from epub_tts_narrator.pipeline import CachedRequests
        from epub_tts_narrator.renderer.name_pronunciation_tool import run_pronounce_name
        from test_name_pronunciations import FakeModel
        answers = iter(commands)
        self.recognizer = Recognizer(heard=[heard])
        args = SimpleNamespace(name="Mara", sound=None, phonemes="M AA1 R AH0", voice_id=None, tag_only=False,
                               review_tags=False, list=False)
        model = FakeModel(decide=lambda item: "name")
        return run_pronounce_name(
            self.bundle, self.config, args, analysis_config=lambda: {"analysis": {"attribution_workers": 1}},
            input_fn=lambda prompt: next(answers), output=self.output.append, engine_factory=self.factory,
            store=self.store, player=self.player, playback_fn=lambda player, path: True,
            requests=CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=0),
            interactive=False, verifier_factory=lambda: self.recognizer)

    def test_an_accepted_test_sentence_teaches_how_the_recognizer_hears_the_name(self):
        self.run_tool(["A"], "The name is Marla. It is Marla’s name. Give it to Marla.")
        self.assertEqual(self.recognizer.calls, 1)
        learned = effective_variants(self.directory, recognizer_key(self.recognizer))
        self.assertEqual(set(learned), {("mara", "marla"), ("mara's", "marla's")})
        self.assertTrue(all(heard_variants.trust_basis(variant) == "listened" for variant in learned.values()))
        self.assertTrue(any("hears Mara as" in line for line in self.output), self.output)
        # The first render take heard that way is accepted.
        report, _ = apply(failed("Mara lit the lamp.", "Marla lit the lamp."), directory=self.directory,
                          verifier=self.recognizer, request_hash=None, audio_sha256=audio(1))
        self.assertTrue(report["passed"])
        self.assertEqual(report["heard_variants"]["variants"][0]["basis"], "listened")

    def test_a_test_sentence_heard_wrong_elsewhere_teaches_nothing(self):
        self.run_tool(["A"], "The name is Marla. It was Marla’s name. Give it to Marla.")
        self.assertFalse((self.directory / FILENAME).exists())
        self.assertTrue(any("not cleanly enough" in line for line in self.output), self.output)


if __name__ == "__main__":
    unittest.main()
