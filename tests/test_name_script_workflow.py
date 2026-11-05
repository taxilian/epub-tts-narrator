from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from epub_tts_narrator.name_script_cli import run_name_script
from epub_tts_narrator.name_script import binding
from epub_tts_narrator.name_script_state import checked_windows, load_name_script, new_state, read_state
from epub_tts_narrator.name_script_workflow import prepare_name_script, resolve_mention
from epub_tts_narrator.scoped_pronunciations import scope_for, set_choice
from name_script_fixtures import NameClient, PHONES, book, requests


class NameScriptWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "book"
        self.bundle = book(self.directory, two=True)

    def prepare(self, client=None):
        return prepare_name_script(self.directory, {}, requests=requests(self.root, client or NameClient()))

    def test_preparation_persists_compact_exact_source_without_canonical_edits(self):
        before = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        result = self.prepare()
        state = read_state(self.directory)
        self.assertEqual(state["windows"][0]["text"], "(N1:Ally) greeted her ally softly. Ally with your friends. (N1:ALLY) waved.")
        self.assertEqual(state["windows"][1]["text"], "(N2:Ally) waited.")
        self.assertEqual(result["unresolved"], [])
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertNotIn("mentions", state)

    def test_partial_failure_resumes_cached_blind_vote_without_stale_ready_state(self):
        client = NameClient(fail_at=2)
        with self.assertRaises(RuntimeError):
            self.prepare(client)
        self.assertEqual(read_state(self.directory)["status"], "pending")
        with self.assertRaises(ValueError):
            load_name_script(self.directory, self.bundle)
        client.fail_at = None
        self.prepare(client)
        self.assertEqual(len(client.calls), 5)
        self.assertEqual([call["reviewer"] for call in client.calls], [1, 2, 2, 1, 2])
        self.prepare(NameClient(fail_at=1))

    def test_manual_exact_mention_resolution_unblocks_only_affected_rule(self):
        result = self.prepare(NameClient(disagree=True))
        scope = scope_for(self.bundle, "N1")
        with self.assertRaises(ValueError):
            set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope)
        for item in result["unresolved"]:
            if item["block_id"] == "b1":
                resolve_mention(self.directory, item["mention_id"], "N1")
        with self.assertRaises(ValueError):
            set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope)
        remaining = load_name_script(self.directory, self.bundle)["unresolved"]
        resolve_mention(self.directory, remaining[0]["mention_id"], "N2")
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope)
        self.assertEqual(load_name_script(self.directory, self.bundle)["unresolved"], [])
        with self.assertRaises(ValueError):
            resolve_mention(self.directory, "stale", "N1")

    def test_voice_descriptions_do_not_invalidate_but_alias_and_source_changes_do(self):
        self.prepare()
        changed = deepcopy(self.bundle)
        changed["cast"]["voices"][0]["description"] = "Updated profile"
        self.assertEqual(binding(changed), binding(self.bundle))
        load_name_script(self.directory, changed)
        changed["cast"]["characters"][0]["aliases"].append("Allie")
        with self.assertRaises(ValueError):
            load_name_script(self.directory, changed)
        changed = deepcopy(self.bundle)
        changed["source"]["blocks"][0]["text"] += "!"
        with self.assertRaises(ValueError):
            load_name_script(self.directory, changed)

    def test_removed_identity_number_is_never_reassigned(self):
        first = new_state(self.bundle)
        changed = deepcopy(self.bundle)
        changed["cast"]["characters"] = [dict(changed["cast"]["characters"][0], id="new-character")]
        state = new_state(changed, first)
        self.assertEqual(list(state["names"]), ["N3"])
        self.assertEqual(state["ids"][:2], first["ids"])

    def test_legacy_listing_and_resolution_preserve_all_correctable_occurrences(self):
        self.prepare()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run_name_script(SimpleNamespace(directory=self.directory, action="list"))
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual([item["decision"] for item in result["occurrences"]], ["N1", "ordinary", "ordinary", "N1", "N2"])
        selected = result["occurrences"][0]["mention_id"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_name_script(SimpleNamespace(directory=self.directory, action="resolve",
                                                   mention_id=selected, decision="ordinary"))
        self.assertEqual(code, 0)
        self.assertEqual(len(load_name_script(self.directory, self.bundle)["mentions"]["b1"]), 1)

    def test_ready_checkpoint_cannot_omit_a_window(self):
        self.prepare()
        state = read_state(self.directory)
        state["windows"].pop()
        with self.assertRaises(ValueError):
            checked_windows(state, self.bundle)

    def test_completed_window_is_not_repeated_after_later_provider_failure(self):
        client = NameClient(fail_at=3)
        with self.assertRaises(RuntimeError):
            self.prepare(client)
        state = read_state(self.directory)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(len(state["windows"]), 1)
        client.fail_at = None
        self.prepare(client)
        self.assertEqual([item["window"]["block_id"] for item in client.calls], ["b1", "b1", "b2", "b2", "b2"])

    def test_incomplete_or_injected_model_decisions_never_publish_ready(self):
        class InvalidClient(NameClient):
            def complete(self, system, payload):
                return {"decisions": ["(N1:Ally)"]}
        with self.assertRaises(ValueError):
            self.prepare(InvalidClient())
        self.assertEqual(read_state(self.directory)["status"], "pending")
        self.assertEqual(read_state(self.directory)["windows"], [])
