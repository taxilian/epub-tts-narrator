"""A group's extra (pool) voices are cast with their group behind them, and told apart from each other."""

from copy import deepcopy
import unittest

from epub_tts_narrator.audition_prompts import GROUP_MEMBER, SYSTEM as SCRIPT_SYSTEM
from epub_tts_narrator.audition_scripts import prepare_samples
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.ensembles import pool_label, pool_member
from epub_tts_narrator.voice_prompt_refresh import refresh_voice_descriptions
from epub_tts_narrator.voice_refresh_prompts import GROUP_VOICE, SYSTEM as REFRESH_SYSTEM
from test_audition_scripts import SCRIPT, bundle
from test_contracts import valid_document
from test_voice_prompt_refresh import RESPONSE


def group_document():
    """Mara's voice (v1) leads a pool of three: v1 and two extra voices."""
    document = valid_document()
    voices = document["cast"]["voices"]
    lead = next(voice for voice in voices if voice["id"] == "v1")
    lead["ensemble"] = {"members": ["v1-pool-2", "v1-pool-3"], "mode": "pool", "basis": "user"}
    voices += [{"id": "v1-pool-2", "description": "Mara's voice. Another of Mara: a little older.", "traits": [],
                "member_of": "v1"},
               {"id": "v1-pool-3", "description": "Mara's voice. Another of Mara: younger and brighter.",
                "traits": [], "member_of": "v1"}]
    return document


class Requests:
    def __init__(self, response):
        self.calls, self.response = [], response

    def run(self, system, payload, validate, *, cancelled=None):
        self.calls.append((system, deepcopy(payload)))
        return validate(deepcopy(self.response))


class PoolMemberTests(unittest.TestCase):
    def test_a_pool_voice_knows_its_group_its_place_and_the_other_voices(self):
        cast = group_document()["cast"]
        self.assertEqual(validate_document(group_document()), [])
        member = pool_member(cast, "v1-pool-3")
        self.assertEqual((member["group"], member["character"]["name"], member["number"], member["voices"]),
                         ("v1", "Mara", 3, 3))
        self.assertEqual(member["others"], ["v1", "v1-pool-2"])
        self.assertEqual(pool_label(cast, "v1-pool-2"), "Mara, voice 2")
        self.assertIsNone(pool_member(cast, "v1"))
        self.assertIsNone(pool_label(cast, "vn"))


class PoolScriptTests(unittest.TestCase):
    def script_request(self, voice_id):
        requests = Requests({"text": SCRIPT, "reason": "An original audition."})
        prepare_samples(bundle(group_document()), requests, voice_ids=[voice_id], workers=1)
        self.assertEqual(len(requests.calls), 1)
        return requests.calls[0]

    def test_a_pool_voices_audition_script_is_asked_for_with_its_group_behind_it(self):
        system, payload = self.script_request("v1-pool-2")
        self.assertEqual([person["name"] for person in payload["characters"]], ["Mara"])
        self.assertTrue(payload["short_speech_examples"], "the group's own lines are its examples")
        self.assertEqual(payload["group_voice"], {"group": "Mara", "voice": 2, "voices": 3})
        self.assertEqual(system, SCRIPT_SYSTEM + GROUP_MEMBER)

    def test_other_voices_requests_are_unchanged(self):
        system, payload = self.script_request("v1")
        self.assertNotIn("group_voice", payload)
        self.assertEqual(system, SCRIPT_SYSTEM)


class PoolDescriptionTests(unittest.TestCase):
    def test_rewriting_a_pool_voice_sends_its_group_and_the_voices_it_must_differ_from(self):
        requests = Requests(RESPONSE)
        refresh_voice_descriptions(bundle(group_document()), requests, voice_ids=["v1-pool-2"], workers=1,
                                   current_prompts={"v1": "A warm, low alto with a calm pace."})
        system, payload = requests.calls[0]
        self.assertEqual(system, REFRESH_SYSTEM + GROUP_VOICE)
        self.assertEqual([person["name"] for person in payload["characters"]], ["Mara"])
        self.assertEqual(payload["group_voice"]["group"], "Mara")
        self.assertEqual(payload["group_voice"]["voice"], 2)
        # The lead as it is cast now (its edited direction), then the other pool voice.
        self.assertEqual(payload["group_voice"]["other_voices"],
                         ["A warm, low alto with a calm pace.", "Mara's voice. Another of Mara: younger and brighter."])

    def test_rewriting_an_ordinary_voice_is_unchanged(self):
        requests = Requests(RESPONSE)
        refresh_voice_descriptions(bundle(group_document()), requests, voice_ids=["v1"], workers=1)
        system, payload = requests.calls[0]
        self.assertEqual(system, REFRESH_SYSTEM)
        self.assertNotIn("group_voice", payload)


if __name__ == "__main__":
    unittest.main()
