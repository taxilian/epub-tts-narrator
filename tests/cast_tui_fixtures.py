"""A recording stand-in for CastingSession and AudioPlayer, for casting-screen tests."""

from copy import deepcopy


def take(candidate_id, *, selected=False, inherited=False, passed=True, clips=None):
    creation = {"created_at": "2026-11-01T10:20:30+00:00",
                "designer": {"engine": "gemini_voice_design", "model": "fixture"}}
    if clips:
        creation["reference_clips"] = [{} for _ in range(clips)]
    return {"candidate_id": candidate_id, "reference_audio": f"/voices/{candidate_id}.wav",
            "reference_text": "Sample.", "creation": creation, "verification": {"passed": passed},
            "selected": selected, "inherited": inherited, "asset_hash": candidate_id}


def voice_row(voice_id, name, *, candidates=(), selected=None, canonical=None, derived=None,
              regeneration=None, clips=3, clip_source="default", traits=""):
    return {
        "voice_id": voice_id, "canonical_voice_id": canonical or voice_id, "label": name,
        "derived_from": derived, "clone_stale": False, "characters": [name], "roles": [f"role-{voice_id}"],
        "description": f"{name}: a calm voice", "speech_traits": traits,
        "reference_clips": {"count": clips, "source": clip_source, "designer_records_several": True},
        "prompt": f"A calm voice for {name}.", "reference_text": f"Hello, I am {name}.",
        "sample_source": {"kind": "book_excerpt", "chapter_title": "Chapter 1"},
        "selected": selected, "regeneration": regeneration, "candidates": list(candidates), "notice": "",
    }


def cast_rows():
    accepted = take("c-ava-2", selected=True, clips=3)
    return [
        voice_row("v-ava", "Ava", candidates=[take("c-ava-1"), accepted], selected=accepted),
        voice_row("v-ben", "Ben"),
        voice_row("v-cole", "Cole", candidates=[take("c-cole-1")]),
        voice_row("v-dee", "Dee", selected=take("c-dee-p", inherited=True) | {"inherited": True}),
        voice_row("v-eli", "Eli", regeneration={"state": "failed", "error": "verification failed", "inherited": False}),
        voice_row("v-fay", "Fay", selected=accepted, canonical="v-ava"),
        voice_row("v-ava-old", "Ava (older)", selected=take("c-old", selected=True),
                  derived={"voice_id": "v-ava", "kind": "aging", "instruction": "older"}),
    ]


class FakeSession:
    """Records every call; ``fail`` maps a method name to an exception to raise once."""

    def __init__(self, rows=None, *, config=None):
        self._rows = cast_rows() if rows is None else rows
        self.calls = []
        self.fail = {}
        self.config = config or {"casting": {"reference_clips": 3}, "voice_design": {"type": "gemini"}}
        self.notify = print

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        error = self.fail.pop(name, None)
        if error is not None:
            raise error

    def rows(self):
        return deepcopy(self._rows)

    def prepare_samples(self):
        self._call("prepare_samples")
        return []

    def regenerate(self, voice_id, *, voice_designer=None):
        self._call("regenerate", voice_id, voice_designer=voice_designer)
        return {"candidate_id": "new"}

    def accept(self, voice_id, candidate_id):
        self._call("accept", voice_id, candidate_id)

    def update_prompt(self, voice_id, **fields):
        self._call("update_prompt", voice_id, **fields)
        row = next(row for row in self._rows if row["voice_id"] == voice_id)
        row.update(fields)

    def share_voice(self, voice_id, target_voice_id):
        self._call("share_voice", voice_id, target_voice_id)

    def stop_sharing(self, voice_id):
        self._call("stop_sharing", voice_id)

    def set_reference_clips(self, voice_id, count):
        self._call("set_reference_clips", voice_id, count)

    def refresh_prompts(self, *, voice_ids=None, config_path=None):
        self._call("refresh_prompts", voice_ids=voice_ids, config_path=config_path)

    def ensure_ready_many(self, voice_ids):
        self._call("ensure_ready_many", list(voice_ids))
        return [{"voice_id": voice_id, "selected": {}, "error": None} for voice_id in voice_ids]


class FakePlayer:
    def __init__(self):
        self.started, self.stopped, self.running = [], 0, False

    def prepare(self, path):
        return path

    def start(self, path):
        self.started.append(path)
        self.running = True

    def poll(self):
        return self.running

    def stop(self):
        self.stopped += 1
        self.running = False
