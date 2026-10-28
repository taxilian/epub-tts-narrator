from copy import deepcopy

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.series import link_analysis
from test_series import fixture, propose


TEXT = "Ally greeted her ally softly. Ally with your friends. ALLY waved."
PHONES = {"alphabet": "cmu_arpabet", "value": "AE1 L IY0"}


def book(directory, *, text=TEXT, two=False, previous=None, reuse=True):
    document = fixture()
    document["source"]["blocks"][0]["text"] = text
    document["cast"]["characters"][0].update(name="Ally", aliases=[], evidence=[{"block_id": "b1", "quote": "Ally"}])
    if two:
        document["source"]["blocks"][1]["text"] = "Ally waited."
        document["cast"]["voices"].append(dict(deepcopy(document["cast"]["voices"][1]), id="v2"))
        document["cast"]["characters"].append(dict(deepcopy(document["cast"]["characters"][0]), id="c2", voice_id="v2",
                                                   evidence=[{"block_id": "b2", "quote": "Ally"}]))
    base = document["segments"][0]
    document["segments"] = [dict(deepcopy(base), id=block["id"], block_id=block["id"], text=block["text"],
                                 start=0, end=len(block["text"])) for block in document["source"]["blocks"]]
    if previous is None:
        registry = fresh_registry(directory)
    else:
        document["source"]["book"].update(id=directory.name, title=directory.name)
        registry = continuation_registry(previous, document["source"])
        if reuse:
            prior = load_bundle(previous)
            propose(document, prior["cast"]["characters"][0]["id"])
    document = link_analysis(document, registry)
    publish_directory(document, registry, directory, export_bundle, continue_from=previous)
    return load_bundle(directory)


class NameClient:
    identity = {"model": "name-fixture"}

    def __init__(self, *, fail_at=None, disagree=False):
        self.calls = []
        self.fail_at = fail_at
        self.disagree = disagree

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if self.fail_at == len(self.calls):
            raise RuntimeError("offline outage")
        if payload["mode"] == "pronunciation_proposal":
            return {"phonemes": PHONES["value"]}
        target = payload["window"]
        decisions = []
        for item in target["candidates"]:
            before = target["text"][:item["start"]]
            after = target["text"][item["end"]:]
            if before.endswith("her ") or after.startswith(" with"):
                decisions.append("ordinary")
            elif self.disagree and payload["reviewer"] == 2:
                decisions.append("unresolved")
            else:
                decisions.append(item["names"][-1] if target["block_id"] == "b2" else item["names"][0])
        return {"decisions": decisions}


def requests(root, client):
    return CachedRequests(client, root / "cache", repair_attempts=0)
