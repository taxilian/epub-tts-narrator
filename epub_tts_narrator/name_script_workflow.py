"""Resumable independent contextual name classification and exact manual review."""

from .analysis_directory import _completed_directory
from .name_script import annotate, binding
from .name_script_prompts import SYSTEM
from .name_script_state import (check_size_budget, checked_windows, load_name_script, mention_id, new_state,
                                read_state, save_state, windows)
from .renderer.run_locks import acquire_directory_lock
from .pronunciations import _directory


def prepare_name_script(directory, config, *, requests=None, progress=print, client_factory=None):
    directory = _directory(directory)
    document = _completed_directory(directory)[0]
    lock = acquire_directory_lock(directory / ".name-script.lock")
    try:
        old = read_state(directory)
        state = old if old and old["binding"] == binding(document) else new_state(document, old)
        checked_windows(state, document)
        if state["status"] == "ready":
            return load_name_script(directory, document)
        check_size_budget(document, state)
        save_state(directory, state)
        for target in list(windows(document, state["names"]))[len(state["windows"]):]:
            occurrences = target["candidates"]
            votes = []
            def validate(response):
                if (not isinstance(response, dict) or set(response) != {"decisions"}
                        or not isinstance(response["decisions"], list)
                        or len(response["decisions"]) != len(occurrences)):
                    raise ValueError("Name classification must cover every candidate exactly once")
                for decision, item in zip(response["decisions"], occurrences):
                    if not isinstance(decision, str) or decision not in [*item["names"], "ordinary", "unresolved"]:
                        raise ValueError("Name classification has an invalid entity decision")
                return response["decisions"]
            if occurrences:
                if requests is None:
                    from .pronunciation_workflow import _requests
                    requests = _requests(directory, config, progress, client_factory)
                relevant = {key for item in occurrences for key in item["names"]}
                for reviewer in (1, 2):
                    payload = {"mode": "name_script", "version": 1, "binding": state["binding"],
                               "reviewer": reviewer, "window": target,
                               "names": {key: state["names"][key] for key in sorted(relevant)}}
                    votes.append(requests.run(SYSTEM, payload, validate))
                decisions = [a if a == b else "unresolved" for a, b in zip(*votes)]
            else:
                decisions = []
            state["windows"].append({key: target[key] for key in ("block_id", "start", "end")} | {
                "text": annotate(target["text"], occurrences, decisions),
                "unresolved": [i for i, value in enumerate(decisions) if value == "unresolved"]})
            save_state(directory, state)
        current = _completed_directory(directory)[0]
        if binding(current) != state["binding"]:
            raise ValueError("Source or name identity changed during preparation; rerun prepare-name-script")
        state["status"] = "ready"
        checked_windows(state, current)
        save_state(directory, state)
        return load_name_script(directory, current)
    finally:
        lock.rmdir()


def resolve_mention(directory, selected, decision):
    directory = _directory(directory)
    document = _completed_directory(directory)[0]
    lock = acquire_directory_lock(directory / ".name-script.lock")
    try:
        state = read_state(directory)
        if state is None or state["status"] != "ready":
            raise ValueError("Prepare the name script before resolving mentions")
        for saved, target, mentions in checked_windows(state, document):
            marked = {(item["start"], item["end"]): item["name_id"] for item in mentions}
            decisions = ["unresolved" if i in saved["unresolved"] else
                         marked.get((item["start"], item["end"]), "ordinary")
                         for i, item in enumerate(target["candidates"])]
            for index, item in enumerate(target["candidates"]):
                identity = mention_id(state, target["block_id"], target["start"] + item["start"],
                                      target["start"] + item["end"])
                if identity != selected:
                    continue
                if decision not in [*item["names"], "ordinary", "unresolved"]:
                    raise ValueError("Choose an allowed name ID, ordinary, or unresolved")
                decisions[index] = decision
                saved["text"] = annotate(target["text"], target["candidates"], decisions)
                saved["unresolved"] = [i for i, value in enumerate(decisions) if value == "unresolved"]
                save_state(directory, state)
                return
        raise ValueError("Mention ID is stale or unknown; list the current name script")
    finally:
        lock.rmdir()
