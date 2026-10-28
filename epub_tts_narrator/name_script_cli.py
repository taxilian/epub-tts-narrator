import json

from .name_script_state import checked_windows, load_name_script, mention_id, read_state


def run_name_script(args):
    from .renderer.bundle import load_bundle
    from .name_script_workflow import prepare_name_script, resolve_mention
    if args.action == "prepare":
        from .config import load_config
        prepare_name_script(args.directory, load_config(args.config))
    elif args.action == "resolve":
        if not args.mention_id or not args.decision:
            raise ValueError("--mention-id and --decision are required for resolve")
        resolve_mention(args.directory, args.mention_id, args.decision)
    bundle = load_bundle(args.directory)
    snapshot = load_name_script(args.directory, bundle)
    state = read_state(args.directory)
    occurrences = []
    for saved, target, mentions in checked_windows(state, bundle):
        marked = {(item["start"], item["end"]): item["name_id"] for item in mentions}
        for index, item in enumerate(target["candidates"]):
            start, end = target["start"] + item["start"], target["start"] + item["end"]
            occurrences.append({"mention_id": mention_id(state, target["block_id"], start, end),
                                "block_id": target["block_id"], "start": start, "end": end,
                                "word": item["word"], "names": item["names"],
                                "decision": "unresolved" if index in saved["unresolved"] else
                                marked.get((item["start"], item["end"]), "ordinary")})
    names = {key: {field: name[field] for field in ("character_id", "aliases")}
             for key, name in snapshot["names"].items()}
    print(json.dumps({"names": names, "occurrences": occurrences}, ensure_ascii=False, indent=2))
    return 0
