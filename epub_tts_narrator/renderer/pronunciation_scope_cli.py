from pathlib import Path

from ..name_script import FILENAME
from ..pronunciations import find_pronunciation, load_pronunciations
from ..scoped_pronunciations import identity_scope, scope_for, set_choice


def command_scope(bundle, args):
    kind, name_id = getattr(args, "pronunciation_class", None), getattr(args, "name_id", None)
    if kind == "name" and name_id is None:
        raise ValueError("--class name requires --name-id N<number>; list prepare-name-script BOOK --action list")
    if name_id is not None and kind != "name":
        raise ValueError("--name-id requires --class name")
    if kind is None:
        if (Path(bundle["directory"]) / FILENAME).exists() and args.action not in {"list", "discover"}:
            raise ValueError("Scoped mode requires --class name --name-id N<number>, or --class special; "
                             "word alone cannot identify an entity")
        return None
    scope = scope_for(bundle, name_id if kind == "name" else None)
    print(f"Pronunciation scope: {name_id or 'special'}")
    if args.action == "migrate":
        if not args.word:
            raise ValueError("--action migrate requires --word")
        old = find_pronunciation(load_pronunciations(bundle["directory"]), args.word)
        if old is None or old["status"] not in {"confirmed", "ignored"}:
            raise ValueError("No explicit legacy pronunciation to migrate for this word")
        set_choice(bundle["directory"], bundle, args.word, old.get("pronunciation"), old["status"],
                   scope=scope, migrate=True)
    return scope


def list_scopes(bundle, config, word=None):
    from ..name_script_state import load_name_script
    from .audition import display
    from .pronunciation_review import PronunciationReviewSession, _show
    snapshot = load_name_script(bundle["directory"], bundle)
    for name_id, name in snapshot["names"].items():
        print(display(f"\nName {name_id}: {', '.join(name['aliases'])}"))
        _show(PronunciationReviewSession(bundle, config, scope=identity_scope(bundle, name["character_id"]),
                                       name_snapshot=snapshot).rows(word), print)
    print("\nSpecial words (global spelling rules; marked names excluded)")
    _show(PronunciationReviewSession(bundle, config, scope={"class": "special"}, name_snapshot=snapshot).rows(word), print)
    return 0


def prepare_scope(bundle, config, config_path, scope, word=None):
    from ..pronunciations import save_proposal, set_pronunciation
    from ..pronunciation_workflow import create_pronunciation_proposer
    from .pronunciation_review import PronunciationReviewSession
    session = PronunciationReviewSession(bundle, config, scope=scope)
    if word and not session.rows(word):
        set_pronunciation(bundle["directory"], bundle, word, status="pending", scope=scope)
    rows = session.rows(word)
    proposer = create_pronunciation_proposer(bundle["directory"], config_path, scope=scope)
    count = 0
    for row in rows:
        if row["status"] == "pending" and "proposal" not in row and "draft" not in row:
            save_proposal(bundle["directory"], bundle, row["word"], proposer(row["word"]), scope=scope)
            count += 1
    print(f"Prepared {count} scoped pronunciation proposal(s); no choices were confirmed.")
    if not rows:
        print("No scoped candidates. Use --word, or run pronunciations --action discover to collect special words.")
    return 0
