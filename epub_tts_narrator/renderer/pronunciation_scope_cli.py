from pathlib import Path

from ..name_script import FILENAME
from ..pronunciations import find_pronunciation, load_pronunciations
from ..scoped_pronunciations import identity_scope, scope_for, set_choice


def command_scope(bundle, args):
    kind, name_id = getattr(args, "pronunciation_class", None), getattr(args, "name_id", None)
    if kind == "name" and name_id is None:
        raise ValueError("--class name requires --name-id N<number> from existing name-script annotations")
    if name_id is not None and kind != "name":
        raise ValueError("--name-id requires --class name")
    if kind is None:
        if (Path(bundle["directory"]) / FILENAME).exists() and args.action != "list":
            from ..name_script_state import load_name_script
            from ..name_script import alias_words
            snapshot = load_name_script(bundle["directory"], bundle)
            possible_names = {word for name in snapshot["names"].values() for word in alias_words(name)}
            if not args.word or args.word.lower() in possible_names:
                raise ValueError("Scoped mode requires --class name --name-id N<number>, or --class special; "
                                 "word alone cannot identify an entity")
        return None
    if not (Path(bundle["directory"]) / FILENAME).exists():
        raise ValueError("Scoped pronunciation requires saved name-script.json annotations; "
                         "the renderer will not generate them")
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
