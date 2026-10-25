"""Resolve book locations and inherited voices without changing any ancestor."""

from copy import deepcopy
from pathlib import Path


def resolve_book_config(bundle, config, *, output=None, work_dir=None, epub=None):
    """Bind global settings to this book; analysis outputs own their voice store."""
    result = deepcopy(config)
    if "aliases" in result.get("voices", {}):
        raise ValueError("voices.aliases is book-specific; use this book's voice-aliases.json")
    directory = Path(bundle["directory"]).resolve()
    if result.get("_book_directory"):
        if result["_book_directory"] != str(directory):
            raise ValueError("Resolved renderer configuration belongs to another book")
    else:
        manifest = bundle["manifest"]
        analysis_directory = "analysis_directory" in manifest
        if analysis_directory:
            for section, key in (("voices", "directory"), ("voices", "series_id"),
                                 ("render", "work_dir"), ("render", "output_dir"),
                                 ("render", "epub")):
                if result[section].get(key) is not None:
                    raise ValueError(
                        f"Book-directory analyses determine their own locations and series identity; "
                        f"remove {section}.{key} from the global config. "
                        "Use --output, --work-dir, or --epub for book-specific rendering options.")
        series_id = manifest.get("provenance", {}).get("series", {}).get("series_id")
        configured = result["voices"].get("series_id")
        if configured and series_id and configured != series_id:
            raise ValueError("Configured voices.series_id must match the analyzer's series ID")
        result["voices"]["series_id"] = series_id or configured or bundle["source"]["book"]["id"]
        result["_book_local_voices"] = result["voices"].get("directory") is None
        for section, key, name in (("voices", "directory", "voices"),
                                   ("render", "work_dir", "render-work"),
                                   ("render", "output_dir", "audiobook")):
            if result[section].get(key) is None:
                result[section][key] = str(directory / name)
        result["_voice_parents"] = []
        if analysis_directory:
            from epub_tts_narrator.analysis_directory import continuation_directories
            result["_voice_parents"] = [str(path) for path in continuation_directories(directory)]
        result["_book_directory"] = str(directory)
    for key, value in (("output_dir", output), ("work_dir", work_dir), ("epub", epub)):
        if value is not None:
            result["render"][key] = str(Path(value).expanduser().resolve())
    if result["render"].get("epub") is None:
        from epub_tts_narrator.source_epub import source_epub_hint
        hinted = source_epub_hint(directory, bundle["source"]["book"]["id"])
        if hinted is not None:
            result["render"]["epub"] = str(hinted)
    validate_book_paths(bundle, result)
    return result


def validate_book_paths(bundle, config):
    """Permit book-owned subdirectories while protecting source and ancestors."""
    directory = Path(bundle["directory"]).resolve()
    stores = [Path(config[s][k]).resolve() for s, k in
              (("voices", "directory"), ("render", "work_dir"), ("render", "output_dir"))]

    def overlaps(a, b):
        return a == b or a in b.parents or b in a.parents

    if any(overlaps(a, b) for i, a in enumerate(stores) for b in stores[i + 1:]):
        raise ValueError("Voice, work, and output directories must not overlap")
    protected = [directory / name for name in
                 ("manifest.json", "analysis.json", "series-cast.json", "source.json", "audition-scripts.json", ".audition-scripts.lock",
                  "cast.json", "review.json", "scripts", "voice-aliases.json", ".voice-aliases.lock",
                  "source-epub.json",
                  ".audio-render.lock", ".analysis-update.lock")]
    if not config.get("_book_local_voices"):
        # Existing shared-store workflows retain their original input isolation.
        protected.append(directory)
    protected.extend(Path(path).resolve() for path in config.get("_voice_parents", []))
    for section, key in (("engine", "checkpoint"), ("verification", "model"),
                         ("verification", "reference_model"), ("voice_design", "checkpoint"),
                         ("render", "epub"), ("chapter_server", "data_dir")):
        if config.get(section, {}).get(key):
            protected.append(Path(config[section][key]).resolve())
    for store in stores:
        if store == directory or store in directory.parents or any(overlaps(store, path) for path in protected):
            raise ValueError("Voice/work/output directories must not overlap source, ancestor books, model inputs, or chapter server data")
    # Resolving a symlink must not turn the local voice directory into a write to
    # an ancestor or a shared library, including before its first generation.
    if config.get("_book_local_voices"):
        raw_voice = directory / "voices"
        if raw_voice.is_symlink() or stores[0] != raw_voice:
            raise ValueError("The book voice directory must not be a symlink")


def voice_store(bundle, config, *, local_aliases=None):
    """Construct current-first lookup; every writable operation stays local."""
    from .bundle import load_bundle
    from .book_aliases import read_book_aliases
    from .voice_bindings import validate_aliases
    from .voices import VoiceStore
    from epub_tts_narrator.narration_identity import author_name_key, author_narrator_name
    from functools import partial
    books = [bundle, *[load_bundle(path) for path in config.get("_voice_parents", [])]]
    entries, allowed_ids = [], set()
    # Author narration is a separate scope, never an implicit alias for the
    # main narrator or any fictional character, even when a cast shares IDs.
    protected_ids = {character["voice_id"] for book in books for character in book["cast"]["characters"]}
    protected_ids.update(role["voice_id"] for book in books for role in book["cast"]["roles"]
                         if role.get("character_id") is not None or role["id"] == "narrator_main")
    for index in range(len(books) - 1, -1, -1):
        book = books[index]
        allowed_ids.update(voice["id"] for voice in book["cast"]["voices"])
        loader = partial(read_book_aliases, Path(book["directory"]) / "voice-aliases.json",
                         config["voices"]["series_id"], book["source"]["book"]["id"], set(allowed_ids))
        if index == 0 and local_aliases is not None:
            overrides = validate_aliases(local_aliases)
            if (set(overrides) | set(overrides.values())) - allowed_ids:
                raise ValueError("Book voice aliases must use exact voice IDs from this book or its ancestors")
            loader = lambda: dict(overrides)
        directory = config["voices"]["directory"] if index == 0 else Path(book["directory"]) / "voices"
        store = VoiceStore(directory, config["voices"]["series_id"], cast=book["cast"],
                           alias_loader=loader)
        scopes = {}
        for role in book["cast"]["roles"]:
            name = author_narrator_name(role, book["source"]["book"])
            if name is not None:
                scopes.setdefault(author_name_key(name), set()).add(role["voice_id"])
        for identities in scopes.values():
            identities.difference_update(protected_ids)
        entries.append((index, book, directory, loader, store, scopes))

    # Old analyses used different IDs for the same author's afterword. Derive
    # aliases at lookup time so these saved books can inherit without rewriting
    # analyses or immutable audio. Local selections (including failed/pending
    # recasts) remain overrides. The active store runs this resolver once after
    # reading the chain's live explicit policy; no derived edges are persisted.
    author_ids = {voice_id for *_, scopes in entries for identities in scopes.values() for voice_id in identities}
    def author_aliases(existing):
        occupied = set()
        # Inspect every identity in every store: a local override may use an
        # ancestor ID through explicit policy rather than this book's cast ID.
        for _, _, _, _, local_store, _ in entries:
            for voice_id in author_ids:
                if local_store._read_selection(voice_id) is not None:
                    occupied.add(voice_id)
                    continue
                folder = local_store.folder(voice_id)
                local_store._check_directory(folder)
                if folder.exists():
                    occupied.add(voice_id)
        derived, nearest_author = {}, {}
        for *_, scopes in entries:
            for author, identities in scopes.items():
                if len(identities) != 1:
                    # Do not skip an ambiguous scope and select an older voice.
                    nearest_author.pop(author, None)
                    continue
                voice_id = next(iter(identities))
                previous = nearest_author.get(author)
                if (previous is not None and previous != voice_id and voice_id not in occupied
                        and voice_id not in existing and voice_id not in derived):
                    candidate = {**derived, voice_id: previous, **existing}
                    try:
                        validate_aliases(candidate)
                    except ValueError:
                        # An explicit reverse link (old -> current replacement)
                        # takes precedence over inferring current -> old.
                        pass
                    else:
                        derived[voice_id] = previous
                nearest_author[author] = voice_id
        return derived

    stores = []
    for index, book, directory, loader, store, _ in entries:
        if index == 0:
            store = VoiceStore(directory, config["voices"]["series_id"], cast=book["cast"],
                               alias_loader=loader, parents=list(reversed(stores)),
                               automatic_alias_loader=author_aliases if author_ids else None)
        stores.append(store)
    store.book_voice_ids = set(allowed_ids)
    return store
