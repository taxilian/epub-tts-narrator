"""A book whose EPUB names no author has it worked out from its own pages, or asked.

Conversion tools often leave ``dc:creator`` out or write a placeholder
(Calibre's ``Unknown``, dropped at extraction: ``book_details.real_authors``).
``analyze`` (a fresh book's checks and every ``analyze BOOK`` rerun) then
sends the book's front and back matter (title page, copyright page, "also by"
lists, about the author) to the model with ``rerun_review_prompts.AUTHOR_FIND``:
each author it names must come with an exact quote from a paragraph shown that
contains the name. Narrator-read books leave publisher pages out of the saved
source, so those paragraphs are read again from the remembered EPUB when it is
this book's own file. The answer is cached like every analysis request.

Without a confident answer, a terminal is asked (the run shell shows it as a
dialog through ``PromptBridge``); Enter leaves the author unknown until the
next rerun, and ``review-ui`` can set it at any time. Nothing is ever made up:
no answer means no author. A found or typed author is saved as
``provenance.book_details.authors`` (``book_details.save_details``, scope
``book-details``); an author a person chose is never replaced.
"""

from __future__ import annotations

from pathlib import Path
import sys

FRONT_BLOCKS, FRONT_CHARS = 60, 9000
BACK_BLOCKS, BACK_CHARS = 25, 4000
BLOCK_CHARS = 600


def needs_author(document):
    """Whether the book has no author and nobody chose one (a person's "none" included)."""
    from .book_details import basis, document_book
    return not document_book(document)["authors"] and basis(document["provenance"], "authors") is None


def _omitted_blocks(directory, document):
    """Paragraphs the saved source left out (publisher pages), from this book's own remembered EPUB."""
    omitted = {entry.get("block_id") for entry in document["source"].get("exclusions", [])
               if isinstance(entry, dict) and isinstance(entry.get("block_id"), str)}
    if not omitted:
        return None
    from .book_update import accepted_epub_digests, epub_digest
    from .source_epub import source_epub_hint
    try:
        path = source_epub_hint(directory, document["source"]["book"]["id"])
        if path is None or not path.is_file() or epub_digest(path) not in accepted_epub_digests(document):
            return None
        from .epub import extract_epub
        policy = document["source"].get("extraction_policy")
        include_auxiliary = isinstance(policy, str) and any(
            part.strip() == "auxiliary-spine=include" for part in policy.split(";"))
        extracted = extract_epub(path, include_auxiliary=include_auxiliary)
    except (ValueError, OSError, KeyError):
        return None
    return extracted["blocks"]


def matter(document, directory=None):
    """The paragraphs shown: the book's start and end, with left-out publisher pages restored where possible."""
    blocks = document["source"]["blocks"]
    extracted = _omitted_blocks(directory, document) if directory is not None else None
    if extracted is not None:
        # Publisher pages are front matter almost always; they go first, in the EPUB's order.
        omitted = {entry["block_id"] for entry in document["source"]["exclusions"]
                   if isinstance(entry, dict) and isinstance(entry.get("block_id"), str)}
        saved = {block["id"] for block in blocks}
        blocks = [block for block in extracted if block["id"] in omitted and block["id"] not in saved] + blocks
    texts = [" ".join(block["text"].split())[:BLOCK_CHARS] for block in blocks]
    texts = [text for text in texts if text]

    def take(rows, limit_blocks, limit_chars):
        chosen, size = [], 0
        for text in rows[:limit_blocks]:
            if size + len(text) > limit_chars and chosen:
                break
            chosen.append(text)
            size += len(text)
        return chosen
    front = take(texts, FRONT_BLOCKS, FRONT_CHARS)
    back = list(reversed(take(list(reversed(texts[len(front):])), BACK_BLOCKS, BACK_CHARS)))
    return front + back


def find_authors(title, paragraphs, run):
    """``[(name, quote)]`` the paragraphs show wrote the book; ``[]`` when they do not show it."""
    from .book_details import clean_name, placeholder
    from .narrator_read import _unwrap
    from .rerun_review_prompts import AUTHOR_FIND, AUTHOR_REVIEW_VERSION
    if not paragraphs:
        return []
    numbered = {number: text for number, text in enumerate(paragraphs, 1)}
    payload = {"mode": "author_find", "v": AUTHOR_REVIEW_VERSION, "title": title,
               "text": "\n".join(f"{number}|{text}" for number, text in numbered.items())}

    def fold(text):
        return " ".join(text.casefold().split())

    def validate(response):
        answers = _unwrap(response.get("a")) if isinstance(response, dict) else None
        if not isinstance(answers, list):
            raise ValueError('response must be {"a":[[name,N,quote]]} or {"a":[]}')
        found, seen = [], set()
        for answer in answers:
            if not (isinstance(answer, list) and len(answer) >= 3 and isinstance(answer[0], str)
                    and type(answer[1]) is int and isinstance(answer[2], str)):
                raise ValueError("each author must be [name, paragraph number, exact quote]")
            name, quote = clean_name(answer[0]), answer[2].strip()
            text = numbered.get(answer[1])
            if text is None or not quote or quote not in text:
                raise ValueError(f"{name!r}: the quote must be exact text from paragraph N shown here")
            if placeholder(name) or len(name) > 200 or fold(name) not in fold(quote):
                raise ValueError(f"{name!r}: the quote must contain the author's name as the book prints it")
            if fold(name) == fold(title) or fold(name) in seen:
                continue
            seen.add(fold(name))
            found.append((name, quote))
        return found
    return run(AUTHOR_FIND, payload, validate)


def ask_author(title, *, ask=None, output=print):
    """A person's answer: author names, or ``[]`` (Enter, end of input) to leave it unknown for now."""
    from .book_details import split_authors
    output(f"The book's author is not in its EPUB and its pages do not say who wrote “{title}”.")
    for _ in range(3):
        try:
            answer = (ask or input)("Author of this book (several: separate with ;), or Enter to leave it "
                                    "unknown for now: ")
        except EOFError:
            return []
        try:
            return split_authors(answer or "")
        except ValueError as error:
            output(f"Author: {error}")
    return []


def review_author(directory, config, client_factory, progress, *, recheck_cache=None, ask=None,
                  is_interactive=None):
    """``analyze``: find a missing author in the book's pages, else ask a terminal; True when one was saved."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .book_details import document_book, save_details
    from .busy_retry import interactive
    from .pipeline import CachedRequests
    from .analysis_revision import related_directories
    import hashlib
    directory = Path(directory).resolve()
    receipt = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
               for path in related_directories([directory])}
    document = _completed_directory(directory)[0]
    if not needs_author(document):
        return False
    title = document_book(document)["title"]
    found = []
    if config["provider"]["model"]:
        settings = config["analysis"]
        client = client_factory(config["provider"])
        print(f"Author: the EPUB names none; reading the book's title, copyright and other front pages for it with "
              f"{client.model} at {client.base_url}; those pages will be sent to this endpoint.", file=sys.stderr)
        requests = CachedRequests(client, recheck_cache or analysis_cache_directory(directory, settings["cache_dir"]),
                                  repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        from .provider import ProviderError
        try:
            found = find_authors(title, matter(document, directory), requests.run)
        except ValueError as error:
            print(f"Author: the model gave no usable answer ({error}).")
        except ProviderError as error:
            # Only the book's metadata waits on this; the rest of analyze goes on and a rerun asks again.
            print(f"Author: the model could not be asked ({error}); rerunning analyze tries again.")
    else:
        print("Author: the EPUB names none; set provider.model to find it in the book's pages.")
    if found:
        names = [name for name, _ in found]
        save_details(directory, authors=names, basis="analysis", evidence=[quote for _, quote in found],
                     document=document, expected_manifest_hashes=receipt, progress=print)
        print("Author: " + "; ".join(f"{name} (“{quote}”)" for name, quote in found)
              + ". review-ui changes it (b).")
        return True
    if not (interactive if is_interactive is None else is_interactive)():
        print("Author: unknown; the book's pages do not say who wrote it. Set it in review-ui (b), or rerun "
              "analyze in a terminal to be asked.")
        return False
    names = ask_author(title, ask=ask)
    if not names:
        print("Author: left unknown. review-ui sets it (b); rerunning analyze asks again.")
        return False
    save_details(directory, authors=names, basis="user", document=document, expected_manifest_hashes=receipt,
                 progress=print)
    print("Author: " + "; ".join(names) + " (as you gave it).")
    return True
