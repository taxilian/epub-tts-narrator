"""Ordered multi-book render queue with tail backfill onto one chapter service.

``audiobook-render render BOOK1 BOOK2`` renders several books through one
client process, in order. Remote books (a configured chapter service, or the
managed Runpod service via ``--remote``) run as independent threads: book k+1
is admitted as soon as the books before it cannot fill the service's total
slot capacity (tail backfill), and a book, once admitted, runs to completion —
books only ever open, never preempt. Without any remote service the books
render strictly sequentially in one process, reusing one engine and verifier
instance (and its warm model) across books.

Every book loop owns a fresh configuration dict (workflow preflight and
render_remote write per-book keys onto the config they receive) and a notify
that prefixes each line with the book's short name, so interleaved remote
output stays readable. A run-fatal failure in one book is recorded and the
queue continues with the next; the exit code is 0 only when every book
rendered. The client never stops compute: the service's idle monitor owns
stopping, and interruption only stops admitting new books.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

from .diagnostics import configured_secrets, format_fatal

# How often the queue reads the service's scheduling snapshot while waiting to
# admit the next book, and how many consecutive below-capacity reads admit it
# (a single dip does not open a book — stage drains must not flap admissions).
POLL_SECONDS = 5.0
GATE_OPEN_POLLS = 2
# Grace period per still-running book thread on Ctrl-C: a thread that is
# already finishing cleans up (session DELETE, lock release) within it.
INTERRUPT_JOIN_SECONDS = 10.0

_LIVE_JOB_STATES = ("queued", "generating", "assembling")
_print_lock = threading.Lock()


class QueueError(ValueError):
    """The queue invocation itself is invalid; nothing was rendered."""


def _emit(notify, message):
    """One serialized notification line; book threads interleave safely."""
    with _print_lock:
        notify(message)


def _book_name(book):
    """The book's directory name, used to prefix its output lines."""
    return Path(book).name or str(book)


def _prefixed(prefix, notify):
    """A notify that prefixes every line, so per-book output stays readable."""
    if not prefix:
        return notify

    def line(message):
        _emit(notify, prefix + str(message))

    return line


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


def _count(value):
    return value if type(value) is int and value >= 0 else 0


def gate_capacity(scheduling, capabilities):
    """Total render slots the service advertises, or None when unknown.

    Resolution order: the engine runtime's TTS slots, the chapter server's
    coordinator count, then the capabilities document's parallel_chapters.
    Older servers may offer none of these.
    """
    engine = scheduling.get("engine") if isinstance(scheduling, dict) else None
    capacity = engine.get("capacity") if isinstance(engine, dict) else None
    slots = _positive_int(capacity.get("tts_slots") if isinstance(capacity, dict) else None)
    if slots is not None:
        return slots
    coordinators = _positive_int(scheduling.get("coordinators") if isinstance(scheduling, dict) else None)
    if coordinators is not None:
        return coordinators
    parallel = capabilities.get("parallel_chapters") if isinstance(capabilities, dict) else None
    return _positive_int(parallel)


def gate_busy(scheduling):
    """Live synthesis work on the service, or None when unknown.

    The engine runtime's synthesis stage gauges when present (an omitted
    stage means that stage is idle), else the chapter server's queued,
    generating, and assembling job counts. Neither being readable means
    the poll cannot describe the service's load.
    """
    engine = scheduling.get("engine") if isinstance(scheduling, dict) else None
    if isinstance(engine, dict):
        stages = engine.get("stages")
        if isinstance(stages, dict):
            synthesis = stages.get("synthesis")
            if not isinstance(synthesis, dict):
                return 0
            return _count(synthesis.get("queued_now")) + _count(synthesis.get("active_now"))
    jobs = scheduling.get("jobs") if isinstance(scheduling, dict) else None
    if isinstance(jobs, dict):
        return sum(_count(jobs.get(state)) for state in _LIVE_JOB_STATES)
    return None


def gate_decision(scheduling, capabilities):
    """One pure admission decision: "admit", "wait", or "unknown"."""
    capacity = gate_capacity(scheduling, capabilities)
    busy = gate_busy(scheduling)
    if capacity is None or busy is None:
        return "unknown"
    return "admit" if busy < capacity else "wait"


def first_completed_chapter(scheduling, rows=None):
    """Whether the service shows at least one completed chapter job."""
    jobs = scheduling.get("jobs") if isinstance(scheduling, dict) else None
    if isinstance(jobs, dict) and _count(jobs.get("complete")) > 0:
        return True
    return any(isinstance(row, dict) and row.get("state") == "complete" for row in (rows or []))


class ServiceGate:
    """Read-only admission gate over the chapter service's scheduling snapshot.

    One ChapterClient per queue run polls ``GET v1/jobs?limit=1``; the
    response's job rows and scheduling block (or their absence) plus the
    capabilities document decide whether the service has spare slots. No
    session is opened and nothing is submitted: observing never mutates the
    service, and an unreachable service simply never admits.
    """

    _UNFETCHED = object()

    def __init__(self, client_config):
        self._client_config = client_config
        self._client = None
        self._capabilities = self._UNFETCHED
        self._open_polls = 0

    def _service(self):
        if self._client is None:
            from .chapter_client import ChapterClient
            self._client = ChapterClient(self._client_config)
        return self._client

    def capabilities(self):
        """The service's capabilities document, fetched once; None without one."""
        if self._capabilities is self._UNFETCHED:
            from .chapter_client import ChapterClientError
            try:
                payload = self._service().request("GET", "v1/capabilities")
            except ChapterClientError:
                payload = None
            self._capabilities = payload if isinstance(payload, dict) else None
        return self._capabilities

    def poll(self):
        """One service read; {"rows", "scheduling"}, or None without data."""
        from .chapter_client import ChapterClientError
        try:
            response = self._service().request("GET", "v1/jobs?limit=1")
        except ChapterClientError:
            return None
        if not isinstance(response, dict):
            return None
        rows = response.get("jobs")
        scheduling = response.get("scheduling")
        return {"rows": rows if isinstance(rows, list) else [],
                "scheduling": scheduling if isinstance(scheduling, dict) else None}

    def observe(self, poll):
        """Apply the admission policy to one poll.

        Returns the admission reason once the service has shown spare
        capacity on GATE_OPEN_POLLS consecutive readable polls, or once a
        first completed chapter appears while capacity is unknown; None
        keeps the next book waiting.
        """
        if poll is None:
            self._open_polls = 0
            return None
        decision = gate_decision(poll["scheduling"], self.capabilities())
        if decision == "unknown":
            self._open_polls = 0
            if first_completed_chapter(poll["scheduling"], poll["rows"]):
                return ("service capacity is unknown; admitting after the previous "
                        "book's first completed chapter")
            return None
        if decision == "wait":
            self._open_polls = 0
            return None
        self._open_polls += 1
        if self._open_polls < GATE_OPEN_POLLS:
            return None
        capacity = gate_capacity(poll["scheduling"], self.capabilities())
        busy = gate_busy(poll["scheduling"])
        return f"spare capacity ({busy} of {capacity} slots)"

    def decidable(self, poll):
        """Whether a poll could describe the service's load at all."""
        if poll is None:
            return False
        return gate_decision(poll["scheduling"], self.capabilities()) != "unknown"


def _wait_for_admission(gate, previous_thread, poll_seconds):
    """Block until the next book may be admitted; returns the reason.

    The gate is polled every ``poll_seconds`` until its policy admits. A
    poll that cannot describe the service's load (unreadable, or without
    usable capacity data) plus a finished previous book also admits: its
    slots are its own again, and a gate that can never decide must not
    stall the queue behind it. A *readable* decision never shortcuts —
    the sustain policy alone opens books then.
    """
    while True:
        poll = gate.poll()
        reason = gate.observe(poll)
        if reason is not None:
            return reason
        if (not gate.decidable(poll) and previous_thread is not None
                and not previous_thread.is_alive()):
            return "the previous book finished and the service's scheduling data cannot decide"
        time.sleep(poll_seconds)


class _BookRun:
    """One admitted book's thread and outcome."""

    def __init__(self, index, book):
        self.index = index
        self.book = book
        self.name = _book_name(book)
        self.thread = None
        self.status = None  # ("ok", None) or ("failed", message)
        self.config = None
        self.finished = False


def _failure_message(error, config):
    secrets = configured_secrets(config) if isinstance(config, dict) else ()
    return format_fatal(error, secrets)


def _validate_queue_invocation(books, args):
    """Reject duplicate books and book-specific options before any work starts."""
    seen, duplicates = set(), []
    for book in books:
        key = str(Path(book).resolve())
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise QueueError("The render queue cannot render the same book more than once "
                         "(paths resolving to the same directory): " + ", ".join(duplicates))
    if len(books) > 1:
        if getattr(args, "chapters", None) is not None:
            raise QueueError("--chapters selects chapters of one book; omit it to render whole "
                             "books, or render one book at a time")
        for attribute, flag in (("output", "--output/-o"), ("work_dir", "--work-dir"),
                                ("epub", "--epub"), ("performance_plan", "--performance-plan")):
            if getattr(args, attribute, None) is not None:
                raise QueueError(f"{flag} belongs to a single book; omit it to render whole "
                                 "books, or render one book at a time")


def _prepare_book(book, args):
    """One book's fresh configuration dict, bundle, and plan.

    Mirrors the single-book render command's prologue. workflow preflight and
    render_remote write per-book keys onto the config they receive, so every
    book loads its own dict and no dict is ever shared between book loops.
    """
    from .bundle import load_bundle, make_plan
    from .book_context import resolve_book_config
    from .config import load_config
    config = load_config(args.config)
    bundle = load_bundle(book)
    config = resolve_book_config(bundle, config, output=getattr(args, "output", None),
                                 work_dir=getattr(args, "work_dir", None), epub=getattr(args, "epub", None))
    if getattr(args, "chapters", None) is not None:
        from .cli import chapter_numbers
        config["render"]["chapter_numbers"] = chapter_numbers(args.chapters, len(bundle["chapters"]))
    if getattr(args, "allow_review", None) is not None:
        config["render"]["allow_review"] = args.allow_review
    if getattr(args, "chapter_format", None) is not None:
        config["render"]["chapter_format"] = args.chapter_format
    if getattr(args, "censor", None) is not None:
        config["censorship"]["enabled"] = args.censor
    if getattr(args, "performance_plan", None) is not None:
        config["render"].update(performance_plan=str(args.performance_plan.expanduser().resolve()),
                                performance_mode="expressive")
    if getattr(args, "performance_mode", None) is not None:
        if getattr(args, "performance_plan", None) is not None and args.performance_mode == "legacy":
            raise QueueError("--performance-plan requires expressive mode; omit it to use legacy conditioning")
        config["render"]["performance_mode"] = args.performance_mode
    if not getattr(args, "chapters_only", False):
        from .book_metadata import check_publication_source
        check_publication_source(bundle, config)
    plan = make_plan(bundle, config)
    expected_series = bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
    if expected_series and expected_series != config["voices"]["series_id"]:
        raise QueueError("Configured voices.series_id must match the analyzer's series ID")
    if plan["errors"]:
        raise QueueError("\n".join(plan["errors"]))
    return bundle, config, plan


def _ensure_models(config):
    """The render command's model setup, before engine construction."""
    from .model_setup import ModelSetup
    setup = ModelSetup(config)
    workers = config.get("render_pool", {}).get("workers", [])
    remote_only = bool(workers) and all(worker["type"] != "local" for worker in workers)
    if not remote_only:
        setup.ensure("engine")
    setup.ensure("verification")


def _ensure_missing_voices(bundle, config, plan, args):
    """The render command's missing-voice handling for local rendering."""
    from .book_context import voice_store
    store = voice_store(bundle, config)
    missing = [voice for voice in plan["voice_ids"] if store.load(voice) is None]
    if not missing:
        return bundle
    workers = config.get("render_pool", {}).get("workers", [])
    remote_only = bool(workers) and all(worker["type"] != "local" for worker in workers)
    if remote_only:
        from .config import resolve_voice_design
        if resolve_voice_design(config)["type"] == "engine":
            raise QueueError("Missing voices need a voice designer: configure Gemini/ElevenLabs, "
                             "add a local renderer, or prepare the voices with cast first")
    from .model_setup import ModelSetup
    ModelSetup(config).ensure("voice_design")
    from epub_tts_narrator.audition_scripts import ensure_bundle_scripts
    return ensure_bundle_scripts(bundle, args.config, canonical_id=store.canonical_id, voice_ids=missing)


def _build_render_components(config):
    """The render command's engine and verifier construction."""
    from .engines import create_engine
    from .pool import create_render_components
    from .verification import create_verifier
    workers = config.get("render_pool", {}).get("workers", [])
    if workers:
        return create_render_components(config, verifier_factory=create_verifier)
    return (create_engine(config["engine"]),
            create_verifier(config["verification"], config.get("jev"), config.get("provider")))


def _report_remote_completion(result, config, bundle, *, chapters_only, notify):
    if chapters_only:
        total = len(bundle["chapters"])
        for chapter in (result or {}).get("chapters", []):
            notify(f"Completed chapter {chapter['number']} / {total}: "
                   f"{chapter.get('audio_file') or 'empty chapter (no speech)'}")
    else:
        notify(f"Completed audiobook: {config['render']['output_dir']}")


def _report_local_completion(result, config, bundle, *, chapters_only, notify):
    if chapters_only:
        notify(f"Completed chapter audio: {Path(config['render']['work_dir']) / 'chapters'}")
        if isinstance(result, dict):
            for chapter in result.get("chapters", []):
                notify(f"  Chapter {chapter['number']} / {len(bundle['chapters'])}: "
                       f"{chapter.get('audio_file') or 'empty chapter (no speech)'}")
    else:
        notify(f"Completed audiobook: {config['render']['output_dir']}")


def _summary(results, notify):
    """Per-book outcome summary; 0 only when every book rendered."""
    failures = [(name, message) for name, status, message in results if status != "ok"]
    if failures:
        _emit(notify, f"Render queue finished: {len(failures)} of {len(results)} book(s) failed:")
        for name, message in failures:
            _emit(notify, f"  {name}: {message}")
        return 1
    _emit(notify, f"Render queue finished: all {len(results)} book(s) rendered.")
    return 0


def _interrupted(runs, books, notify, join_seconds):
    """Stop admitting, give running books a brief cleanup window, report, 130."""
    _emit(notify, "Interrupted: no further books will be admitted; waiting briefly "
                  "for running books to clean up.")
    for run in runs:
        if run.thread is not None and run.thread.is_alive():
            run.thread.join(timeout=join_seconds)
    _emit(notify, "Render queue interrupted:")
    for run in runs:
        if run.status is None:
            state = "was still rendering and stopped with the process"
        else:
            state = "rendered" if run.status[0] == "ok" else f"failed: {run.status[1]}"
        _emit(notify, f"  {run.name}: {state}")
    for index in range(len(runs), len(books)):
        _emit(notify, f"  {_book_name(books[index])}: not started")
    return 130


def _acquire_managed_service(args, notify):
    """--remote only: reuse or bootstrap the managed Runpod chapter service.

    Returns the service's client configuration (its ``remote_chapters`` feeds
    every book's fresh config dict). The service keeps running afterwards; the
    server-side idle monitor owns stopping compute, never this client.
    """
    if not getattr(args, "remote", False):
        return None
    from .runpod import ensure_service
    from .runpod_config import load_settings
    service = ensure_service(load_settings(args.config), notify=notify)
    remote_chapters = service.get("remote_chapters") if isinstance(service, dict) else None
    if not isinstance(remote_chapters, dict) or not remote_chapters.get("url"):
        raise QueueError("The managed chapter service did not provide a remote_chapters "
                         "client configuration")
    return service


def _run_remote_queue(books, args, config, *, notify, poll_seconds, join_seconds):
    """Admit books in order while the service has spare slots; record failures.

    Each book runs ``render_remote`` on its own daemon thread with a fresh
    config dict and a prefixed notify; raising through that loop stops the
    book losslessly server-side (heartbeat stop, session DELETE, lock release).
    """
    from .chapter_client import render_remote
    # A configured [remote_chapters].url is a deliberate standing choice and
    # takes precedence: it is used as-is and --remote never bootstraps a
    # managed worker alongside it.
    managed = None if (config.get("remote_chapters") or {}).get("url") \
        else _acquire_managed_service(args, notify)
    multi = len(books) > 1
    total = len(books)
    gate = None
    if multi:
        managed_chapters = managed["remote_chapters"] if managed is not None else None
        gate_config = dict(managed_chapters) if managed_chapters is not None else dict(config["remote_chapters"])
        gate = ServiceGate(gate_config)
    runs = []

    def prefix_for(run):
        return f"[{run.name}] " if multi else ""

    def admit(index):
        run = _BookRun(index, books[index])
        book_notify = _prefixed(prefix_for(run), notify)

        def thread_main():
            try:
                bundle, book_config, plan = _prepare_book(run.book, args)
                run.config = book_config
                if managed is not None:
                    # The managed service produces M4A chapters, exactly like
                    # the runpod encode path's projected client configuration.
                    book_config["remote_chapters"] = dict(managed["remote_chapters"])
                    book_config["render"]["chapter_format"] = "m4a"
                result = render_remote(bundle, book_config, notify=book_notify,
                                       chapters_only=args.chapters_only)
                _report_remote_completion(result, book_config, bundle,
                                          chapters_only=args.chapters_only, notify=book_notify)
            except Exception as error:
                run.status = ("failed", _failure_message(error, run.config))
                _emit(notify, f"{prefix_for(run)}failed: {run.status[1]}")
            else:
                run.status = ("ok", None)
                _emit(notify, f"{prefix_for(run)}ok")
            finally:
                run.finished = True

        run.thread = threading.Thread(target=thread_main, name=f"render-queue-book-{index + 1}",
                                      daemon=True)
        runs.append(run)
        _emit(notify, f"{prefix_for(run)}Rendering book {index + 1} of {total}: {run.book}")
        run.thread.start()

    admit(0)
    next_index = 1
    try:
        while next_index < total:
            reason = _wait_for_admission(gate, runs[-1].thread, poll_seconds)
            _emit(notify, f"Admitting book {next_index + 1} of {total}: {reason}.")
            admit(next_index)
            next_index += 1
    except KeyboardInterrupt:
        return _interrupted(runs, books, notify, join_seconds)
    for run in runs:
        run.thread.join()
    return _summary([(run.name, run.status[0], run.status[1]) for run in runs], notify)


def _run_local_queue(books, args, *, notify):
    """Render books strictly sequentially, reusing one engine and verifier."""
    from .interrupts import RenderInterrupts, stop_if_requested
    from .workflow import render as render_book
    total = len(books)
    engine = verifier = None
    results = []
    interrupts = RenderInterrupts()
    interrupts.__enter__()
    try:
        for index, book in enumerate(books):
            name = _book_name(book)
            prefix = f"[{name}] "
            book_notify = _prefixed(prefix, notify)
            _emit(notify, f"{prefix}Rendering book {index + 1} of {total}: {book}")
            config = None
            try:
                bundle, config, plan = _prepare_book(book, args)
                bundle = _ensure_missing_voices(bundle, config, plan, args)
                if engine is None:
                    # Built once from the first successfully prepared book's
                    # config; the warm model then serves every later book.
                    _ensure_models(config)
                    engine, verifier = _build_render_components(config)
                result = render_book(bundle, config, engine, verifier, notify=book_notify,
                                     chapters_only=args.chapters_only)
                _report_local_completion(result, config, bundle,
                                         chapters_only=args.chapters_only, notify=book_notify)
                results.append((name, "ok", None))
                _emit(notify, f"{prefix}ok")
            except Exception as error:
                message = _failure_message(error, config)
                results.append((name, "failed", message))
                _emit(notify, f"{prefix}failed: {message}")
            stop_if_requested()
    except KeyboardInterrupt:
        _emit(notify, "Render queue interrupted:")
        for name, status, message in results:
            _emit(notify, f"  {name}: " + ("rendered" if status == "ok" else f"failed: {message}"))
        for index in range(len(results), total):
            _emit(notify, f"  {_book_name(books[index])}: not started")
        raise
    finally:
        interrupts.__exit__(None, None, None)
        for component in (verifier, engine):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except (ValueError, RuntimeError, OSError) as exc:
                    _emit(notify, f"Renderer cleanup warning: {exc}")
    return _summary(results, notify)


def render_queue_command(args, config_path, *, notify=print,
                         poll_seconds=POLL_SECONDS, join_seconds=INTERRUPT_JOIN_SECONDS):
    """Render every requested book in order; returns the process exit code.

    Remote mode (a configured ``[remote_chapters].url`` used as-is, or the
    managed Runpod service via ``--remote``) admits books with tail backfill;
    otherwise books render sequentially in one process with a shared engine.
    Each book loads its own fresh configuration from ``args.config``.
    """
    from .config import load_config
    books = [Path(book) for book in args.bundle]
    _validate_queue_invocation(books, args)
    config = load_config(config_path)
    remote = bool(getattr(args, "remote", False)) or bool((config.get("remote_chapters") or {}).get("url"))
    if remote:
        return _run_remote_queue(books, args, config, notify=notify,
                                 poll_seconds=poll_seconds, join_seconds=join_seconds)
    return _run_local_queue(books, args, notify=notify)
