"""``audiobook-render local-server [status|stop|restart]``: see and control the local speech server.

``render`` and ``pronounce`` share one auto-started loopback ``serve`` process
(``auto_serve``). It outlives them, so this command shows whether it is
running, what it is doing and whether it still runs this checkout's code, and
stops or restarts it.
"""

from __future__ import annotations

from datetime import datetime


def _ago(started):
    try:
        then = datetime.fromisoformat(started)
    except (TypeError, ValueError):
        return ""
    seconds = max(0, int((datetime.now().astimezone() - then).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    when = then.strftime("%H:%M") if then.date() == datetime.now().date() else then.strftime("%b %d %H:%M")
    return f"started {when} ({f'{hours}h {minutes}m' if hours else f'{minutes}m'} ago)"


def describe(status):
    """Plain lines describing ``auto_serve.server_status``."""
    port = status["port"]
    lines = []
    if status["state"] == "absent":
        lines.append(f"Local speech server (port {port}): not running.")
        lines.append("  render and pronounce start it when they need it; just server restart starts it now.")
    elif status["state"] == "loading":
        lines.append(f"Local speech server (port {port}): starting (loading Drama 3).")
    else:
        health = status["health"]
        lines.append(f"Local speech server (port {port}): running.")
        facts = [f"process {health['pid']}" if health.get("pid") else None, _ago(health.get("started_at")) or None,
                 f"model {health.get('model')}", f"{health.get('concurrent_requests', 1)} batch row(s)"]
        lines.append("  " + ", ".join(fact for fact in facts if fact))
        work = health.get("work") or {}
        if work.get("requests"):
            rows = [f"{work[key]} {key}" for key in ("decoding", "prefilling", "waiting") if work.get(key)]
            lines.append(f"  working on {work['requests']} request(s)" + (f" ({', '.join(rows)})" if rows else ""))
        else:
            idle = health.get("idle_seconds")
            lines.append("  idle" + (f" for {int(idle // 60)}m {int(idle % 60)}s" if isinstance(idle, (int, float)) else "")
                         + f"; it exits after {status['idle_timeout_seconds']:g}s without requests")
        changed = health.get("code_changed")
        if not isinstance(changed, list):
            lines.append("  code: older than this checkout (it predates code reporting); restart it to update")
        elif changed:
            shown = ", ".join(path.rsplit("/", 1)[-1] for path in changed[:5])
            more = f" and {len(changed) - 5} more" if len(changed) > 5 else ""
            lines.append(f"  code: older than this checkout ({len(changed)} file(s) changed since it started: "
                         f"{shown}{more}); restart it to update")
        else:
            lines.append("  code: current")
    lines.append(f"  log: {status['log']}")
    if status["state"] != "absent":
        lines.append("just server restart replaces it; just server stop stops it (--force while it is working).")
    return lines


def run(args, config, *, notify=print):
    from .auto_serve import AutoServeError, resolve_small_job_engine, server_status, stop_server
    action = args.action or "status"
    try:
        if action == "stop":
            stopped = stop_server(config, force=args.force, notify=notify)
            notify("Stopped." if stopped else "The local speech server was not running.")
        elif action == "restart":
            stop_server(config, force=args.force, notify=notify)
            resolve_small_job_engine(config, notify=notify)
    except AutoServeError as exc:
        notify(str(exc))
        return 1
    for line in describe(server_status(config)):
        notify(line)
    return 0
