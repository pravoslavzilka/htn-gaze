"""Dashboard backend: FastAPI + one static page. Also hosts the UDP receiver, so one process does everything:

    Pi / bridge --UDP--> Receiver --> Tiger (batched)   + Sentry (traces, logs)
    browser <-- /api/* (Tiger reads, live health)       + MJPEG scene stream straight from the gaze app

  python app.py            # http://127.0.0.1:8800

Tiger, Sentry, or the gaze app being down never crashes this: reads return {"ok": false, ...} and the page shows it.
"""
import logging
import re
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import queries
import telemetry
from receiver import Receiver
from tiger import TigerWriter

log = logging.getLogger("app")
HERE = Path(__file__).parent
PORT = config.env("DASHBOARD_PORT", 8800, int)
SCENE_URL = config.env("SCENE_URL", "http://127.0.0.1:8790/stream")

S = {}          # process state, filled at startup


@asynccontextmanager
async def lifespan(app):
    S["sentry"] = telemetry.init("dashboard")
    S["tiger"] = TigerWriter(config.TIGER_DSN, config.FLUSH_SECONDS, config.MAX_BUFFER_ROWS).start()
    S["db"] = queries.Db(config.TIGER_DSN)
    S["rx"] = Receiver(sink=S["tiger"], on_frame=telemetry.FrameTelemetry().on_frame)
    S["run"] = config.env("RUN_ID", "live")
    S["issues"] = (0.0, None)
    threading.Thread(target=S["rx"].serve, daemon=True, name="udp-receiver").start()
    yield
    S["rx"].stop()
    S["tiger"].stop()
    telemetry.flush()


app = FastAPI(lifespan=lifespan, title="Gaze dashboard")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def guarded(fn):
    """Run a Tiger read; a failure becomes {"ok": false, "error": ...} instead of a 500."""
    try:
        return {"ok": True, **fn()}
    except queries.QueryError as e:
        return {"ok": False, "error": str(e)}


def resolve_run(run):
    """'auto' -> the run that is receiving right now, else the newest run in Tiger."""
    if run and run != "auto":
        return run
    snap = S["rx"].snapshot()
    if snap:
        best = min(snap, key=lambda r: snap[r]["age_s"])
        if snap[best]["age_s"] < 10:
            return best
    try:
        return queries.latest_run(S["db"])
    except queries.QueryError:
        return None


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/config")
def get_config():
    dsn = config.env("SENTRY_BROWSER_DSN") or config.env("SENTRY_DSN") or ""
    pid = (re.search(r"/(\d+)$", dsn) or [None, None])[1]
    org = config.env("SENTRY_ORG_SLUG")
    base = f"https://{org}.sentry.io" if org else "https://sentry.io"
    q = f"?project={pid}" if pid else ""
    return {"sentry_dsn": dsn, "environment": config.env("SENTRY_ENV", "demo"), "scene_url": SCENE_URL,
            "links": {"issues": f"{base}/issues/{q}", "traces": f"{base}/explore/traces/{q}",
                      "logs": f"{base}/explore/logs/{q}", "replays": f"{base}/replays/{q}",
                      "dashboards": f"{base}/dashboards/"},
            "issues_api": bool(config.env("SENTRY_AUTH_TOKEN") and org and config.env("SENTRY_PROJECT_SLUG"))}


@app.get("/api/runs")
def get_runs():
    r = guarded(lambda: {"runs": queries.runs(S["db"])})
    r["current"], r["live"] = S["run"], resolve_run("auto")
    return r


@app.get("/api/attention")
def get_attention(run: str = "auto", seconds: int = Query(60, ge=10, le=3600)):
    run = resolve_run(run)
    if not run:
        return {"ok": True, "run": None, "dwell": [], "timeline": []}
    r = guarded(lambda: queries.attention(S["db"], run, seconds, max(300, seconds)))
    r["run"] = run
    return r


@app.get("/api/omni")
def get_omni(limit: int = Query(10, ge=1, le=50)):
    return guarded(lambda: {"items": queries.omni_history(S["db"], limit)})


@app.get("/api/compare")
def get_compare(runs: str):
    ids = [r for r in runs.split(",") if r][:4]
    return guarded(lambda: {"runs": queries.compare(S["db"], ids)})


@app.get("/api/health")
def get_health(run: str = "auto"):
    run = resolve_run(run)
    snap = S["rx"].snapshot().get(run)
    t = S["tiger"]
    return {"run": run, "receiving": bool(snap and snap["age_s"] < 3), "live": snap,
            "bad_packets": S["rx"].bad_packets,
            "tiger": {"enabled": t.enabled, "reachable": S["db"].ok, "inserted": t.inserted, "buffered": len(t.buf),
                      "failures": t.failures, "dropped": t.dropped, "omni_inserted": t.omni_inserted},
            "sentry": S["sentry"]}


@app.get("/api/sentry/issues")
def get_issues():
    """Recent Sentry issues, only if a read-only token + org/project slugs are configured. Cached; never blocks long."""
    token, org, proj = (config.env(k) for k in ("SENTRY_AUTH_TOKEN", "SENTRY_ORG_SLUG", "SENTRY_PROJECT_SLUG"))
    if not (token and org and proj):
        return {"ok": False, "configured": False}
    at, cached = S["issues"]
    if cached is not None and time.time() - at < 15:
        return cached
    try:
        req = urllib.request.Request(f"https://sentry.io/api/0/projects/{org}/{proj}/issues/?statsPeriod=1h&limit=6",
                                     headers={"Authorization": f"Bearer {token}"})
        import json
        data = json.loads(urllib.request.urlopen(req, timeout=4).read())
        out = {"ok": True, "configured": True, "issues": [
            {"title": i.get("title"), "count": i.get("count"), "level": i.get("level"), "link": i.get("permalink"),
             "last_seen": i.get("lastSeen")} for i in data]}
    except Exception as e:
        out = {"ok": False, "configured": True, "error": str(e)[:120]}
    S["issues"] = (time.time(), out)
    return out


@app.get("/api/current_run")
def current_run():
    return {"run": S["run"]}


@app.post("/api/runs/new")
def new_run(body: dict = Body(default={})):
    """Start a new run: the bridge / sender pick the id up from /api/current_run. Nothing is deleted."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "", str(body.get("name") or "")) or datetime.now().strftime("run_%H%M%S")
    S["run"] = name[:40]
    return {"run": S["run"]}


@app.post("/api/reset")
def reset(body: dict = Body(default={})):
    """Demo reset in seconds: forget live counters; with {"run": "...", "delete": true} also delete that run from Tiger."""
    S["rx"].reset()
    res = {"ok": True, "reset": True}
    run = body.get("run")
    if body.get("delete") and run:
        try:
            queries.delete_run(S["db"], run)
            res["deleted"] = run
        except queries.QueryError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    return res


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host=config.env("DASHBOARD_HOST", "127.0.0.1"), port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
