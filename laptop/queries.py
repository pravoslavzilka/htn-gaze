"""Read side of Tiger for the dashboard. A dead or slow database returns QueryError, never an exception the API can't
handle, and the connection is re-opened on the next call."""
import logging
import threading
import time
from datetime import datetime, timezone

log = logging.getLogger("queries")
BUCKET_S = 10
BACKOFF_S = 5.0
NO_OBJECT = "(no object)"
STAGES = ("cap", "pupil", "gaze", "scene", "fix")


class QueryError(RuntimeError):
    pass


class Db:
    def __init__(self, dsn):
        self.dsn, self.conn, self.lock = dsn, None, threading.Lock()
        self.ok = None                    # last query succeeded? (None = never tried)
        self.retry_at = 0.0               # circuit breaker: after a failure, fail fast until then

    def _fail_fast(self):
        """While Tiger is down, answer instantly instead of every request waiting out its own connect timeout
        (which piled up the 1 s dashboard polling until the API stalled)."""
        if not self.dsn:
            raise QueryError("TIGER_DSN not set")
        wait = self.retry_at - time.monotonic()
        if wait > 0:
            raise QueryError(f"Tiger unreachable (retrying in {wait:.0f}s)")

    def _failed(self):
        self.ok = False
        self.retry_at = time.monotonic() + BACKOFF_S

    def q(self, sql, params=()):
        self._fail_fast()
        with self.lock:
            try:
                if self.conn is None or self.conn.closed:
                    import psycopg
                    # prepare_threshold=None: psycopg auto-prepares a query after 5 uses, and TimescaleDB's generic plan for
                    # "ORDER BY time DESC LIMIT $1" then returns DUPLICATE rows (seen live: 8 rows from a 5-row table).
                    self.conn = psycopg.connect(self.dsn, connect_timeout=3, autocommit=True, prepare_threshold=None,
                                                options="-c statement_timeout=4000")
                rows = self.conn.execute(sql, params).fetchall()
                self.ok = True
                return rows
            except Exception as e:
                self._failed()
                try:
                    if self.conn:
                        self.conn.close()
                except Exception:
                    pass
                self.conn = None
                raise QueryError(str(e)[:200]) from None

    def execute(self, sql, params=()):
        self._fail_fast()
        with self.lock:
            try:
                if self.conn is None or self.conn.closed:
                    import psycopg
                    self.conn = psycopg.connect(self.dsn, connect_timeout=4, autocommit=True, prepare_threshold=None)
                self.conn.execute(sql, params)
                self.ok = True
            except Exception as e:
                self._failed()
                self.conn = None
                raise QueryError(str(e)[:200]) from None


def dwell_and_timeline(rows, now, window_s):
    """Pure. rows = [(bucket_start_dt, obj, frames)] from the continuous aggregate.
    Dwell seconds per object over the last `window_s`: within each 10 s bucket an object gets its share of the frames
    times the seconds of that bucket that have actually elapsed (the newest bucket is only partly over).
    -> {"dwell": [{"obj", "frames", "seconds"}] sorted desc, "timeline": [{"bucket": iso, "counts": {obj: frames}}]}"""
    by_bucket = {}
    for b, obj, n in rows:
        by_bucket.setdefault(b, {})[obj or NO_OBJECT] = by_bucket.setdefault(b, {}).get(obj or NO_OBJECT, 0) + int(n)
    dwell = {}
    for b, counts in by_bucket.items():
        elapsed = (now - b).total_seconds()
        if elapsed <= 0 or elapsed > window_s + BUCKET_S:
            continue
        covered = min(BUCKET_S, elapsed)
        total = sum(counts.values()) or 1
        for obj, n in counts.items():
            e = dwell.setdefault(obj, {"obj": obj, "frames": 0, "seconds": 0.0})
            e["frames"] += n
            e["seconds"] += n / total * covered
    out = sorted(({**v, "seconds": round(v["seconds"], 1)} for v in dwell.values()), key=lambda v: -v["seconds"])
    timeline = [{"bucket": b.isoformat(), "counts": c} for b, c in sorted(by_bucket.items())]
    return {"dwell": out, "timeline": timeline}


def latest_run(db):
    r = db.q("select run from gaze_frames order by time desc limit 1")
    return r[0][0] if r else None


def attention(db, run, window_s=60, history_s=300):
    now = datetime.now(timezone.utc)
    rows = db.q("""select bucket, obj, frames from gaze_10s
                   where run = %s and bucket >= now() - make_interval(secs => %s) order by bucket""",
                (run, history_s + BUCKET_S))
    return dwell_and_timeline(rows, now, window_s)


def runs(db):
    return [{"run": r, "frames": n, "first": a.isoformat(), "last": b.isoformat()} for r, n, a, b in
            db.q("select run, count(*), min(time), max(time) from gaze_frames group by run order by max(time) desc")]


def omni_history(db, limit=10):
    return [{"time": t.isoformat(), "run": r, "object": o, "transcript": q, "answer": a, "latency_ms": ms}
            for t, r, o, q, a, ms in db.q(
                "select time, run, object, transcript, answer, latency_ms from omni_interactions order by time desc limit %s",
                (limit,))]


def compare(db, run_ids):
    """Per-stage average and p95 (ms) for each run, plus tracking loss, for the before/after comparison."""
    cols = ", ".join(f"avg({s}_ms), percentile_cont(0.95) within group (order by {s}_ms)" for s in STAGES)
    rows = db.q(f"""select run, count(*), count(scene_ms), avg((conf < 0.5)::int), {cols}
                    from gaze_frames where run = any(%s) group by run""", (list(run_ids),))
    out = {}
    for row in rows:
        run, n, timed, loss = row[:4]
        vals = row[4:]
        stages = {s: {"avg": _r(vals[2 * i]), "p95": _r(vals[2 * i + 1])} for i, s in enumerate(STAGES)}
        out[run] = {"frames": n, "timed_frames": timed, "tracking_loss": _r(loss, 3), "stages": stages,
                    "total_avg": _r(sum(v["avg"] or 0 for v in stages.values())) if timed else None}
    return out


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def delete_run(db, run):
    db.execute("delete from gaze_frames where run = %s", (run,))
    db.execute("delete from omni_interactions where run = %s", (run,))
    db.execute("call refresh_continuous_aggregate('gaze_10s', NULL, NULL)")
