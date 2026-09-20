"""Tiger Data (TimescaleDB) writer. Batches rows and flushes about once a second on its own thread.

An outage never reaches the pipeline: failed batches stay in a bounded buffer, the connection is retried
with backoff, and the oldest rows are dropped (and counted) if the buffer overflows.
Two tables are written: gaze_frames (one row per Pi frame) and omni_interactions (one row per voice request).
"""
import collections
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("tiger")

COLS = ("time", "run", "frame_id", "cap_ms", "pupil_ms", "gaze_ms", "scene_ms", "fix_ms",
        "conf", "gx", "gy", "obj", "obj_conf")
INSERT = f"INSERT INTO gaze_frames ({','.join(COLS)}) VALUES ({','.join(['%s'] * len(COLS))})"
OMNI_COLS = ("time", "run", "object", "transcript", "answer", "latency_ms")
OMNI_INSERT = f"INSERT INTO omni_interactions ({','.join(OMNI_COLS)}) VALUES ({','.join(['%s'] * len(OMNI_COLS))})"


def schema_statements():
    """schema.sql split into single statements: comments removed, and semicolons inside $$ ... $$ bodies kept."""
    lines = Path(__file__).with_name("schema.sql").read_text().splitlines()
    text = "\n".join(l for l in lines if not l.strip().startswith("--"))
    out, cur, in_body = [], [], False
    for part in re.split(r"(\$\$|;)", text):
        if part == "$$":
            in_body = not in_body
        if part == ";" and not in_body:
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
        else:
            cur.append(part)
    tail = "".join(cur).strip()
    return out + ([tail] if tail else [])


class TigerWriter:
    def __init__(self, dsn, flush_s=1.0, max_rows=50000, connect=None):
        self.dsn, self.flush_s = dsn, flush_s
        self.buf = collections.deque(maxlen=max_rows)          # gaze_frames rows
        self.omni_buf = collections.deque(maxlen=1000)         # omni_interactions rows
        self.lock = threading.Lock()
        self.conn, self.next_try, self.backoff = None, 0.0, 1.0
        self.inserted = self.dropped = self.failures = self.omni_inserted = 0
        self._connect = connect or self._psycopg_connect
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="tiger-writer")

    @property
    def enabled(self):
        return bool(self.dsn)

    def start(self):
        if not self.enabled:
            log.warning("TIGER_DSN not set: nothing is stored")
            return self
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(3)
        if self.enabled:
            self.flush()

    def add(self, row):
        """Never blocks or raises: called from the UDP receive loop."""
        if not self.enabled:
            return
        with self.lock:
            if len(self.buf) == self.buf.maxlen:
                self.dropped += 1
            self.buf.append(row)

    def add_omni(self, row):
        """(time, run, object, transcript, answer, latency_ms). Never blocks or raises."""
        if self.enabled:
            with self.lock:
                self.omni_buf.append(row)

    def _psycopg_connect(self):
        import psycopg
        conn = psycopg.connect(self.dsn, connect_timeout=5, autocommit=True)
        for stmt in schema_statements():
            conn.execute(stmt)
        return conn

    def flush(self):
        with self.lock:
            rows, omni = list(self.buf), list(self.omni_buf)
        if (not rows and not omni) or time.monotonic() < self.next_try:
            return
        try:
            if self.conn is None:
                self.conn = self._connect()
                self.backoff = 1.0
                log.info("connected to Tiger")
            with self.conn.cursor() as cur:
                if rows:
                    cur.executemany(INSERT, rows)
                if omni:
                    cur.executemany(OMNI_INSERT, omni)
        except Exception as e:  # any driver/network error: keep the rows, retry later
            self.failures += 1
            log.error("Tiger insert failed (%s); %d rows buffered, retry in %.0fs", e, len(rows) + len(omni), self.backoff)
            try:
                if self.conn:
                    self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.next_try = time.monotonic() + self.backoff
            self.backoff = min(self.backoff * 2, 30)
            return
        with self.lock:
            for _ in range(min(len(rows), len(self.buf))):
                self.buf.popleft()
            for _ in range(min(len(omni), len(self.omni_buf))):
                self.omni_buf.popleft()
        self.inserted += len(rows)
        self.omni_inserted += len(omni)

    def _loop(self):
        while not self._stop.wait(self.flush_s):
            self.flush()
