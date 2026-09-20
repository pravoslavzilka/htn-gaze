"""Sentry on the laptop: tracing, structured logs, and spans around the OMNI calls. A no-op unless SENTRY_DSN is set,
so Sentry being down (or unconfigured) can never affect the pipeline.

Frame traces are RECONSTRUCTED: the Pi only sends durations, so for 1 in N frames (plus anomalous ones, rate-limited)
we build a "gaze_pipeline" transaction whose child spans cap / pupil / gaze / scene / fix are laid end to end and
anchored so the transaction ENDS at the laptop receive time. Verified against sentry-sdk 2.69: Span(start_timestamp=...)
and Span.finish(end_timestamp=...) accept datetimes.
"""
import contextlib
import logging
import time
from datetime import datetime, timedelta, timezone

import config

log = logging.getLogger("telemetry")
STAGES = ("cap", "pupil", "gaze", "scene", "fix")
_sdk = None
TRACED = ("gaze_pipeline", "omni_request")          # everything else (dashboard HTTP polling) is not sampled


def init(service, release=None):
    """-> True if Sentry is on. Safe to call once per process."""
    global _sdk
    dsn = config.env("SENTRY_DSN")
    if not dsn:
        log.info("SENTRY_DSN not set: Sentry is off")
        return False
    import sentry_sdk
    import sentry_sdk.logger  # noqa: F401  (the logs API lives in a submodule)

    def sampler(ctx):
        name = (ctx.get("transaction_context") or {}).get("name")
        return 1.0 if name in TRACED else 0.0

    sentry_sdk.init(dsn=dsn, environment=config.env("SENTRY_ENV", "demo"), release=release or config.env("SENTRY_RELEASE"),
                    traces_sampler=sampler, enable_logs=True, send_default_pii=False,
                    trace_propagation_targets=[],       # never send sentry-trace/baggage headers to OMNI / ElevenLabs / Tiger hosts
                    server_name=service, debug=config.env("SENTRY_DEBUG") == "1")
    sentry_sdk.set_tag("service", service)
    _sdk = sentry_sdk
    return True


def enabled():
    return _sdk is not None


def frame_spans(pkt, rx):
    """Pure. -> (start, [(stage, start, end)], total_ms): the stages laid end to end, ending at `rx` (a datetime).
    Stages the Pi did not measure (None) are left out."""
    durs = [(s, float(pkt[f"{s}_ms"])) for s in STAGES if pkt.get(f"{s}_ms") is not None]
    total = sum(d for _, d in durs)
    start = rx - timedelta(milliseconds=total)
    out, t = [], start
    for stage, d in durs:
        out.append((stage, t, t + timedelta(milliseconds=d)))
        t += timedelta(milliseconds=d)
    return start, out, total


class FrameTelemetry:
    """Receiver hook: traces 1 in `every` frames and anomalous ones, and logs tracking loss / packet loss / Pi events."""

    def __init__(self, every=None, anomaly_trace_gap_s=1.0):
        self.every = every or config.env("TRACE_EVERY_N", 30, int)
        self.gap = anomaly_trace_gap_s
        self.state = {}

    def on_frame(self, pkt, d):
        if _sdk is None:
            return
        run, f = d["run"], d["frame"]
        st = self.state.setdefault(run, {"lost": False, "last_anomaly_trace": 0.0})
        logs = []
        # log the TRANSITIONS of tracking loss, not every lost frame (that would be hundreds of identical lines)
        if d["tracking_lost"] and not st["lost"]:
            logs.append(("warning", "tracking_lost", {"conf": pkt.get("conf")}))
        elif st["lost"] and not d["tracking_lost"]:
            logs.append(("info", "tracking_recovered", {"conf": pkt.get("conf")}))
        st["lost"] = d["tracking_lost"]
        if d["lost_packets"]:
            logs.append(("warning", "packet_loss", {"missing_packets": d["lost_packets"]}))
        for ev in pkt.get("ev") or []:
            logs.append(("info" if ev == "blink" else "warning", f"pi_event_{ev}", {"event": ev}))
        anomaly = any(lvl == "warning" for lvl, _, _ in logs)
        now = time.monotonic()
        traced = f % self.every == 0 or (anomaly and now - st["last_anomaly_trace"] >= self.gap)
        if anomaly and traced:
            st["last_anomaly_trace"] = now
        try:
            if traced:
                self._traced(pkt, d, logs, anomaly)
            else:
                self._emit(logs, run, f)
        except Exception:
            log.exception("telemetry failed (ignored)")

    @staticmethod
    def _emit(logs, run, f):
        for level, msg, attrs in logs:
            getattr(_sdk.logger, level)(msg, attributes={"run": run, "frame": f, **attrs})

    def _traced(self, pkt, d, logs, anomaly):
        rx = datetime.fromtimestamp(d["rx"], timezone.utc)
        start, spans, total = frame_spans(pkt, rx)
        with _sdk.start_transaction(op="gaze_pipeline", name="gaze_pipeline", start_timestamp=start) as tx:
            tx.set_tag("run", d["run"])
            tx.set_tag("anomaly", str(anomaly).lower())
            tx.set_data("frame", d["frame"])
            tx.set_data("obj", pkt.get("obj"))
            tx.set_data("conf", pkt.get("conf"))
            tx.set_data("fps", d.get("fps"))
            if spans:
                tx.set_data("total_ms", total)
            for stage, s, e in spans:
                sp = tx.start_child(op=stage, description=f"pipeline {stage}", start_timestamp=s)
                sp.finish(end_timestamp=e)
            self._emit(logs, d["run"], d["frame"])          # inside the transaction: the logs link to this trace
            tx.finish(end_timestamp=rx)


# ------------------------------------------------------------------ OMNI / voice loop helpers
@contextlib.contextmanager
def transaction(name, op):
    """A transaction around one voice request (yields None when Sentry is off)."""
    if _sdk is None:
        yield None
        return
    with _sdk.start_transaction(name=name, op=op) as tx:
        yield tx


@contextlib.contextmanager
def span(parent, op, description, **data):
    """A child span of `parent` (a transaction/span, or None). Works from any thread; no-op when Sentry is off."""
    if _sdk is None or parent is None:
        yield None
        return
    with parent.start_child(op=op, description=description) as sp:
        for k, v in data.items():
            sp.set_data(k, v)
        yield sp


def drop(tx):
    """Do not send this transaction (e.g. a clip that turned out to be noise)."""
    if tx is not None:
        tx.sampled = False


def log_info(msg, **attrs):
    if _sdk is not None:
        _sdk.logger.info(msg, attributes=attrs)


def flush(timeout=3):
    if _sdk is not None:
        _sdk.flush(timeout=timeout)
