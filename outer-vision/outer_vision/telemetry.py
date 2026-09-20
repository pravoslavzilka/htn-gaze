"""Optional Sentry observability: per-frame traces (sampled), lock/assistant logs, errors, profiling.

Enabled only when SENTRY_DSN is set; otherwise every call is a no-op, so the pipeline never depends on it.
What it's for: the pipeline has a latency budget (look -> note ≤ 200 ms, voice reply ≤ ~2 s). Traces show
which stage eats it (colour LUT, shape net, OMNI understand vs. first audio) across real sessions.
"""
from __future__ import annotations

import contextlib
import os

_sdk = None


def init(release="outer-vision", sample_rate=0.05):
    global _sdk
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    import sentry_sdk
    import sentry_sdk.logger  # noqa: F401  (logs API lives in a submodule)
    sentry_sdk.init(dsn=dsn, release=release, traces_sample_rate=1.0, enable_logs=True,
                    profile_session_sample_rate=1.0, profile_lifecycle="trace",
                    environment=os.environ.get("SENTRY_ENV", "dev"))
    _sdk = sentry_sdk
    _sdk._ov_frame_rate = sample_rate
    return True


def enabled():
    return _sdk is not None


@contextlib.contextmanager
def frame_trace(idx: int):
    """Traces 1 in 1/sample_rate frames, so tracing costs ~nothing on the other frames."""
    if _sdk is None or idx % max(1, int(1 / _sdk._ov_frame_rate)) != 0:
        yield None
        return
    with _sdk.start_transaction(op="frame", name="vision frame") as tx:
        yield tx


@contextlib.contextmanager
def span(parent, op: str, **data):
    if _sdk is None:
        yield None
        return
    ctx = parent.start_child(op=op) if parent is not None else _sdk.start_span(op=op)
    with ctx as sp:
        for k, v in data.items():
            sp.set_data(k, v)
        yield sp


def log(msg: str, **attrs):
    if _sdk is not None:
        _sdk.logger.info(msg, attributes=attrs)


def metric(name: str, value: float, unit="millisecond"):
    if _sdk is not None:
        _sdk.set_measurement(name, value, unit)
