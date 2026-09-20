"""Pi-side profiling, for TEST RUNS ONLY.

    PROFILE=1 SENTRY_DSN=... python <your pipeline>      # needs internet on the Pi

With PROFILE unset this module never imports sentry_sdk and does nothing, so normal mode carries no SDK at all
(tests/test_profiling.py checks that in a clean interpreter).

With PROFILE=1 the SDK is initialised with tracing and the newer *continuous* profiler (docs.sentry.io/platforms/python/
profiling: `profile_session_sample_rate` + `profile_lifecycle="trace"`, sentry-sdk >= 2.24.1). In "trace" lifecycle the profiler
runs while a span is active, so a few seconds of frames are wrapped in ONE transaction ("gaze_chunk") and the profile attaches to it.
(The older transaction-based mode would use `profiles_sample_rate` instead.)
"""
import contextlib
import os
import time


def enabled():
    return os.environ.get("PROFILE") == "1"


class Chunks:
    def __init__(self, sdk, run, seconds):
        self.sdk, self.run, self.seconds = sdk, run, seconds
        self.stack = None
        self.tx = None
        self.t0 = 0.0
        self.frames = 0

    def _open(self):
        self.stack = contextlib.ExitStack()
        self.tx = self.stack.enter_context(self.sdk.start_transaction(op="gaze_chunk", name="gaze_chunk"))
        self.tx.set_tag("run", str(self.run))
        self.t0, self.frames = time.monotonic(), 0

    def _close(self):
        if self.stack is not None:
            self.tx.set_data("frames", self.frames)
            self.tx.set_data("seconds", round(time.monotonic() - self.t0, 2))
            self.stack.close()                                   # finishes the transaction; the profile is sent with it
            self.stack = self.tx = None

    @contextlib.contextmanager
    def frame(self):
        """Wrap one frame of the pipeline. Frames within `seconds` of each other share one transaction."""
        if self.stack is not None and time.monotonic() - self.t0 >= self.seconds:
            self._close()
        if self.stack is None:
            self._open()
        self.frames += 1
        yield

    def close(self):
        self._close()
        try:
            self.sdk.flush(3)
        except Exception:
            pass


def start(run="run", seconds=None):
    """-> Chunks when PROFILE=1 and a DSN is available, else None (and sentry_sdk is NOT imported)."""
    if not enabled():
        return None
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        print("PROFILE=1 but SENTRY_DSN is not set: profiling stays off", flush=True)
        return None
    import sentry_sdk
    sentry_sdk.init(dsn=dsn, environment=os.environ.get("SENTRY_ENV", "profiling"),
                    traces_sample_rate=1.0,
                    profile_session_sample_rate=1.0, profile_lifecycle="trace",
                    trace_propagation_targets=[], send_default_pii=False,
                    server_name="pi-profiling", debug=os.environ.get("SENTRY_DEBUG") == "1")
    sentry_sdk.set_tag("service", "pi-profiling")
    return Chunks(sentry_sdk, run, float(os.environ.get("PROFILE_CHUNK_S", seconds or 5)))
