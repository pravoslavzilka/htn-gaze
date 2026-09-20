import os
import subprocess
import sys
import unittest
from pathlib import Path

SENDER = str(Path(__file__).resolve().parents[2] / "sender")


def run(code, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("PROFILE", "SENTRY_DSN")}
    e.update(env)
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, r'{SENDER}')\n" + code],
                       capture_output=True, text=True, env=e, timeout=60)
    return r.stdout.strip()


class ProfileFlag(unittest.TestCase):
    CODE = ("import gaze_sender\n"
            "tx = gaze_sender.GazeSender('127.0.0.1', 9, 'x')\n"
            "with tx.frame() as fr: pass\n"
            "print('sdk_loaded=%s profiler=%s' % ('sentry_sdk' in sys.modules, tx.prof))")

    def test_off_by_default_and_the_sdk_is_never_imported(self):
        self.assertEqual(run(self.CODE), "sdk_loaded=False profiler=None")

    def test_profile_1_without_a_dsn_stays_off(self):
        self.assertTrue(run(self.CODE, PROFILE="1").endswith("sdk_loaded=False profiler=None"))

    def test_only_the_exact_value_1_enables_it(self):
        self.assertEqual(run(self.CODE, PROFILE="true", SENTRY_DSN="https://k@o1.ingest.sentry.io/1"),
                         "sdk_loaded=False profiler=None")


class Chunking(unittest.TestCase):
    def test_frames_share_a_transaction_until_the_chunk_time_passes(self):
        sys.path.insert(0, SENDER)
        import contextlib
        import pi_profiling

        events = []

        class Tx:
            def set_tag(self, *a): pass
            def set_data(self, k, v): events.append((k, v))

        class Sdk:
            @contextlib.contextmanager
            def start_transaction(self, **k):
                events.append(("open", k["name"]))
                yield Tx()
                events.append(("close",))

            def flush(self, t): pass

        c = pi_profiling.Chunks(Sdk(), "r", seconds=0.2)
        for _ in range(3):
            with c.frame():
                pass
        self.assertEqual([e[0] for e in events], ["open"])                  # one transaction for the three frames
        import time
        time.sleep(0.25)
        with c.frame():
            pass
        self.assertEqual([e[0] for e in events], ["open", "frames", "seconds", "close", "open"])
        self.assertEqual(events[1], ("frames", 3))
        c.close()


if __name__ == "__main__":
    unittest.main()
