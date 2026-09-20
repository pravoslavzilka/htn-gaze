import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library as lib  # noqa: E402
from queries import QueryError  # noqa: E402

PCM = np.arange(200, dtype="<i2").tobytes()


class FakeDb:
    def __init__(self):
        self.rows, self.down, self.calls = {}, False, []

    def execute(self, sql, params=()):
        if self.down:
            raise QueryError("down")
        self.calls.append(sql.split()[0])
        if sql.startswith("insert"):
            self.rows[params[0]] = {"audio": params[4], "sr": params[3]}

    def q(self, sql, params=()):
        if self.down:
            raise QueryError("down")
        if "select name" in sql:
            return [(n,) for n in sorted(self.rows)]
        r = self.rows.get(params[0])
        return [(r["audio"], r["sr"])] if r else []


class Protection(unittest.TestCase):
    def test_the_five_colours_are_locked_in_every_spelling(self):
        for n in ("red", "Red", "BLUE", "green", "yellow", "orange", "dark red", "blue-green", " Yellow "):
            self.assertTrue(lib.is_protected(n), n)
        for n in ("pink", "purple", "black", "white", "teal", "gray", "redwood"):
            self.assertFalse(lib.is_protected(n), n)

    def test_save_refuses_before_touching_anything(self):
        d, db = tempfile.mkdtemp(), FakeDb()
        L = lib.Library(db, d)
        with self.assertRaises(lib.ProtectedError):
            L.save("Red", PCM, "x", 2)
        self.assertEqual(db.calls, [])
        self.assertEqual(list(Path(d).iterdir()), [])                     # not even a cache file


class Storage(unittest.TestCase):
    def test_save_get_list_roundtrip_through_tiger(self):
        d, db = tempfile.mkdtemp(), FakeDb()
        L = lib.Library(db, d)
        self.assertEqual(L.save("Pink", PCM, "soft", 2.0), {"name": "pink", "tiger": True})
        pcm, sr, src = L.get("pink")
        self.assertEqual((pcm, sr, src), (PCM, 24000, "tiger"))
        self.assertEqual(L.names(ttl=0), ["pink"])

    def test_tiger_down_keeps_the_sound_and_still_plays_it(self):
        d, db = tempfile.mkdtemp(), FakeDb()
        db.down = True
        L = lib.Library(db, d)
        self.assertEqual(L.save("teal", PCM, "x", 1)["tiger"], False)     # cached locally, reported not stored
        self.assertEqual(L.get("teal")[2], "cache")
        db.down = False
        self.assertEqual(L.sync_local(), 1)                               # uploaded once Tiger is back
        self.assertIn("teal", db.rows)

    def test_protected_files_in_the_cache_are_never_served_or_imported(self):
        d, db = tempfile.mkdtemp(), FakeDb()
        L = lib.Library(db, d)
        L.save("pink", PCM, "x", 1)
        (Path(d) / "red.wav").write_bytes((Path(d) / "pink.wav").read_bytes())          # an old leftover
        import json
        idx = json.loads((Path(d) / "index.json").read_text())
        idx["red"] = {"file": "red.wav", "prompt": "old", "seconds": 1}
        (Path(d) / "index.json").write_text(json.dumps(idx))
        self.assertIsNone(L.get("red"))
        self.assertNotIn("red", L.names(ttl=0))
        db.rows.clear()
        self.assertEqual(L.sync_local(), 1)                               # pink only
        self.assertEqual(list(db.rows), ["pink"])


if __name__ == "__main__":
    unittest.main()
