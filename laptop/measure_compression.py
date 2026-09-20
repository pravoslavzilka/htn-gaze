"""Measure Tiger/TimescaleDB compression on a scratch COPY of the real gaze_frames rows (the live table is not touched),
for a few segmentby/orderby settings, and optionally enable the best one on the live table plus a compression policy.

  python measure_compression.py            # measure only
  python measure_compression.py --apply    # also: ALTER TABLE gaze_frames SET (compress...) + add_compression_policy

The ratio depends on the data: bigger tables amortise the per-chunk metadata, and noisy float timings compress worse than
counters. Quote the number printed here together with the row count, never a round "90%".
"""
import argparse

import psycopg

import config

CONFIGS = {
    "segmentby run, orderby time desc": ("run", "time desc"),
    "segmentby run+obj, orderby time desc": ("run, obj", "time desc"),
    "segmentby run, orderby time asc": ("run", "time asc"),
}


def measure(c, segby, orderby):
    c.execute("drop table if exists zz_compress_test")
    c.execute("create table zz_compress_test (like gaze_frames including defaults)")
    c.execute("select create_hypertable('zz_compress_test','time', chunk_time_interval => interval '7 days')")
    c.execute("insert into zz_compress_test select * from gaze_frames")
    n = c.execute("select count(*) from zz_compress_test").fetchone()[0]
    c.execute(f"alter table zz_compress_test set (timescaledb.compress, timescaledb.compress_segmentby = '{segby}', "
              f"timescaledb.compress_orderby = '{orderby}')")
    for (ch,) in c.execute("select format('%I.%I', chunk_schema, chunk_name) from timescaledb_information.chunks "
                           "where hypertable_name = 'zz_compress_test'").fetchall():
        c.execute("select compress_chunk(%s)", (ch,))
    b, a = c.execute("select sum(before_compression_total_bytes), sum(after_compression_total_bytes) "
                     "from chunk_compression_stats('zz_compress_test')").fetchone()
    c.execute("drop table zz_compress_test")
    return n, int(b), int(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    c = psycopg.connect(config.TIGER_DSN, autocommit=True, prepare_threshold=None)
    best = None
    for name, (seg, order) in CONFIGS.items():
        n, b, af = measure(c, seg, order)
        print(f"{name:<40} {n} rows: {b / 1024:7.0f} kB -> {af / 1024:6.0f} kB   {100 * (1 - af / b):5.1f}% smaller")
        if best is None or af < best[1]:
            best = ((seg, order), af, name)
    print(f"best: {best[2]}")
    if a.apply:
        seg, order = best[0]
        c.execute(f"alter table gaze_frames set (timescaledb.compress, timescaledb.compress_segmentby = '{seg}', "
                  f"timescaledb.compress_orderby = '{order}')")
        c.execute("select add_compression_policy('gaze_frames', compress_after => interval '1 hour', if_not_exists => true)")
        print("applied to gaze_frames: compression settings + policy (chunks older than 1 hour after their end are compressed)")


if __name__ == "__main__":
    main()
