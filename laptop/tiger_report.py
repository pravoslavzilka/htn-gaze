"""What is in Tiger right now: row counts, attention per object (from the continuous aggregate), OMNI history.

  python tiger_report.py [--run live] [--seconds 60]
"""
import argparse

import psycopg

import config

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="only this run (default: all)")
ap.add_argument("--seconds", type=int, default=60)
a = ap.parse_args()
run_filter, params = ("AND run = %s", [a.run]) if a.run else ("", [])

with psycopg.connect(config.TIGER_DSN, connect_timeout=10, autocommit=True) as c:
    print("runs (gaze_frames):")
    for r in c.execute(f"select run, count(*), min(time), max(time) from gaze_frames {'where run=%s' if a.run else ''} group by run order by 3", params):
        print(f"  {r[0]:<14} {r[1]:>7} frames   {r[2]:%H:%M:%S} -> {r[3]:%H:%M:%S} UTC")
    print("\nlatency per stage, frames with timings (avg ms; scene also p95) and tracking loss (conf < 0.5):")
    q = """select run, count(*), count(scene_ms), round(avg(cap_ms)::numeric,2), round(avg(pupil_ms)::numeric,2),
                  round(avg(gaze_ms)::numeric,2), round(avg(scene_ms)::numeric,2), round(avg(fix_ms)::numeric,2),
                  round((percentile_cont(0.95) within group (order by scene_ms))::numeric,2),
                  round(avg((conf < 0.5)::int)::numeric,2)
           from gaze_frames {w} group by run order by min(time)"""
    for run, n, nt, cap, pup, gz, sc, fx, sc95, loss in c.execute(q.format(w="where run=%s" if a.run else ""), params):
        if nt:
            print(f"  {run:<14} n={n:<6} cap {cap} | pupil {pup} | gaze {gz} | scene {sc} (p95 {sc95}) | fix {fx}   loss {loss}")
        else:
            print(f"  {run:<14} n={n:<6} no stage timings recorded (older app build)   loss {loss}")
    print(f"\nattention, last {a.seconds}s (continuous aggregate gaze_10s, includes the newest unmaterialized rows):")
    rows = c.execute(f"""select coalesce(obj, '(no object)') o, sum(frames) f, round(avg(avg_total_ms)::numeric, 1), round(avg(tracking_loss_rate)::numeric, 2)
                         from gaze_10s where bucket > now() - make_interval(secs => %s) {run_filter} group by 1 order by 2 desc""",
                     [a.seconds] + params).fetchall()
    tot = sum(r[1] for r in rows) or 1
    for o, f, lat, loss in rows:
        print(f"  {o:<22} {int(f):>6} frames  {100 * f / tot:5.1f}%   latency {lat if lat is not None else 'n/a'} ms   tracking-loss {loss}")
    if not rows:
        print("  (nothing in that window)")
    print("\nOMNI interactions (latest 8):")
    for r in c.execute(f"select time, run, object, transcript, answer, latency_ms from omni_interactions {'where run=%s' if a.run else ''} order by time desc limit 8", params):
        print(f"  {r[0]:%H:%M:%S} [{r[1]}] on {r[2] or '-'}: {r[3]!r} -> {r[4]!r} ({r[5]:.0f} ms)")
    print("\nsound library (created by OMNI; blue/green/yellow/red/orange are protected and never appear here):")
    for name, prompt, secs, plays, by, at in c.execute("select name, prompt, seconds, plays, created_by, updated_at from sound_library order by name"):
        print(f"  {name:<12} {secs:.1f}s  played {plays}x  by {by}  {at:%H:%M:%S} UTC  {prompt!r}")
