#!/usr/bin/env python3
"""Outer vision: world camera -> objects -> gaze target -> dwell -> note events, plus Maestro (blink menu + OMNI).

Examples:
  python run.py --source synthetic --gaze synthetic                  # no hardware at all
  python run.py --source 0 --gaze mouse                              # webcam, mouse = gaze, keys 1/2/3/x = blinks
  python run.py --source recordings/X/world.mp4 --gaze replay:recordings/X/log.jsonl
  python run.py --source qnx --gaze qnx                              # the QNX rig: scene camera + gaze
  python run.py --source qnx:192.168.2.2 --gaze qnx --stream 8080    # ...and serve the overlay to a browser
  python run.py --source http://192.168.2.2:8081/stream --gaze udp   # any world camera + gaze over UDP

Keys: q quit | 1/2/3 long blink left/right/both | x double blink | m colour view | f features | r record
      c depth-calibrate | p pause | s snapshot
Env (or .env): OMNI_API_KEY (Maestro's decisions), ELEVENLABS_API_KEY (Maestro's voice), SENTRY_DSN (optional)
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import cv2

from outer_vision import config, overlay, shape_net, telemetry
from outer_vision.detector import Detector
from outer_vision.eleven import Eleven
from outer_vision.io import CalibMarker, MjpegServer, MouseGaze, Publisher, Recorder, open_gaze, open_source
from outer_vision.menu import Menu
from outer_vision.music import Music
from outer_vision.omni import Assistant
from outer_vision.selector import Selector
from outer_vision.usercal import ManualOffset, UserCal
from outer_vision.whitebal import AutoWB
from outer_vision.tracker import Tracker, estimate_depth, reference_sizes
from outer_vision.voice import Voice

WIN = "outer-vision"
last_lock_info = {}
_status_last = [0.0]


def now_status_due(period=0.15):
    t = time.monotonic()
    if t - _status_last[0] >= period:
        _status_last[0] = t
        return True
    return False

KEY_BLINKS = {ord("1"): "left", ord("2"): "right", ord("3"): "both"}


def obj_json(t, depth, w, h):
    x, y, bw, bh = t.det.bbox
    dist, vol = depth.get(t.id, (None, None))
    return {
        "id": t.id, "color": t.color, "shape": t.shape,
        "x": round(t.cx / w, 4), "y": round(t.cy / h, 4),
        "bbox": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
        "distance_cm": None if dist is None else round(dist, 1),
        "volume": None if vol is None else round(vol, 3),
        "partial": t.det.partial,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index | video file | 'synthetic' | 'qnx[:host]' "
                                                  "(the QNX board's scene camera) | 'picam[:N]' | MJPEG URL")
    ap.add_argument("--gaze", default="mouse", help="mouse | qnx[:host] (poll the QNX board directly) | udp "
                                                    "(any tracker, see INTERFACE.md) | synthetic | replay:<log.jsonl> | none")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--send", default=None, help="host:port for events (default from config)")
    ap.add_argument("--record", action="store_true", help="start recording immediately")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=0, help="headless: save annotated frame every N frames")
    ap.add_argument("--calib-marker", action="store_true", help="report ArUco markers for gaze calibration")
    ap.add_argument("--ref-distance", type=float, default=60.0, help="cm, used by the 'c' depth calibration")
    ap.add_argument("--realtime", action="store_true", help="pace file/synthetic sources to their fps")
    ap.add_argument("--stream", type=int, default=0, help="serve the debug overlay as MJPEG on this port")
    ap.add_argument("--no-net", action="store_true", help="ignore the shape net; contour rules only")
    ap.add_argument("--offline", action="store_true", help="never call OMNI; blink commands use the built-in defaults")
    ap.add_argument("--no-audio", action="store_true", help="no speaker: menu prompts and replies are shown, not spoken")
    ap.add_argument("--status-file", default=None, help="write a small JSON of what it sees / is looking at / last played (for the control page)")
    ap.add_argument("--user-cal", default="user_calibration.json", help="per-user gaze correction written by the control page calibration (re-read when it changes)")
    ap.add_argument("--manual-offset", default="manual_offset.json", help="hand-set pointer shift written by the control page (re-read when it changes)")
    ap.add_argument("--no-menu", action="store_true", help="ignore blink gestures: no blink menu, colours -> notes only")
    args = ap.parse_args()

    config.load_env()
    cfg = config.load(args.config)
    if args.no_audio:
        cfg["omni"]["speak_with"] = "none"
    if args.no_net:
        cfg["shape_net"]["enabled"] = False
    tele = telemetry.init()
    src = open_source(args.source, cfg)
    gaze = open_gaze(args.gaze, cfg)
    net = shape_net.load(cfg)
    det, trk, sel = Detector(cfg, net), Tracker(cfg), Selector(cfg)
    music = Music(cfg)
    backend = net.backend if net else "rules"
    host, port = (args.send.split(":") if args.send else (cfg["events"]["host"], cfg["events"]["port"]))
    pub = Publisher(host, int(port))
    rec = Recorder(fps=src.fps, cfg=cfg) if args.record else None
    stream = MjpegServer(args.stream) if args.stream else None
    marker = CalibMarker(cfg["calib_marker"]["dictionary"]) if args.calib_marker else None
    print(f"[shape] {backend}  [sentry] {'on' if tele else 'off'}", flush=True)
    if stream:
        print(f"[stream] http://0.0.0.0:{args.stream}/", flush=True)

    # ---- Maestro: blink menu -> OMNI decides (offline defaults if unreachable) -> speech. No microphone.
    log = lambda m: print(m, flush=True)  # noqa: E731
    latest = {}                                   # newest scene, read by the assistant thread
    recent = collections.deque(maxlen=12)         # notes just played, so OMNI can judge "faster"/"slower"
    player = None
    if not args.no_audio:
        try:
            from outer_vision.audio import Player
            player = Player()
        except Exception as e:                   # no output device: prompts fall back to local TTS
            log(f"[audio] no speaker ({e})")
    voice = Voice(player, Eleven(cfg), log=log)

    def context():
        s = dict(latest)
        voices = {tid: music.voice_of(o["color"], o["shape"]) for tid, o in s["objects"].items()}
        img = overlay.draw(s["frame"], s["tracks"], s["depth"], cfg, s["gaze_px"], sel, [], voices=voices)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        now = time.monotonic()
        return {"jpeg": jpg.tobytes(), "objects": {tid: {**o, **voices[tid]} for tid, o in s["objects"].items()},
                "selector": sel,
                "recent": [{**{k: v for k, v in r.items() if k != "at"}, "ago_s": round(now - r["at"], 1)} for r in recent]}

    assistant = Assistant(cfg, music, context, player, pub.send, log=log, voice=voice)
    if args.offline:
        assistant.client.key = ""

    def apply_local(action):                      # swap / stop lesson: nothing for OMNI to decide
        r = music.apply(action, latest.get("objects", {}), sel)
        log(f"[menu] {action} -> {r}")
        return r

    menu = Menu(cfg, (lambda key: None) if args.no_audio else voice.prompt, assistant.submit, apply_local)
    speaks = cfg["omni"]["speak_with"]
    if speaks != "none" and speaks != "omni" and not voice.eleven.ok:
        speaks = "local (no ELEVENLABS_API_KEY)"
    print(f"[maestro] {'offline defaults' if not assistant.client.key else assistant.client.model + ' via ' + assistant.client.url}"
          f"  [voice] {speaks}, {len(voice.clips)} prompt clips", flush=True)

    show = not args.headless
    view_mask = show_feat = paused = False
    flash_id, flash_until = None, 0.0
    snap_dir = Path("recordings/snapshots")
    fps_ema, last_wall = 0.0, time.monotonic()
    frame = t = idx = None
    locks = 0
    cap_ms = gaze_ms = scene_ms = fix_ms = 0.0   # per-stage timings, reported in --status-file
    usercal = UserCal(args.user_cal)
    moff = ManualOffset(args.manual_offset)
    wb = AutoWB() if cfg.get("white_balance", {}).get("enabled") else None
    key_gestures = []

    if show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        if not paused or frame is None:
            t_cap0 = time.perf_counter()
            raw, t, idx = src.read()
            if raw is None:
                break
            scale = cfg["process_width"] / raw.shape[1]
            frame = raw if abs(scale - 1) < 1e-3 else cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cap_ms = (time.perf_counter() - t_cap0) * 1000
        h, w = frame.shape[:2]
        if show and isinstance(gaze, MouseGaze):
            cv2.setMouseCallback(WIN, gaze.on_mouse, (w, h))

        t0 = time.perf_counter()
        with telemetry.frame_trace(idx) as tx:
            _s0 = time.perf_counter()
            with telemetry.span(tx, "detect", n_colors=len(cfg["colors"])):
                dets = det.detect(wb.apply(frame) if wb is not None else frame)
            with telemetry.span(tx, "track"):
                tracks = trk.update(dets, t, w)
            scene_ms = (time.perf_counter() - _s0) * 1000
            _g0 = time.perf_counter()
            g_raw = gaze.get(idx)
            g = moff.apply(usercal.apply(g_raw))
            if g is not None:
                g = (min(1.0, max(0.0, g[0])), min(1.0, max(0.0, g[1])))
            gaze_px = (g[0] * w, g[1] * h) if g is not None else None
            closed = gaze.eyes_closed(idx)
            gaze_ms = (time.perf_counter() - _g0) * 1000
            _f0 = time.perf_counter()
            with telemetry.span(tx, "select"):
                if closed:                      # blinking: keep the target and freeze its dwell
                    sel.hold(t)
                    events = []
                else:
                    events = sel.update(tracks, gaze_px, t, w)
            fix_ms = (time.perf_counter() - _f0) * 1000
            depth = {tr.id: estimate_depth(tr, cfg, w) for tr in tracks}
            markers = marker.detect(frame) if marker else []
            proc_ms = (time.perf_counter() - t0) * 1000
            if tx is not None:
                tx.set_data("proc_ms", proc_ms)
                tx.set_data("objects", len(tracks))

        objs = {tr.id: obj_json(tr, depth, w, h) for tr in tracks}
        latest.update(frame=frame, tracks=tracks, depth=depth, gaze_px=gaze_px, objects=objs)

        # ---- blink gestures drive the Maestro menu (the object under gaze is what "this one" means)
        gestures = gaze.gestures(idx) + key_gestures
        if args.no_menu:
            gestures = []
        key_gestures = []
        focus = sel.target_id if sel.target_id in objs else None
        was_open = menu.active
        for gs in gestures:
            log(f"[blink] {gs['kind']} {gs.get('side', '')} on {focus}  menu={menu.state}")
            telemetry.log("gesture", kind=gs["kind"], side=gs.get("side"), menu=str(menu.state))
            pub.send({"type": "gesture", **gs, "focus": focus})
            menu.on_gesture(gs, focus, music.lesson is not None, time.monotonic())
        menu.tick(time.monotonic())
        if was_open and not menu.active:
            sel.locked = True                   # don't play the object you were answering on; look away first
        voices = {tid: music.voice_of(o["color"], o["shape"]) for tid, o in objs.items()}

        # ---- locks = notes. None while a menu is open: the eyes are answering, not playing.
        out_events = []
        for e in events:
            if menu.active:
                continue
            o = {**objs.get(e["id"], {}), **voices.get(e["id"], {})}
            e = {**e, "t": round(t, 4), "object": o}
            fb = music.on_lock(o)
            if fb is not None:
                e["lesson"] = fb
            out_events.append(e)
            pub.send(e)
            locks += 1
            recent.append({"note": o.get("note"), "instrument": o.get("instrument"), "best_guess": e["best_guess"],
                           "lesson_correct": None if fb is None else fb["correct"], "at": time.monotonic()})
            flash_id, flash_until = e["id"], time.monotonic() + 0.25
            telemetry.log("lock", note=o.get("note"), instrument=o.get("instrument"), best_guess=e["best_guess"])
            last_lock_info["v"] = {"color": o.get("color"), "shape": o.get("shape"), "note": o.get("note"),
                                   "instrument": o.get("instrument"), "at": time.time(), "best_guess": e["best_guess"]}
            print(f"[lock] #{e['id']} {o.get('color')} {o.get('shape')} -> {o.get('note')} {o.get('instrument')}"
                  f"{' (best guess)' if e['best_guess'] else ''}"
                  f"{'' if fb is None else ('  lesson ' + ('✓' if fb['correct'] else '✗ want ' + fb['expected']))}"
                  f" t={t:.2f}", flush=True)

        mstate = music.state()
        state = {
            "type": "state", "t": round(t, 4), "frame": idx,
            "gaze": None if g is None else [round(g[0], 4), round(g[1], 4)],
            "eyes_closed": closed,
            "target": sel.target_id, "dwell": round(sel.progress, 3), "best_guess": sel.best_guess,
            "dwell_s": sel.p["dwell_s"],
            "objects": [{**o, **voices[tid]} for tid, o in objs.items()],
            "lesson": mstate["lesson"],
            "menu": None if not menu.active else {"state": menu.state, "focus": menu.focus, "options": menu.options()},
            "assistant": {"status": assistant.status, "caption": assistant.caption, **assistant.last_metrics},
            "health": {
                "fps": round(fps_ema, 1), "proc_ms": round(proc_ms, 1), "shape": backend,
                "rejected": det.rejected,
                "gaze_age_ms": gaze.age_ms() if hasattr(gaze, "age_ms") else None,
                **({"tracker": gaze.health()} if hasattr(gaze, "health") else {}),
            },
        }
        if marker:
            state["markers"] = [{k: m[k] for k in ("id", "x", "y")} for m in markers]
        if args.status_file and now_status_due():
            tgt = next((tr for tr in tracks if tr.id == sel.target_id), None)
            dist_px = None
            if tgt is not None and gaze_px is not None:
                dist_px = round(max(0.0, -cv2.pointPolygonTest(tgt.det.contour, (float(gaze_px[0]), float(gaze_px[1])), True)), 1)
            status_out = {
                "t": time.time(), "fps": round(fps_ema, 1), "frame_w": w, "frame_h": h,
                "gaze_px": None if gaze_px is None else [round(gaze_px[0]), round(gaze_px[1])],
                "gaze_raw": None if g_raw is None else [round(g_raw[0], 4), round(g_raw[1], 4)],
                "calibrated": usercal.A is not None,
                "offset": [round(moff.dx, 4), round(moff.dy, 4)],
                "wb_gains": None if wb is None else [round(float(x), 2) for x in wb.gains],
                "eyes_closed": bool(closed),
                "seeing": [{"id": tid, "color": o["color"], "shape": o["shape"], "note": voices[tid].get("note"),
                            "instrument": voices[tid].get("instrument"), "area_pct": round(100 * o["bbox"][2] * o["bbox"][3], 1),
                            "cx": round(o["x"] * w), "cy": round(o["y"] * h)} for tid, o in objs.items()],
                "target": None if tgt is None else {"id": tgt.id, "color": tgt.color, "shape": tgt.shape,
                                                    "note": voices[tgt.id].get("note"), "instrument": voices[tgt.id].get("instrument"),
                                                    "dist_px": dist_px, "dwell": round(sel.progress, 2)},
                "stage_ms": {"cap": round(cap_ms, 2), "pupil": (gaze.health() if hasattr(gaze, "health") else {}).get("model_ms"),
                             "gaze": round(gaze_ms, 2), "scene": round(scene_ms, 2), "fix": round(fix_ms, 2)},
                "proc_ms": round(proc_ms, 2),
                "tracker": gaze.health() if hasattr(gaze, "health") else None,
                "locks": locks,
                "last_lock": last_lock_info.get("v"),
            }
            try:
                Path(args.status_file).write_text(json.dumps(status_out), encoding="utf-8")
            except OSError:
                pass
        pub.send(state)
        if rec:
            rec.write(frame, {"t": state["t"], "gaze": state["gaze"], "eyes_closed": closed, "gestures": gestures,
                              "target": sel.target_id, "dwell": state["dwell"], "events": out_events})

        now = time.monotonic()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(now - last_wall, 1e-6))
        last_wall = now
        want_snap = args.headless and args.snapshot_every and idx % args.snapshot_every == 0
        want_stream = stream is not None and stream.wants_frame()
        if show or want_snap or want_stream:
            tr = gaze.health() if hasattr(gaze, "health") else None
            link = "" if tr is None else (
                f"  board:{'up' if tr['connected'] else 'DOWN'} {tr['infer_fps']:.0f}fps eyes:{tr['n']} pupil:{tr['pupil_ok']}")
            hud = [f"{fps_ema:4.1f} fps  proc {proc_ms:4.1f} ms  shape:{backend}  gaze:{gaze.name}{link}  "
                   f"objs:{len(tracks)}  rejected:{det.rejected}  locks:{locks}  dwell {sel.p['dwell_s']:.2f}s",
                   ("REC " if rec else "") + ("PAUSED " if paused else "") + ("EYES CLOSED " if closed else "") +
                   (f"depth ref {cfg['depth']['ref_distance_cm']}cm" if cfg['depth']['ref_distance_cm'] else "depth: uncalibrated (c)")
                   + (f"   lesson: {mstate['lesson']['title']} {mstate['lesson']['index']}/{len(mstate['lesson']['notes'])}"
                      if mstate["lesson"] else "")]
            next_id = None
            if mstate["lesson"]:
                next_id = next((tid for tid, v in voices.items() if v["note"] == mstate["lesson"]["next"]), None)
            if menu.active:
                opts = menu.options()
                caption = ("menu", f"L: {opts['left']}   R: {opts['right']}   both: {opts['both']}   (double blink = cancel)")
            else:
                caption = (assistant.status, assistant.caption)
            img = overlay.draw(frame, tracks, depth, cfg, gaze_px, sel, hud,
                               flash_id if now < flash_until else None, markers, show_feat,
                               voices=voices, next_id=next_id, menu_focus=menu.focus, caption=caption, t=now)
            if view_mask:
                img = overlay.mask_view(det.label_map(frame), det.names, cfg)
            if want_stream:
                stream.publish(img)
            if want_snap:
                snap_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap_dir / f"frame_{idx:05d}.png"), img)
        if show:
            cv2.imshow(WIN, img)
            wait = 1
            if args.realtime and not src.live:
                wait = max(1, int(1000 / src.fps - (time.monotonic() - now) * 1000))
            k = cv2.waitKey(wait) & 0xFF
            if k == ord("q"):
                break
            elif k in KEY_BLINKS:                  # stand-ins for blink gestures (testing without the eye tracker)
                key_gestures.append({"kind": "long", "side": KEY_BLINKS[k], "t": round(t, 3), "source": "key"})
            elif k == ord("x"):
                key_gestures.append({"kind": "double", "t": round(t, 3), "source": "key"})
            elif k == ord("m"):
                view_mask = not view_mask
            elif k == ord("f"):
                show_feat = not show_feat
            elif k == ord("p"):
                paused = not paused
            elif k == ord("s"):
                snap_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap_dir / f"snap_{time.strftime('%H%M%S')}.png"), img)
            elif k == ord("r"):
                if rec:
                    rec.close()
                    print(f"[rec] saved {rec.dir}", flush=True)
                    rec = None
                else:
                    rec = Recorder(fps=src.fps, cfg=cfg)
                    print(f"[rec] recording to {rec.dir}", flush=True)
            elif k == ord("c"):
                cfg["depth"]["ref_distance_cm"] = args.ref_distance
                cfg["depth"]["ref_size"] = reference_sizes(tracks, w)
                config.save(cfg, args.config)
                print(f"[depth] reference at {args.ref_distance} cm: {cfg['depth']['ref_size']} -> {args.config}", flush=True)
        elif args.realtime and not src.live:
            time.sleep(max(0.0, 1.0 / src.fps - (time.monotonic() - now)))
        if args.max_frames and idx + 1 >= args.max_frames:
            break

    if rec:
        rec.close()
        print(f"[rec] saved {rec.dir}")
    if hasattr(src, "close"):
        src.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
