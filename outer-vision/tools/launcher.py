#!/usr/bin/env python3
"""Control page for the screen-colour / balloon instrument: http://127.0.0.1:8780/

  Run          colour-block window (optional), tone player and the gaze pipeline
  Stop         stops all of it
  Record       saves the eye camera (with its tracking) next to the front camera (with the pointing) to recordings/*.mp4
  Re-tune      measures how the scene camera sees the on-screen colours (block window in front, camera still)

Run:  python tools/launcher.py --host 169.254.96.94
"""
import argparse
import collections
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import screen_colors as SC  # noqa: E402
from outer_vision import usercal  # noqa: E402
PY = sys.executable
HOST = "169.254.96.94"
STATUS_FILE = Path(tempfile.gettempdir()) / "ovn_status.json"
CAL_FILE = ROOT / "user_calibration.json"
OFFSET_FILE = ROOT / "manual_offset.json"
REC_DIR = ROOT / "recordings"

procs = {}
log = collections.deque(maxlen=200)
notes = collections.deque(maxlen=14)
lock = threading.Lock()
status = {"msg": "stopped", "blocks": True}
cal = {"abort": False, "active": False, "target": None, "text": "", "result": None, "i": 0, "n": 0}
rec = {"on": False, "file": None, "frames": 0, "stop": threading.Event()}

PAGE = """<!doctype html><meta charset=utf-8><title>Colour tones</title>
<style>
body{font:16px system-ui;margin:0;background:#111;color:#eee;padding:14px 20px}
h2{margin:4px 0 10px}
button{font-size:18px;padding:10px 22px;margin:0 8px 8px 0;border:0;border-radius:8px;cursor:pointer;color:#fff}
#start{background:#2e9e4f}#stop{background:#c0392b}#tune{background:#444}#cal{background:#1f6fb2}#calreset{background:#555;font-size:14px;padding:6px 12px}#rec{background:#8e44ad}#rec.on{background:#e74c3c;animation:p 1s infinite}
@keyframes p{50%{opacity:.55}}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#1b1b1b;border-radius:10px;padding:10px}
.card h4{margin:0 0 6px;font-weight:600;color:#aaa;font-size:14px}
img{width:100%;border-radius:6px;background:#000;display:block;min-height:120px}
.row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.info{flex:1;min-width:230px;background:#1b1b1b;border-radius:10px;padding:10px 14px}
.info h4{margin:0 0 4px;color:#aaa;font-weight:600;font-size:14px}
.val{font-size:26px;font-weight:700;min-height:34px}
.sw{display:inline-block;width:18px;height:18px;border-radius:50%;vertical-align:-2px;margin-right:6px;border:1px solid #fff5}
.chip{display:inline-block;padding:3px 10px;margin:2px 4px 2px 0;border-radius:14px;background:#2a2a2a;font-size:14px}
pre{background:#000;padding:8px;border-radius:6px;height:150px;overflow:auto;font-size:12px;margin:4px 0}
#st{color:#bbb;margin:2px 0 8px}#offw{margin-top:8px;color:#ccc}.nb{font-size:15px;padding:5px 12px;margin:0 4px 4px 4px;background:#3a3a3a}.sl{display:flex;align-items:center;gap:8px;margin:2px 0}.sl input{flex:1}.sl span{width:70px;text-align:right;color:#9cd;font-size:13px}
</style>
<h2>Colour tones</h2>
<button id=start onclick="go('start')">Run</button><button id=stop onclick="go('stop')">Stop</button>
<button id=cal onclick="go('calibrate')">Calibrate for me</button><button id=rec onclick="go('rec')">Record</button><button id=tune onclick="go('tune')">Re-tune colours</button>
<label style="color:#bbb"><input type=checkbox id=blocks checked onchange="fetch('/blocks?'+(this.checked?1:0),{method:'POST'})"> show colour blocks window</label>
<div id=st></div><div id=calinfo style="margin:4px 0 8px;color:#9cd"></div>
<div class=grid>
 <div class=card><h4>FRONT CAMERA: pointing (red pointer = where you look). Drag on the picture to move the pointer.</h4><img id=front onerror="retry('front')" draggable=false style="cursor:move;user-select:none"><div id=offw><b>Manual offset</b> <button class=nb onclick="nudge(-0.01,0)">&#9664;</button><button class=nb onclick="nudge(0.01,0)">&#9654;</button><button class=nb onclick="nudge(0,-0.01)">&#9650;</button><button class=nb onclick="nudge(0,0.01)">&#9660;</button><button class=nb onclick="setOff(0,0)">reset offset</button><div class=sl>left / right <input type=range id=sx min=-0.3 max=0.3 step=0.002 value=0 oninput="setOff(+this.value,off.dy)"> <span id=vx></span></div><div class=sl>up / down <input type=range id=sy min=-0.3 max=0.3 step=0.002 value=0 oninput="setOff(off.dx,+this.value)"> <span id=vy></span></div></div></div>
 <div class=card><h4>EYE CAMERA: eye tracking</h4><img id=eye onerror="retry('eye')"></div>
</div>
<div class=row>
 <div class=info><h4>SEES</h4><div id=sees></div></div>
 <div class=info><h4>LOOKING AT</h4><div class=val id=look>-</div><div id=lookd style="color:#bbb"></div></div>
 <div class=info><h4>PLAYING</h4><div class=val id=play>-</div><div id=playc style="color:#bbb"></div></div>
</div>
<div class=grid>
 <div class=card><h4>Notes played</h4><pre id=notes></pre></div>
 <div class=card><h4>Log</h4><pre id=log></pre></div>
</div>
<script>
const CSS={red:'#e02b20',yellow:'#e8d21d',green:'#26b34a',blue:'#2f6df0',purple:'#a93fd0',orange:'#f08a20'};
const sw=c=>'<span class=sw style="background:'+(CSS[c]||'#888')+'"></span>';
let viewsOn=false;
let off={dx:0,dy:0},dragging=false,lastSend=0;
function showOff(){sx.value=off.dx;sy.value=off.dy;vx.textContent=Math.round(off.dx*640)+' px';vy.textContent=Math.round(off.dy*360)+' px'}
function setOff(dx,dy,force){off={dx:Math.max(-0.3,Math.min(0.3,dx)),dy:Math.max(-0.3,Math.min(0.3,dy))};showOff();
 const now=Date.now();if(force||now-lastSend>70){lastSend=now;fetch('/offset?dx='+off.dx.toFixed(4)+'&dy='+off.dy.toFixed(4),{method:'POST'})}}
function nudge(a,b){setOff(off.dx+a,off.dy+b,true)}
addEventListener('DOMContentLoaded',()=>{
 front.addEventListener('mousedown',e=>{dragging=true;e.preventDefault()});
 addEventListener('mouseup',()=>{if(dragging){dragging=false;setOff(off.dx,off.dy,true)}});
 addEventListener('mousemove',e=>{if(dragging){const r=front.getBoundingClientRect();setOff(off.dx+e.movementX/r.width,off.dy+e.movementY/r.height)}});
});
async function go(a){await fetch('/'+a,{method:'POST'});if(a=='start')setTimeout(attach,4000)}
const SRC={front:()=>'http://127.0.0.1:8790/stream',eye:()=>'http://__HOST__:8080/stream.mjpg'};
function retry(k){setTimeout(()=>{document.getElementById(k).src=SRC[k]()},1500)}
function attach(){front.src='http://127.0.0.1:8790/stream';eye.src='http://__HOST__:8080/stream.mjpg'}
async function tick(){try{const s=await (await fetch('/status')).json();
 st.textContent=s.msg+(s.running.length?' | '+s.running.join(', '):'');
 rec.classList.toggle('on',s.rec.on);rec.textContent=s.rec.on?('Stop recording ('+s.rec.frames+' frames)'):'Record';
 if(s.running.includes('run')&&!viewsOn){viewsOn=true;attach()} if(!s.running.includes('run'))viewsOn=false;
 const c=s.cal;
 const rb=' <button id=calreset onclick="go(&quot;calreset&quot;)">reset</button>';
 calinfo.innerHTML=c.active?('<b>Calibrating:</b> '+c.text):(c.result?c.result+rb:(s.pipe&&s.pipe.calibrated?'calibrated (saved for this user)'+rb:'not calibrated for you yet: press "Calibrate for me"'));
 document.getElementById('cal').disabled=c.active;
 const p=s.pipe;
 if(p&&p.offset&&!dragging&&Date.now()-lastSend>800){off={dx:p.offset[0],dy:p.offset[1]};showOff()}
 if(p){
  const seen=p.seeing||[];
  sees.innerHTML=seen.length?seen.map(o=>'<span class=chip>'+sw(o.color)+o.color+' '+o.shape+' <b>'+o.note+'</b> ('+o.area_pct+'%)</span>').join(''):'<span style="color:#888">no coloured surface</span>';
  const t=p.target;
  if(p.gaze_px===null)look.innerHTML='<span style="color:#e55">no gaze (eyes not detected)</span>',lookd.textContent='';
  else if(t){look.innerHTML=sw(t.color)+t.color+' '+t.shape;lookd.textContent=(t.dist_px<1?'pointing on it':t.dist_px+' px from it')+' | hold '+Math.round(t.dwell*100)+'%'}
  else{look.textContent='nothing near';lookd.textContent='gaze at '+p.gaze_px.join(', ')}
  const l=p.last_lock;
  if(l){play.innerHTML=sw(l.color)+l.note+' '+l.instrument;playc.textContent='colour: '+l.color+' ('+Math.round(Date.now()/1000-l.at)+' s ago)'+(l.best_guess?' - near, not on it':'')}
 }
 notes.textContent=s.notes.slice().reverse().join('\\n');log.textContent=s.log.join('\\n');log.scrollTop=log.scrollHeight;
}catch(e){}setTimeout(tick,300)}tick();
</script>"""


def pump(name, p):
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        with lock:
            log.append(f"[{name}] {line}")
            if name == "run" and line.startswith("[lock]"):
                notes.append(line[len("[lock] "):].split(" t=")[0])


def spawn(name, args):
    p = subprocess.Popen([PY, "-u"] + args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    procs[name] = p
    threading.Thread(target=pump, args=(name, p), daemon=True).start()


def stop_all():
    stop_rec()
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for p in procs.values():
        try:
            p.wait(3)
        except Exception:
            p.kill()
    procs.clear()
    status["msg"] = "stopped"


def start():
    if cal["active"]:
        return                                   # do not kill the pipeline under a running calibration
    stop_all()
    cfg = "config.laptop.json"
    if not (ROOT / cfg).exists():
        status["msg"] = "no config.laptop.json yet: press Re-tune colours first"
        return
    if status["blocks"]:
        spawn("blocks", ["tools/screen_colors.py", "show", "--port", "8770"])
    spawn("synth", ["tools/synth.py"])
    STATUS_FILE.unlink(missing_ok=True)
    spawn("run", ["run.py", "--source", f"qnx:{HOST}", "--gaze", f"qnx:{HOST}", "--config", cfg, "--offline", "--no-menu", "--no-audio",
                  "--headless", "--stream", "8790", "--status-file", str(STATUS_FILE)])
    status["msg"] = "running: look at a colour block or a balloon"


def tune():
    if cal["active"]:
        return
    stop_all()
    status["msg"] = "tuning colours (keep the block window in front and the camera still)..."
    spawn("tune", ["tools/screen_colors.py", "tune", "--host", HOST, "--port", "8771"])

    def waiter():
        procs["tune"].wait()
        status["msg"] = "tuned: press Run"
    threading.Thread(target=waiter, daemon=True).start()


# ------------------------------------------------------------------ recording: eye tracking + front pointing, side by side
def mjpeg_frames(url, stop):
    """Yield decoded frames from an MJPEG stream (multipart or plain concatenated JPEGs)."""
    buf = b""
    while not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                while not stop.is_set():
                    chunk = r.read1(16384)
                    if not chunk:
                        break
                    buf += chunk
                    while True:                       # decode every complete JPEG, keep the tail
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                        if a < 0 or b < 0:
                            break
                        img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
                        buf = buf[b + 2:]
                        if img is not None:
                            yield img
                    if len(buf) > 4_000_000:
                        buf = b""
        except Exception:
            time.sleep(0.5)


def rec_worker(stop):
    latest = {}

    def reader(key, url):
        for img in mjpeg_frames(url, stop):
            latest[key] = img
    for key, url in (("eye", f"http://{HOST}:8080/stream.mjpg"), ("front", "http://127.0.0.1:8790/stream")):
        threading.Thread(target=reader, args=(key, url), daemon=True).start()
    REC_DIR.mkdir(exist_ok=True)
    path = REC_DIR / time.strftime("rec_%Y%m%d_%H%M%S.mp4")
    rec["file"] = str(path)
    W, H, FPS = 640, 360, 15
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * W, H + 34))
    nxt = time.time()
    while not stop.is_set():
        nxt += 1.0 / FPS
        time.sleep(max(0.0, nxt - time.time()))
        if "front" not in latest and "eye" not in latest:
            continue
        panels = []
        for k in ("front", "eye"):
            im = latest.get(k)
            panels.append(cv2.resize(im, (W, H)) if im is not None else np.zeros((H, W, 3), np.uint8))
        bar = np.zeros((34, 2 * W, 3), np.uint8)
        try:
            p = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            t = p.get("target")
            l = p.get("last_lock")
            txt = "sees: " + ", ".join(f"{o['color']}" for o in p.get("seeing", [])) + \
                  ("   |   looking at: %s (%s px)" % (t["color"], t["dist_px"]) if t else "   |   looking at: -") + \
                  ("   |   last: %s %s" % (l["color"], l["note"]) if l else "")
            cv2.putText(bar, txt[:150], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        vw.write(np.vstack([np.hstack(panels), bar]))
        rec["frames"] += 1
    vw.release()


def start_rec():
    if rec["on"]:
        return
    rec["stop"] = threading.Event()
    rec["frames"] = 0
    rec["on"] = True
    threading.Thread(target=rec_worker, args=(rec["stop"],), daemon=True).start()
    log.append(f"[rec] recording to recordings/")


def stop_rec():
    if rec["on"]:
        rec["stop"].set()
        rec["on"] = False
        log.append(f"[rec] saved {rec['file']} ({rec['frames']} frames)")


# ------------------------------------------------------------------ per-user calibration
CAL_PAGE = """<!doctype html><meta charset=utf-8><title>Calibration</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none;font-family:system-ui}canvas{position:fixed;inset:0;width:100%;height:100%}</style>
<canvas id=c></canvas><script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function rs(){cv.width=innerWidth;cv.height=innerHeight}addEventListener('resize',rs);rs();
addEventListener('click',()=>document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen());
async function tick(){try{const s=await (await fetch('/calstate',{cache:'no-store'})).json();
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 const t=s.target;
 if(t){const side=t.size*cv.height,x=side/2+t.x*(cv.width-side),y=side/2+t.y*(cv.height-side);
  ctx.fillStyle='rgb('+t.rgb.join(',')+')';ctx.fillRect(x-side/2,y-side/2,side,side);
  ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-14,y);ctx.lineTo(x+14,y);ctx.moveTo(x,y-14);ctx.lineTo(x,y+14);ctx.stroke();}
 ctx.fillStyle='#fff';ctx.font='bold '+Math.round(cv.height*0.045)+'px system-ui';ctx.textAlign='center';
 ctx.fillText(s.text,cv.width/2,cv.height*0.93);
}catch(e){}setTimeout(tick,60)}tick();
</script>"""

# anchors over the whole screen: 4 columns x 3 rows, squares flush with the screen edges (0 = edge on the left/top, 1 = edge on the right/bottom)
CAL_POINTS = [(cx, ry) for ry in (0.0, 0.5, 1.0) for cx in ((0.0, 1 / 3, 2 / 3, 1.0) if ry != 0.5 else (1.0, 2 / 3, 1 / 3, 0.0))]
CAL_COLORS = ["red", "yellow", "green", "blue", "purple"] * 3
CAL_RGB = {"red": (235, 30, 30), "yellow": (240, 215, 20), "green": (30, 205, 70), "blue": (55, 90, 240), "purple": (175, 55, 225)}   # bright: the square is found by frame difference, not by colour
CAL_SIZE = 0.18


def _read_status():
    try:
        p = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return p if time.time() - p["t"] < 1.5 else None
    except Exception:
        return None


def _scene_frame(timeout=3):
    raw = urllib.request.urlopen(f"http://{HOST}:8081/api/frame.jpg", timeout=timeout).read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def find_square(frame, base):
    """Centre (normalized) of the square that lit up since the black baseline, or None.
    Independent of the colour classifier. The camera's auto exposure shifts when the screen lights up, so the
    baseline is first scaled to the frame's overall brightness; then any compact, filled, clearly brighter blob counts."""
    f = frame.astype(np.float32)
    b = base.astype(np.float32)
    lum_f, lum_b = f.sum(axis=2), b.sum(axis=2)
    ok = lum_b > 60
    k = float(np.median(lum_f[ok] / lum_b[ok])) if ok.sum() > 1000 else 1.0
    diff = np.clip(f - b * k, 0, 255).max(axis=2)
    mask = (diff > max(45.0, 0.35 * float(np.percentile(diff, 99.9)))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = frame.shape[:2]
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.002 * W * H or area > 0.2 * W * H or not (0.6 < w / h < 1.6) or area / (w * h) < 0.7:
            continue
        if best is None or area > stats[best, cv2.CC_STAT_AREA]:
            best = i
    if best is None:
        return None
    return [float(cent[best][0] / W), float(cent[best][1] / H)]


def calibrate_worker(port):
    try:
        if not any(n == "run" and p.poll() is None for n, p in procs.items()):
            cal["result"] = "start the app with Run first"
            return
        if not _read_status():
            cal["result"] = "no data from the pipeline yet, wait a few seconds after Run"
            return
        cal["active"], cal["result"], cal["abort"] = True, None, False
        cal["n"] = len(CAL_POINTS)
        SC._open_fullscreen(f"http://127.0.0.1:{port}/calpage")
        for i in range(5, 0, -1):
            cal["target"], cal["text"] = None, f"Calibration starts in {i}. Keep your head still, look at each square's centre"
            time.sleep(1)
        base = np.median(np.stack([_scene_frame() for _ in range(3)]), axis=0).astype(np.uint8)   # black screen, nothing lit
        raws, targets = [], []
        far_total = [0]
        fw = fh = None
        for i, ((x, y), col) in enumerate(zip(CAL_POINTS, CAL_COLORS)):
            if cal["abort"]:
                cal["result"] = "calibration cancelled"
                return
            if not any(n == "run" and p.poll() is None for n, p in procs.items()):
                cal["result"] = "calibration stopped: the pipeline is not running any more (was Stop or Run pressed?). Press Run, then calibrate again"
                return
            cal["i"] = i
            cal["target"] = {"x": x, "y": y, "size": CAL_SIZE, "rgb": CAL_RGB[col]}
            cal["text"] = f"Look at the centre of the {col.upper()} square  ({i + 1}/{len(CAL_POINTS)})"
            time.sleep(1.4)                                    # the eyes travel to the square
            g_s, t_s, last_t = [], [], None
            n_nogaze = n_nosq = n_far = 0
            t_end = time.time() + 2.0
            while time.time() < t_end:
                st = _read_status()
                if not st or st["t"] == last_t:
                    time.sleep(0.03)
                    continue
                last_t = st["t"]
                fw, fh = st["frame_w"], st["frame_h"]
                gr = st.get("gaze_raw")
                if st.get("eyes_closed") or gr is None or not (-4.5 < gr[0] < 5.5 and -4.5 < gr[1] < 5.5):
                    n_nogaze += 1
                    continue
                try:
                    sq = find_square(_scene_frame(), base)
                except Exception:
                    sq = None
                if sq is None:
                    n_nosq += 1
                    continue
                n_far += not (-0.5 < gr[0] < 1.5 and -0.5 < gr[1] < 1.5)
                g_s.append(st["gaze_raw"])
                t_s.append(sq)
            far_total[0] += n_far
            if len(g_s) >= 6:
                raws.append(np.median(g_s, axis=0).tolist())
                targets.append(np.median(t_s, axis=0).tolist())
            else:
                log.append(f"[cal] point {i + 1} ({col}): {len(g_s)} usable samples ({n_nogaze} without gaze, {n_nosq} without the square in the scene picture), skipped")
        cal["target"] = None
        cal["text"] = "fitting..."
        if len(raws) < 6:
            cal["result"] = (f"calibration failed: only {len(raws)} of {len(CAL_POINTS)} points had both your gaze and the square seen. "
                             "Check that the eye camera sees your eyes and the scene camera sees the whole screen")
            return
        raws, targets = np.array(raws), np.array(targets)
        res = usercal.fit(raws, targets, fw, fh)
        A = np.array(res["A"])                                   # drop clearly disagreeing points, once
        pred = np.hstack([raws, np.ones((len(raws), 1))]) @ A.T
        err = np.linalg.norm((pred - targets) * np.array([fw, fh]), axis=1)
        keep = err <= max(2.5 * np.median(err), 25.0)
        if keep.sum() >= 6 and not keep.all():
            log.append(f"[cal] dropped {int((~keep).sum())} outlier point(s)")
            res = usercal.fit(raws[keep], targets[keep], fw, fh)
        CAL_FILE.write_text(json.dumps(res), encoding="utf-8")
        OFFSET_FILE.unlink(missing_ok=True)          # a fresh calibration starts with no hand offset
        if far_total[0] > 0.4 * 12 * 13:
            log.append("[cal] note: the raw gaze is far outside the picture for many samples, so the rig geometry does not match this mounting; "
                       "the calibration corrects it, but if it is poor, re-seat the headset the way it was when the rig was fitted")
        cal["result"] = (f"calibrated with {res['n']} points: error {res['before_px']:.0f} px before, {res['after_px']:.0f} px after "
                         f"({res['loo_px']:.0f} px on held-out points, in a {fw}-px-wide picture)")
        log.append("[cal] " + cal["result"])
    except Exception as e:  # noqa: BLE001
        cal["result"] = f"calibration error: {e}"
    finally:
        cal["active"], cal["target"], cal["text"] = False, None, ""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status"):
            pipe = None
            try:
                p = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
                if time.time() - p["t"] < 3:
                    pipe = p
            except Exception:
                pass
            with lock:
                d = {"cal": {k: cal[k] for k in ("active", "text", "result")}, "msg": status["msg"], "running": [n for n, p in procs.items() if p.poll() is None],
                     "notes": list(notes), "log": list(log)[-60:], "pipe": pipe,
                     "rec": {"on": rec["on"], "frames": rec["frames"], "file": rec["file"]}}
            self._send(json.dumps(d).encode())
        elif self.path.startswith("/calstate"):
            self._send(json.dumps({"target": cal["target"], "text": cal["text"] or (cal["result"] or "")}).encode())
        elif self.path.startswith("/calpage"):
            self._send(CAL_PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(PAGE.replace("__HOST__", HOST).encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path.startswith("/blocks"):
            status["blocks"] = self.path.endswith("1")
        elif self.path.startswith("/offset"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            try:
                dx, dy = float(q["dx"][0]), float(q["dy"][0])
                dx, dy = max(-0.3, min(0.3, dx)), max(-0.3, min(0.3, dy))
                if dx == 0.0 and dy == 0.0:
                    OFFSET_FILE.unlink(missing_ok=True)
                else:
                    OFFSET_FILE.write_text(json.dumps({"dx": dx, "dy": dy}), encoding="utf-8")
            except (KeyError, ValueError):
                pass
        elif self.path == "/calibrate":
            if not cal["active"]:
                threading.Thread(target=calibrate_worker, args=(self.server.server_port,), daemon=True).start()
        elif self.path == "/calreset":
            CAL_FILE.unlink(missing_ok=True)
            cal["result"] = "calibration reset"
        elif self.path == "/rec":
            stop_rec() if rec["on"] else start_rec()
        else:
            if self.path == "/stop" and cal["active"]:
                cal["abort"] = True                        # Stop during a calibration cancels it first
            {"/start": start, "/stop": stop_all, "/tune": tune}.get(self.path, lambda: None)()
        self._send(b"{}")


def main():
    global HOST
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    HOST = a.host
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}/"
    print("control page:", url)
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == "__main__":
    main()
