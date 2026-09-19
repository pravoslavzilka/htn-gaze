# QNX / Raspberry Pi 5 side: camera streamer with face mesh and iris tracking

Code that runs **on the Raspberry Pi 5 under QNX 8.0** for an eye-tracking rig:
two Raspberry Pi Camera Module 3 cameras (an *eye* camera and a *scene* camera) are served over HTTP,
and the eye camera also runs MediaPipe Face Mesh (TFLite) to give eyelid and iris landmarks.
The laptop-side calibration and gaze mapping live in `../calib/`.

```
Pi camera (CSI) -> QNX sensor framework -> camera_streamer -> HTTP :8080 (eye), :8081 (scene) -> laptop
```

## Layout (mirrors `~/gazecomp` on the board)

| Path | What |
|---|---|
| `camera_streamer.c` | Main program: opens a camera unit with libcamapi, converts NV12 to rotated RGB, cuts full-resolution eye crops, starts the threads and the HTTP server |
| `http_server.c` | HTTP endpoints (`/api/state`, `/api/frame.jpg`, `/stream.mjpg`, `/api/eyepack`, `/api/focus`) |
| `process.c` | Two threads: JPEG encoding with the pupil overlay, and face-mesh inference |
| `face_lm.cpp`, `face_lm.h` | MediaPipe Face Mesh v2 on TFLite: face detector, 478 landmarks, eyelids, iris ring |
| `jpeg_enc.c`, `jpeg_enc.h` | Baseline JPEG encoder |
| `app_state.h` | Shared state between the threads |
| `include/camera/camera_api.h` | Minimal QNX camera API header (the board has no SDP headers) |
| `models/` | MediaPipe models (`face_detector`, `face_landmarks_detector`, `iris_landmark` `.tflite`, `face_landmarker.task`) |
| `third_party/tflite/` | TensorFlow Lite headers and prebuilt QNX aarch64 shared libraries (about 8 MB) |
| `scripts/` | `build.sh`, `start_streamers.sh`, `health.sh` |
| `sdcard/` | Boot-partition files (`config.txt`, `network`, `wpa_supplicant.conf.example`) |
| `Makefile`, `build_and_run.sh`, `probe.c` | Original build helpers and a small camera probe, kept as found |

## Deploy and build (on the board, no SDP needed)

```sh
# from the laptop: copy this folder's contents into ~/gazecomp on the board
scp -r ./* qnxuser@<board-ip>:gazecomp/

# on the board
sh ~/gazecomp/scripts/build.sh                 # builds camera_streamer.new (takes a couple of minutes)
slay -f -9 camera_streamer; mv -f ~/gazecomp/camera_streamer.new ~/gazecomp/camera_streamer
sh ~/gazecomp/scripts/start_streamers.sh       # eye camera :8080, scene camera :8081
```

Stop old processes with `slay -f -9 camera_streamer` (a plain `slay` can leave one holding the camera, and the next
`camera_open` then fails with error 16). The streamers do not survive a reboot.

## Streamer options

`--http PORT` `--bind IP` `--width W --height H` (RGB output size, default 960x540 before rotation)
`--unit N` (camera unit; on a Pi 5 the two CSI ports are units 3 and 4)
`--rotate 0|90|180|270` (default 270; 90/270 give a portrait image and swap width/height)
`--no-infer` (video only, skip the face model; use for the scene camera)
`--crop-w W --crop-h H` (full-resolution eye crop size, default 512x384)
`--focus-mode M --focus-step S`, `--vf-width --vf-height --vf-fps` (see known issues)

## HTTP API

| Endpoint | Returns |
|---|---|
| `GET /api/frame.jpg` | Newest JPEG (with the pupil overlay on the eye camera); landmarks in the `X-Gazecomp-State` header |
| `GET /stream.mjpg` | MJPEG stream (browsers may lag; polling `frame.jpg` gives lower delay) |
| `GET /api/state` | JSON: `camera_fps`, `infer_fps`, `n` (eyes found, 0 or 2), `score`, and per eye `lid` (16 points), `iris`, `ring` (4 points), `center`, `off`, `noff` (iris offset from the eye-corner midpoint, divided by eye width). Coordinates are normalized to the rotated RGB frame |
| `GET /api/eyepack` | JPEG mosaic `[left eye | right eye]` cut from the **full-resolution** frame (503 until a face was seen). The `X-Pack` header holds JSON: `fid`, `cur`, `vw`, `vh` (virtual full-resolution frame size), `cw`, `ch` (crop size) and per eye `x0`, `y0` (crop origin), `c`, `w`, `iris`, `ring`, `lid` |
| `GET /api/focus[?mode=N&step=M]` | Focus state / request (not effective on the Camera Module 3 here, see below) |

## SD-card / boot configuration

Files on the FAT boot partition (mounted as `/boot` on the board):

- `config.txt`: `kernel=qnx_sdp.ifs`, `enable_uart=1`. The line `cmdline=startup.txt` is commented out because that file is not on the card.
- `network`: `HOSTNAME=...`; IP comes from DHCP.
- `wpa_supplicant.conf`: copied to `/data/var/etc/settings/` at boot. Use `sdcard/wpa_supplicant.conf.example` as a template and
  set `country=` (needed for the Broadcom chip to use all channels). **Never commit the real SSID/password.**

## Notes and known issues (observed on our board)

- **Fixed focus.** The QNX sensor framework here drives the IMX708 sensor but not its lens motor: `camera_set_manual_focus_step`
  returns error 103 and only one focus mode (1) is reported. Place the eye camera at the distance where the picture is sharp.
- **`--vf-width/--vf-height/--vf-fps`** are passed to `camera_set_vf_property`, but we saw no change in camera rate or size.
- **CPU is the bottleneck.** Both cameras run the ISP on the CPU and `drm-rpi5` uses about one core even with no display.
  Copying the full-resolution NV12 frame on every frame starved the face model (about 4 fps instead of about 30); this was removed.
- **Camera handle value.** `camera_open` can legitimately return handle 0, which the header calls `CAMERA_HANDLE_INVALID`; track validity with a flag.
- **Camera units.** A camera that is not detected at boot leaves its `/dev/sensor/cameraN` missing (reseat the ribbon with the board off).
- **Power.** The board went offline several times with normal temperature and memory (about 50 C, about 7 GB free) just before, which points at
  the power supply or cable rather than load. Use a proper 5 V / 5 A supply. This is our inference, not confirmed.
- `scripts/health.sh` logs temperature (`/dev/thermal`) and free memory to `~/health.log`.

## Provenance

The base program (`GazeComp` camera streamer, face-mesh code, `Makefile`) was already on the board when this work started; its original author
is not recorded here. Changes made during this project: image rotation (`--rotate`), the pupil overlay in `process.c`, `/api/focus`,
`--no-infer`, viewfinder options, removal of the per-frame NV12 copy, the full-resolution eye crops and `/api/eyepack`, and the scripts above.
Models are MediaPipe assets (Apache-2.0); the TFLite libraries are prebuilt binaries as found on the board (check their upstream licences before redistributing).
