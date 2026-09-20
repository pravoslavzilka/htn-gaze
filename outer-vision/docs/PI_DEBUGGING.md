# Raspberry Pi: connection and debugging log

> **Superseded — kept as a record.** This log is from an attempt to run this repo's Python on a Raspberry
> Pi OS board. The rig now runs **QNX 8.0**, the board runs only the eye team's C/C++ `camera_streamer`,
> and `run.py` stays on the laptop and reads the board over HTTP (see README → "The rig"). Nothing below
> applies to the QNX board; the QNX notes live in the eye repo's `pi/README.md`. What is still useful here
> is the Ethernet setup on the Mac side and the power finding in row 4 — the QNX board has shown the same
> symptom, so use a proper 5 V / 5 A supply.
>
> **For how to actually set up and run the rig today, see [SETUP.md](SETUP.md).**

*As of 2026-09-19, ~10:30. Goal: get the code onto the Pi and verify the camera, speed and tests.*

## Status at the time: blocked on login
The network works; SSH password login fails ("Permission denied"), and no SSH key is installed yet.

## Setup as found
| Item | Value |
|---|---|
| Pi | Raspberry Pi 5, **Raspberry Pi OS 13 "Trixie"** (SSH banner: `OpenSSH_10.0p2 Debian-7+deb13u4`) |
| Pi address | `192.168.2.2` (static) · user tried: `aabzakh` (**not confirmed** by whoever imaged the card) |
| Link | Direct Ethernet cable → docking station port **AX88179A** (`en7`), 100 Mbit/s full duplex |
| Mac address on that port | set to **192.168.2.1 / 255.255.255.0**, no router |
| Mac SSH key | `~/.ssh/id_ed25519` (ED25519, `SHA256:XJboalS4axvQ5pgCZGpoSizeSS5TkxbkXSpTzyEQbA0`) |

## Timeline
| # | Problem | Cause | Fix / status |
|---|---|---|---|
| 1 | Pi unreachable at 192.168.2.2 | The Mac's dock port only had a self-assigned `169.254.x.x` address, so there was no route to 192.168.2.x | ✅ Manual IP 192.168.2.1/24 on AX88179A. Ping OK (~0.4 ms), SSH port open |
| 2 | Nothing to authorize with | The Mac had no SSH key | ✅ Generated `~/.ssh/id_ed25519` |
| 3 | `ssh-copy-id` hung / timed out | The Pi went offline right then (see 4) | ✅ Not an SSH problem |
| 4 | Pi vanished from the cable (no ping, ARP incomplete, no IPv6) | **Suspected power brownout → reboot.** Came back 10:21:47 after ~110 s, which looks like a boot | ⚠️ Unresolved: power the Pi 5 from a **5 V / 5 A (27 W)** supply, not the dock/laptop USB-C |
| 5 | SSH password login: "Permission denied" | Wrong username or password (typos are invisible at the prompt) | ⏳ Open: see "Next steps" |

## Next steps (in order)
1. **Confirm the username** with whoever set up the SD card (Permission denied also appears for an unknown user).
2. If there's a monitor + keyboard: micro-HDMI + USB keyboard; if it boots to the desktop, open Terminal →
   `sudo passwd aabzakh` (the first user normally has passwordless sudo).
3. **Most reliable:** flash the **spare 16 GB card** with Raspberry Pi Imager: Raspberry Pi 5, Pi OS 64-bit;
   Edit settings → hostname `gazepi`, user `aabzakh` + new password, **SSH: public-key only** (auto-fills
   `~/.ssh/id_ed25519.pub`). A fresh image won't have the static IP; reach it as `aabzakh@gazepi.local`
   and set 192.168.2.2 again with `nmcli`.
4. Fix power (27 W supply) before running the camera, which draws more current.
5. Once `ssh -o BatchMode=yes aabzakh@192.168.2.2 true` succeeds: `deploy/deploy_pi.sh aabzakh@192.168.2.2`.

## What the deploy does once we're in
- Copies the repo (without `.git`, venvs, recordings, data) to `~/outer-vision`.
- If the Pi has no internet (likely on a direct cable), downloads the OpenCV wheel on the Mac and copies it
  over (`opencv_python_headless-4.10.0.84-cp37-abi3-…aarch64.whl`, 29 MB; one wheel covers every Pi OS Python).
- Creates `.venv` sharing the system `picamera2`/`numpy`, installs OpenCV without touching numpy.
- Checks: camera 0 captures (saves `recordings/pi_camera_check.jpg`, measures fps), unit tests pass, and
  ms per frame of the full pipeline on the Pi.

## Useful commands (on the Mac)
```bash
ping -c 3 192.168.2.2                                   # is the Pi on the cable?
nc -z -G 3 192.168.2.2 22                               # is SSH listening?
arp -a -i en7                                           # "(incomplete)" for .2.2 = Pi not answering at all
ssh -o BatchMode=yes aabzakh@192.168.2.2 true && echo key-ok
ifconfig en7 | grep inet                                # should show 192.168.2.1
```
On the Pi, to confirm undervoltage after a drop: `vcgencmd get_throttled` (anything other than `0x0` means
it has been undervolted or throttled) and `journalctl -b -1 | tail` (the previous boot's last messages).

## Security note
The Pi password was typed into the chat. Change it once you have access (`passwd`), and prefer key-only SSH.
