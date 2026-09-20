#!/usr/bin/env python3
"""Find the QNX board on the network and say exactly what is wrong when you can't reach it.

"Just ping 192.168.2.2" is not good enough: on a direct cable there is no DHCP server, so the board
falls back to a link-local 169.254.x.x address of its own choosing. It is then on a different subnet
from the laptop, and every ping times out even though the board is sitting right there.

This looks in every place the board can show up -- the ARP table (by Raspberry Pi MAC prefix), mDNS,
and the addresses the two repos hard-code -- then tells you which of the four things is actually true:

  1. nothing on the cable at all          -> cable, adapter, or the board is off
  2. the board is there but silent        -> almost always power; see the notes it prints
  3. the board answers, wrong subnet      -> it prints the one command that fixes it
  4. the board answers                    -> it prints the config.json line to paste

  python tools/find_board.py
  python tools/find_board.py --watch        # keep looking; use while you power-cycle the board
"""
import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config  # noqa: E402

# Raspberry Pi Trading / Foundation OUIs, lowercase, colon-separated.
PI_OUI = ("88:a2:9e", "2c:cf:67", "d8:3a:dd", "e4:5f:01", "dc:a6:32", "b8:27:eb")
EYE_PORT, SCENE_PORT, SSH_PORT = 8080, 8081, 22


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def interfaces():
    """-> [(name, ip, netmask_bits)] for every up IPv4 interface except loopback."""
    out = []
    cur = None
    for line in sh("ifconfig").splitlines():
        m = re.match(r"^(\w+):", line)
        if m:
            cur = m.group(1)
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+) netmask (0x[0-9a-f]+)", line)
        if m and cur and cur != "lo0":
            bits = bin(int(m.group(2), 16)).count("1")
            out.append((cur, m.group(1), bits))
    return out


def same_subnet(a, b, bits):
    def n(ip):
        p = [int(x) for x in ip.split(".")]
        return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]
    mask = (0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF
    return (n(a) & mask) == (n(b) & mask)


ARP_TIMEOUT_S = 1200        # macOS default; the "Expire" column counts down from it


def arp_age(ip):
    """Roughly how long ago this ARP entry was last confirmed, in seconds, or None if not cached.

    An entry survives 20 minutes after the host stops answering, so a cached MAC is NOT proof that
    anything is there now. macOS will not re-ARP while an entry is still valid, so age is the only
    signal available without root: a fresh entry means the host answered recently, an old one means
    it answered once and may be long gone.
    """
    for line in sh("netstat -rn -f inet").splitlines():
        parts = line.split()
        if parts and parts[0] == ip and parts[-1].isdigit():
            return max(0, ARP_TIMEOUT_S - int(parts[-1]))
    return None


def arp_candidates():
    """Anything in the ARP table with a Raspberry Pi MAC. -> [(ip, mac, iface, age_s)]"""
    found = []
    for line in sh("arp -a").splitlines():
        m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-f:]+) on (\w+)", line, re.I)
        if not m:
            continue
        ip, mac, iface = m.group(1), m.group(2).lower(), m.group(3)
        mac = ":".join(p.zfill(2) for p in mac.split(":"))
        if mac.startswith(PI_OUI):
            found.append((ip, mac, iface, arp_age(ip)))
    return found


def mdns_candidates():
    """.local names that look like the board. -> [(ip, name)]"""
    out, names = [], set()
    for line in sh("dns-sd -t 3 -B _services._dns-sd._udp local").splitlines():
        for w in re.findall(r"[\w-]+\.local", line):
            names.add(w)
    names |= {"qnxpi60.local", "gazepi.local"}
    for name in names:
        if not re.search(r"qnx|gaze|pi\d*\.local", name, re.I):
            continue
        try:
            ip = socket.gethostbyname(name)
            out.append((ip, name))
        except OSError:
            pass
    return out


def probe(ip, timeout=1.5):
    """-> dict of what answers at this address."""
    res = {"ip": ip, "ports": {}}
    for label, port in (("eye", EYE_PORT), ("scene", SCENE_PORT), ("ssh", SSH_PORT)):
        s = socket.socket()
        s.settimeout(timeout)
        res["ports"][label] = s.connect_ex((ip, port)) == 0
        s.close()
    res["ping"] = subprocess.run(["ping", "-c", "1", "-t", "1", ip],
                                 capture_output=True).returncode == 0
    res["alive"] = res["ping"] or any(res["ports"].values())
    if res["ports"]["eye"]:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://{ip}:{EYE_PORT}/api/state", timeout=2) as r:
                res["state"] = json.loads(r.read())
        except Exception as e:
            res["state_error"] = str(e)[:80]
    return res


def scan(cfg):
    ifaces = interfaces()
    print("Your Mac's network interfaces:")
    for name, ip, bits in ifaces:
        print(f"    {name:8s} {ip}/{bits}")
    if not any(n != "en0" for n, _, _ in ifaces):
        print("    (nothing but Wi-Fi -- is the Ethernet adapter plugged in?)")
    print()

    seen, cands = set(), []
    for ip, mac, iface, age in arp_candidates():
        when = "just now" if age is not None and age < 60 else (
            f"{age // 60} min ago" if age is not None else "at some point")
        cands.append((ip, f"a Raspberry Pi MAC ({mac}) answered on {iface} {when}"))
        seen.add(ip)
    for ip, name in mdns_candidates():
        if ip not in seen:
            cands.append((ip, f"mDNS name {name}"))
            seen.add(ip)
    for ip in (cfg["qnx"]["host"], "192.168.2.2", "192.168.127.94"):
        if ip not in seen:
            cands.append((ip, "an address the repos hard-code"))
            seen.add(ip)

    print(f"Checking {len(cands)} possible addresses ({EYE_PORT}=eye camera, {SCENE_PORT}=scene, {SSH_PORT}=ssh):")
    results = []
    for ip, why in cands:
        r = probe(ip)
        r["why"] = why
        results.append(r)
        marks = "".join(f" {k}" for k, v in r["ports"].items() if v) or " nothing"
        print(f"    {ip:<16s} ping {'yes' if r['ping'] else 'no ':<4s} open:{marks:<20s} <- {why}")
    return ifaces, results


def report(ifaces, results, cfg):
    print()
    working = [r for r in results if r["ports"]["eye"]]
    alive = [r for r in results if r["alive"]]
    pi_seen = [r for r in results if "Raspberry Pi MAC" in r["why"]]

    if working:
        r = working[0]
        st = r.get("state", {})
        print(f"FOUND IT. The eye camera is serving on {r['ip']}:{EYE_PORT}.")
        if st:
            print(f"    eyes found: {st.get('n')}   camera {st.get('camera_fps')} fps   "
                  f"infer {st.get('infer_fps')} fps   engine {st.get('engine')}")
            if st.get("n") == 0:
                print("    n=0 means it is running but cannot see a face yet -- reposition the eye camera.")
        if not r["ports"]["scene"]:
            print(f"    The scene camera on :{SCENE_PORT} is NOT up. start_streamers.sh starts both;"
                  " check /tmp/stream_u3.log on the board.")
        print(f'\n    Put this in config.json:   "qnx": {{ "host": "{r["ip"]}", ... }}')
        print(f'    Then:  .venv/bin/python run.py --source qnx --gaze qnx')
        return 0

    if alive:
        r = alive[0]
        print(f"The board is reachable at {r['ip']}, but the cameras are not running.")
        if r["ports"]["ssh"]:
            print("    SSH is open, so start them (this is SETUP.md step 4):")
            print(f"        ssh qnxuser@{r['ip']} 'sh ~/gazecomp/scripts/start_streamers.sh'")
        else:
            print("    SSH is not open either. Get in over the serial console, or ask whoever set the board up.")
        return 1

    if pi_seen:
        r = pi_seen[0]
        ip = r["ip"]
        stale = "min ago" in r["why"] or "at some point" in r["why"]
        if stale:
            print(f"A Raspberry Pi answered at {ip} earlier, but nothing is there NOW.")
            print("    That address came out of the ARP cache, which keeps a MAC for 20 minutes after the")
            print("    host stops replying -- so it is a ghost, not proof the board is up. It will vanish")
            print("    on its own. Treat this as 'the board is not running'.")
        else:
            print(f"A Raspberry Pi is answering at layer 2 on {ip}, but not at IP level.")
        print()
        print("    CHECK THE BOARD'S LIGHTS FIRST. Green blinking = it is running; red only = it has")
        print("    power but never booted, and no amount of network debugging will help.")
        print()
        on_subnet = [(n, i, b) for n, i, b in ifaces if same_subnet(i, ip, b)]
        if not on_subnet:
            guess = next((n for n, i, b in ifaces if n != "en0"), "en7")
            octets = ip.split(".")
            mine = ".".join(octets[:3] + ["1"]) if octets[3] != "1" else ".".join(octets[:3] + ["2"])
            mask = "255.255.0.0" if ip.startswith("169.254.") else "255.255.255.0"
            print(f"    REASON 1 -- your Mac has no address on {ip}'s subnet, so the board cannot reply")
            print(f"    to anything you send. Give this Mac an address there (temporary, gone on reboot):")
            print(f"        sudo ifconfig {guess} alias {mine} {mask}")
            print(f"        ping -c 3 {ip}")
            if ip.startswith("169.254."):
                print("    A 169.254.x.x address means the board found no DHCP server and picked its own.")
                print("    That is normal on a direct cable. Giving the Mac a 169.254 address is the fix.")
            print()
        print("    REASON 2 -- the board is not booting. The Ethernet PHY holds the link up on its own,")
        print("    so an 'active' link proves only that the PHY has power, not that the board runs.")
        print("    Almost always POWER: the Pi 5 with two cameras needs a real 5 V / 5 A (27 W) supply,")
        print("    and most USB-C chargers only offer 5 V / 3 A. Not the laptop, not the dock. If it")
        print("    still comes up red with a good supply, suspect the SD card.")
        print()
        print("    Watch for it coming back:   .venv/bin/python tools/find_board.py --watch")
        return 1

    print("Nothing that looks like the board is on the network.")
    print("    - Is the Ethernet cable in, at both ends?")
    print("    - Is the board powered from a 5 V / 5 A supply (not the laptop)?")
    print("    - Does an interface above show 'status: active'?  ifconfig en7 | grep status")
    print("    - Give it 60 s after power-on, then:  .venv/bin/python tools/find_board.py --watch")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--watch", action="store_true", help="keep scanning until the cameras answer")
    args = ap.parse_args()
    cfg = config.load(args.config)

    while True:
        print(f"--- {time.strftime('%H:%M:%S')} ---")
        ifaces, results = scan(cfg)
        code = report(ifaces, results, cfg)
        if not args.watch or code == 0:
            return code
        print("\nwaiting 15 s...\n")
        time.sleep(15)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
