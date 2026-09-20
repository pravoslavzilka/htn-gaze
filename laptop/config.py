"""Settings from the environment, with a tiny .env loader (no dependency). Secrets never live in code."""
import os
from pathlib import Path


def load_env(path=None):
    for p in ([Path(path)] if path else [Path(__file__).with_name(".env"), Path(".env")]):
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def env(name, default=None, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


UDP_HOST = env("UDP_HOST", "0.0.0.0")
UDP_PORT = env("UDP_PORT", 9999, int)
TIGER_DSN = env("TIGER_DSN")
TRACKING_LOST_CONF = env("TRACKING_LOST_CONF", 0.5, float)
FLUSH_SECONDS = env("FLUSH_SECONDS", 1.0, float)
MAX_BUFFER_ROWS = env("MAX_BUFFER_ROWS", 50000, int)
