"""Does Sentry actually ACCEPT our data? Posts one small event straight to the project's ingest endpoint and prints the HTTP
answer. 200 + an event id = accepted (DSN valid, project active, not rate-limited). It does NOT show that the events are
visible in the UI: open the project's Issues / Traces / Logs for that (the event is tagged verify=true).

  python verify_sentry.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

import config

dsn = config.env("SENTRY_DSN")
m = re.match(r"https://([0-9a-f]+)@([^/]+)/(\d+)$", dsn or "")
if not m:
    sys.exit("SENTRY_DSN missing or not in the expected form")
key, host, pid = m.groups()
eid = uuid.uuid4().hex
event = {"event_id": eid, "message": "verify_sentry.py: ingest check (safe to ignore)", "level": "info", "platform": "python",
         "timestamp": time.time(), "environment": config.env("SENTRY_ENV", "demo"),
         "tags": {"verify": "true", "service": "verify_sentry"}}
body = (json.dumps({"event_id": eid, "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
        + json.dumps({"type": "event"}) + "\n" + json.dumps(event) + "\n").encode()
req = urllib.request.Request(f"https://{host}/api/{pid}/envelope/", body, method="POST", headers={
    "Content-Type": "application/x-sentry-envelope",
    "X-Sentry-Auth": f"Sentry sentry_version=7, sentry_client=htn-verify/1.0, sentry_key={key}"})
try:
    r = urllib.request.urlopen(req, timeout=10)
    print(f"ACCEPTED  HTTP {r.status}  response {r.read().decode()[:80]}  event id {eid}")
except urllib.error.HTTPError as e:
    print(f"REJECTED  HTTP {e.code}  {e.read().decode()[:200]}  (429 = rate limited, 403 = bad key / project disabled)")
    sys.exit(1)
except OSError as e:
    print(f"NO ANSWER  {e}")
    sys.exit(2)
