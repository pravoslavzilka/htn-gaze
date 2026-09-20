"""Create the three Sentry dashboard widgets through the API instead of clicking them together.

    python create_sentry_dashboard.py --dry-run          # prints the request body and writes sentry_dashboard.json
    python create_sentry_dashboard.py                    # needs SENTRY_AUTH_TOKEN (scope org:write) and SENTRY_ORG_SLUG

STATUS: NOT TESTED against the live API (no auth token was available when this was written). The endpoint and top-level fields
follow docs.sentry.io/api/dashboards/create-a-new-dashboard-for-an-organization; the exact widgetType / aggregate names for the
Spans and Logs datasets are my best reading and may need a tweak. The script prints Sentry's answer verbatim, so a rejected
field is easy to fix, and DASHBOARD.md has the same widgets as manual click-through steps.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

import config


def widget(title, dataset, aggregates, conditions, columns=(), display="line", pos=(0, 0), orderby=""):
    fields = list(columns) + list(aggregates)
    return {"title": title, "displayType": display, "widgetType": dataset, "interval": "1m",
            "queries": [{"name": "", "fields": fields, "aggregates": list(aggregates), "columns": list(columns),
                         "conditions": conditions, "orderby": orderby or (f"-{aggregates[0]}" if columns else ""),
                         "fieldAliases": []}],
            "layout": {"x": pos[0], "y": pos[1], "w": 3, "h": 2, "minH": 2}}


def body(project_id=None):
    b = {"title": "Gaze pipeline: system health", "period": "1h", "widgets": [
        widget("Pipeline stage latency (p95 by span op)", "spans", ["p95(span.duration)"],
               "transaction:gaze_pipeline span.op:[cap,pupil,gaze,scene,fix]", columns=["span.op"], display="bar", pos=(0, 0)),
        widget("tracking_lost logs over time", "logs", ["count(message)"], "message:tracking_lost", columns=["run"], pos=(3, 0)),
        widget("OMNI call latency (p50 / p95)", "spans", ["p50(span.duration)", "p95(span.duration)"],
               "span.op:gen_ai.chat", pos=(0, 2)),
        widget("Whole voice request + sound generation p95", "spans", ["p95(span.duration)"],
               "transaction:omni_request OR span.op:elevenlabs.sound_generation", columns=["span.op"], display="bar", pos=(3, 2)),
        widget("Pi events and packet loss", "logs", ["count(message)"],
               "message:pi_event_* OR message:packet_loss", columns=["message"], pos=(0, 4)),
    ]}
    if project_id:
        b["projects"] = [int(project_id)]
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dsn = config.env("SENTRY_DSN") or ""
    pid = (re.search(r"/(\d+)$", dsn) or [None, None])[1]
    payload = body(pid)
    if a.dry_run:
        open("sentry_dashboard.json", "w").write(json.dumps(payload, indent=1))
        print(json.dumps(payload, indent=1)[:1500], "\n... written to sentry_dashboard.json")
        return
    token, org = config.env("SENTRY_AUTH_TOKEN"), config.env("SENTRY_ORG_SLUG")
    if not (token and org):
        sys.exit("Set SENTRY_AUTH_TOKEN (org:write) and SENTRY_ORG_SLUG in laptop/.env, or use --dry-run")
    req = urllib.request.Request(f"https://sentry.io/api/0/organizations/{org}/dashboards/", json.dumps(payload).encode(),
                                 method="POST", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=15)
        d = json.loads(r.read())
        print(f"CREATED dashboard {d.get('id')} -> https://sentry.io/organizations/{org}/dashboard/{d.get('id')}/")
    except urllib.error.HTTPError as e:
        print(f"REJECTED HTTP {e.code}: {e.read().decode()[:600]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
