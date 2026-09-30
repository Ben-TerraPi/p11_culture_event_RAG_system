import json
import time
from datetime import datetime, timedelta, timezone

import requests

from config import OPENAGENDA_API, OPENAGENDA_KEY, RAW_FILE, DEPARTEMENTS, HISTORY_DAYS, FUTURE_DAYS


def fetch_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    events, skip = [], 0
    while True:
        params = {
            "key": OPENAGENDA_KEY, "lang": "fr", "size": 100, "skip": skip,
            "search[passed]": 1,
            "search[timings][from]": (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d"),
            "search[timings][to]": (now + timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d"),
            "search[department][]": DEPARTEMENTS,
        }
        data = requests.get(OPENAGENDA_API, params=params, timeout=30).json()
        page = data.get("events") or []
        events += page
        total = data.get("pagination", {}).get("total", len(events))
        print(f"[extract] {len(events)} / {total}")
        if not page or len(events) >= total:
            return events
        skip += 100
        time.sleep(0.1)


if __name__ == "__main__":
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(fetch_events(), ensure_ascii=False), encoding="utf-8")
    print(f"[extract] sauvegardé : {RAW_FILE}")