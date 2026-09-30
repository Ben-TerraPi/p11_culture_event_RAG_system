import json
import time
from datetime import datetime, timedelta, timezone

import requests

from config import OPENAGENDA_API, OPENAGENDA_KEY, RAW_FILE, DEPARTEMENTS, HISTORY_DAYS, FUTURE_DAYS


def fetch_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    events, after = [], None

    while True:
        params = {
            "monolingual": "fr",
            "detailed": 1,
            "includeLabels": 1,
            "longDescriptionFormat": "html",
            "size": 300,
            "timings[gte]": (
                now - timedelta(days=HISTORY_DAYS)
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "timings[lte]": (
                now + timedelta(days=FUTURE_DAYS)
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "adminLevel2[]": DEPARTEMENTS,
        }
        if after:
            params["after[]"] = after

        response = requests.get(
            OPENAGENDA_API,
            headers={"key": OPENAGENDA_KEY},
            params=params,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"OpenAgenda HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        if "events" not in data:
            raise RuntimeError(f"Réponse OpenAgenda inattendue : {data}")

        page = data["events"]
        events.extend(page)

        total = data.get("total", len(events))
        print(f"[extract] {len(events)} / {total}")

        after = data.get("after")
        if not page or not after:
            return events

        time.sleep(0.1)


if __name__ == "__main__":
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(fetch_events(), ensure_ascii=False), encoding="utf-8")
    print(f"[extract] sauvegardé : {RAW_FILE}")