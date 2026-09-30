import json
from datetime import timedelta

import requests

from config import (
    EXPORT_API,
    DEPARTEMENTS,
    RAW_FILE,
    WINDOW_END,
    WINDOW_START,
)


def fetch_events() -> list[dict]:
    where = "location_department in (" + ", ".join(
        f'"{department}"' for department in DEPARTEMENTS
    ) + ") AND " + (
        f'firstdate_begin >= "{WINDOW_START.isoformat()}T00:00:00+00:00" '
        f'AND firstdate_begin < "{(WINDOW_END + timedelta(days=1)).isoformat()}T00:00:00+00:00"'
    )

    response = requests.get(
        EXPORT_API,
        params={"lang": "fr", "where": where},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    events = data.get("results") if isinstance(data, dict) else data

    if not isinstance(events, list):
        raise RuntimeError(f"Réponse OpenDataSoft inattendue : {data}")

    print(f"[extract] {len(events)} événements récupérés")
    return events


if __name__ == "__main__":
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(fetch_events(), ensure_ascii=False), encoding="utf-8")
    print(f"[extract] événements sauvegardés : {RAW_FILE}")