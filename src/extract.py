from datetime import datetime, timedelta, timezone

import requests

from api_utils import with_retry
from config import (
    DEPARTEMENTS, FUTURE_DAYS, HISTORY_DAYS, OPENAGENDA_API, OPENAGENDA_KEY,
    SEARCH_QUERY, require,
)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fetch_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    headers = {"key": require(OPENAGENDA_KEY, "OPENAGENDA_KEY")}
    params = {
        "monolingual": "fr",
        "detailed": 1,
        "includeLabels": 1,
        "longDescriptionFormat": "html",
        "size": 300,
        "timings[gte]": _iso(now - timedelta(days=HISTORY_DAYS)),
        "timings[lte]": _iso(now + timedelta(days=FUTURE_DAYS)),
        "adminLevel2[]": DEPARTEMENTS,
    }
    if SEARCH_QUERY:
        params["search"] = SEARCH_QUERY
        params["threshold"] = "auto"

    def get_page() -> dict:
        response = requests.get(OPENAGENDA_API, headers=headers, params=params, timeout=30)
        if not response.ok:
            print(f"[extract] HTTP {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        return response.json()

    events = {}  # uid -> événement : écarte les doublons éventuels entre pages
    while True:
        data = with_retry(get_page)
        page = data["events"]
        events.update({event["uid"]: event for event in page})
        print(f"[extract] {len(events)} / {data.get('total')}")

        if not page or not data.get("after"):
            return list(events.values())
        params["after[]"] = data["after"]
