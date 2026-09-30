import json
import re
from datetime import datetime, timezone
from html import unescape

import pandas as pd

from config import RAW_FILE, CLEAN_FILE

TAGS = re.compile(r"<[^>]+>")


def strip_html(text):
    """Retire les balises HTML et normalise les espaces."""
    if not text:
        return None
    if isinstance(text, dict):
        text = text.get("fr") or next(iter(text.values()), None)
        if not text:
            return None
    text = unescape(str(text))
    text = re.sub(r"\s+", " ", TAGS.sub(" ", text)).strip()
    return text or None


def clean_event(ev: dict) -> dict:
    """Projette un événement brut vers le schéma cible."""
    loc = ev.get("location") or {}
    timings = ev.get("timings") or []
    dates = sorted(t.get("begin") for t in timings if t.get("begin"))
    next_begin = (ev.get("nextTiming") or {}).get("begin")
    if next_begin:
        dates = [next_begin]
    elif not dates and (ev.get("firstTiming") or {}).get("begin"):
        dates = [ev["firstTiming"]["begin"]]

    image = ev.get("image")
    if isinstance(image, dict):
        filename = next(
            (variant.get("filename") for variant in image.get("variants", [])
             if variant.get("type") == "full"),
            image.get("filename"),
        )
        image = f"https://img.openagenda.com/main/{filename}" if filename else None

    return {
        "event_id": str(ev.get("uid") or ""),
        "title": strip_html(ev.get("title")),
        "description": strip_html(ev.get("longDescription") or ev.get("description")),
        "date_start": dates[0] if dates else None,
        "city": loc.get("city"),
        "department": loc.get("adminLevel2"),
        "latitude": loc.get("latitude") or (loc.get("coordinates") or {}).get("lat"),
        "longitude": loc.get("longitude") or (loc.get("coordinates") or {}).get("lon"),
        "url": ev.get("canonicalUrl") or ev.get("slug"),
        "image": image,
    }


def build_dataset(raw_events: list[dict]) -> pd.DataFrame:
    columns = [
        "event_id", "title", "description", "date_start", "city",
        "department", "latitude", "longitude", "url", "image", "is_upcoming",
    ]
    if not raw_events:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame([clean_event(e) for e in raw_events])
    df = df[df["event_id"].ne("") & df["date_start"].notna()]
    df = df.drop_duplicates(subset=["event_id", "date_start"])
    df["is_upcoming"] = df["date_start"].str[:10] >= datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    events = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    df = build_dataset(events)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    print(f"[clean] {len(df)} événements -> {CLEAN_FILE}")