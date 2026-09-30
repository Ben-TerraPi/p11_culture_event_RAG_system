import json
import re
from html import unescape

import pandas as pd

from config import CLEAN_FILE, RAW_FILE, REFERENCE_DATE, WINDOW_END, WINDOW_START

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


def clean_event(event: dict) -> dict:
    """Projette un événement du dump vers le schéma cible."""
    keywords = event.get("keywords_fr") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    return {
        "event_id": str(event.get("uid") or ""),
        "agenda_id": str(event.get("originagenda_uid") or ""),
        "agenda_title": event.get("originagenda_title"),
        "title": strip_html(event.get("title_fr")),
        "description": strip_html(
            event.get("longdescription_fr") or event.get("description_fr")
        ),
        "date_start": event.get("firstdate_begin"),
        "city": event.get("location_city"),
        "department": event.get("location_department"),
        "latitude": (event.get("location_coordinates") or {}).get("lat"),
        "longitude": (event.get("location_coordinates") or {}).get("lon"),
        "url": event.get("canonicalurl"),
        "image": event.get("image"),
        "keywords": ", ".join(str(keyword) for keyword in keywords),
    }


def build_dataset(raw_events: list[dict]) -> pd.DataFrame:
    columns = [
        "event_id", "agenda_id", "agenda_title", "title", "description",
        "date_start", "city", "department", "latitude", "longitude", "url",
        "image", "keywords", "is_upcoming",
    ]
    if not raw_events:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame([clean_event(event) for event in raw_events])
    df = df[df["event_id"].ne("") & df["date_start"].notna()]
    df = df.drop_duplicates(subset=["event_id", "date_start"])
    dates = pd.to_datetime(df["date_start"], utc=True, errors="coerce")
    window = (
        dates.notna()
        & (dates >= pd.Timestamp(WINDOW_START, tz="UTC"))
        & (dates < pd.Timestamp(WINDOW_END + pd.Timedelta(days=1), tz="UTC"))
    )
    df = df[window]
    dates = dates[window]
    df["is_upcoming"] = dates >= pd.Timestamp(REFERENCE_DATE, tz="UTC")
    return df


if __name__ == "__main__":
    events = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    df = build_dataset(events)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    print(f"[clean] {len(df)} événements -> {CLEAN_FILE}")