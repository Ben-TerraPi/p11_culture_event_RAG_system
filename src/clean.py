import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from html import unescape

import pandas as pd

from config import DEPARTEMENTS, FUTURE_DAYS, HISTORY_DAYS

COLUMNS = [
    "event_id", "title", "description", "date_start", "date_end", "city",
    "department", "venue", "address", "url", "is_upcoming",
]


def norm(text) -> str:
    """Normalise un texte pour les comparaisons (accents, casse, apostrophes)."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("’", "'").casefold().strip()


ALLOWED_DEPARTMENTS = {norm(d) for d in DEPARTEMENTS}


def strip_html(text) -> str | None:
    """Retire les balises HTML et normalise les espaces."""
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", unescape(str(text)))
    return " ".join(text.split()) or None


def pick_timing(ev: dict, start: str, end: str, today: str) -> dict | None:
    """Choisit le créneau à retenir parmi ceux situés DANS la fenêtre.

    `timings` contient tous les créneaux de l'événement, même hors fenêtre :
    on garde le prochain créneau à venir, sinon le plus récent du passé.
    """
    timings = [t for t in ev.get("timings") or [] if t.get("begin")]
    in_window = [t for t in timings if start <= t["begin"][:10] <= end]
    upcoming = [t for t in in_window if t["begin"][:10] >= today]
    if upcoming:
        return min(upcoming, key=lambda t: t["begin"])
    if in_window:
        return max(in_window, key=lambda t: t["begin"])
    return None


def clean_event(ev: dict, start: str, end: str, today: str) -> dict | None:
    """Convertit un événement brut vers le schéma cible (None s'il est hors fenêtre)."""
    timing = pick_timing(ev, start, end, today)
    if timing is None or not ev.get("uid"):
        return None

    loc = ev.get("location") or {}
    return {
        "event_id": str(ev["uid"]),
        "title": strip_html(ev.get("title")),
        "description": strip_html(ev.get("longDescription") or ev.get("description")),
        "date_start": timing["begin"],
        "date_end": timing.get("end"),
        "city": loc.get("city"),
        "department": loc.get("adminLevel2"),
        "venue": strip_html(loc.get("name")),
        "address": strip_html(loc.get("address")),
        "url": ev.get("canonicalUrl"),
        "is_upcoming": timing["begin"][:10] >= today,
    }


def build_dataset(raw_events: list[dict], today: date | None = None) -> pd.DataFrame:
    """Un événement par ligne, dans la fenêtre de dates et les départements choisis."""
    today = today or datetime.now(timezone.utc).date()
    start = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    end = (today + timedelta(days=FUTURE_DAYS)).isoformat()

    rows = [clean_event(ev, start, end, today.isoformat()) for ev in raw_events]
    df = pd.DataFrame([row for row in rows if row], columns=COLUMNS)
    df = df.drop_duplicates("event_id")
    df = df[df["department"].map(norm).isin(ALLOWED_DEPARTMENTS)]
    return df.reset_index(drop=True)


def check_dataset(df: pd.DataFrame, today: date | None = None, slack_days: int = 1) -> list[str]:
    """Renvoie la liste des problèmes (vide si le jeu de données est conforme).

    `slack_days` absorbe l'écart de fuseau horaire entre jour local et UTC.
    """
    if df.empty:
        return ["jeu de données vide"]

    today = today or datetime.now(timezone.utc).date()
    oldest = (today - timedelta(days=HISTORY_DAYS + slack_days)).isoformat()
    latest = (today + timedelta(days=FUTURE_DAYS + slack_days)).isoformat()
    starts = df["date_start"].astype(str).str[:10]

    checks = {
        f"événements de plus de {HISTORY_DAYS} jours": starts < oldest,
        "événements trop lointains": starts > latest,
        "événements hors des départements choisis":
            ~df["department"].map(norm).isin(ALLOWED_DEPARTMENTS),
        "dates manquantes": df["date_start"].isna(),
        "titres manquants": df["title"].isna(),
    }
    return [f"{int(bad.sum())} {label}" for label, bad in checks.items() if bad.any()]
