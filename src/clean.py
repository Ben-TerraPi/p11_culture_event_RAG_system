import json
import re

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
    text = re.sub(r"\s+", " ", TAGS.sub(" ", str(text))).strip()
    return text or None


def clean_agenda(agenda: dict) -> dict:
    """Projette un agenda brut vers le schéma cible."""
    summary = agenda.get("summary") or {}
    published = summary.get("publishedEvents") or {}
    return {
        "agenda_id": str(agenda.get("uid") or ""),
        "title": strip_html(agenda.get("title")),
        "description": strip_html(agenda.get("description")),
        "slug": agenda.get("slug"),
        "url": agenda.get("url"),
        "official": agenda.get("official"),
        "private": agenda.get("private"),
        "indexed": agenda.get("indexed"),
        "image": agenda.get("image"),
        "keywords": ", ".join(summary.get("keywords") or []),
        "published_current": published.get("current", 0),
        "published_passed": published.get("passed", 0),
        "published_upcoming": published.get("upcoming", 0),
    }


def build_dataset(raw_agendas: list[dict]) -> pd.DataFrame:
    columns = [
        "agenda_id", "title", "description", "slug", "url", "official",
        "private", "indexed", "image", "keywords", "published_current",
        "published_passed", "published_upcoming",
    ]
    if not raw_agendas:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame([clean_agenda(agenda) for agenda in raw_agendas])
    df = df[df["agenda_id"].ne("")]
    df = df.drop_duplicates(subset=["agenda_id"])
    return df


if __name__ == "__main__":
    agendas = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    df = build_dataset(agendas)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    print(f"[clean] {len(df)} agendas -> {CLEAN_FILE}")