from datetime import date

import clean
import embed
from config import CHUNK_SIZE

TODAY = date(2026, 10, 14)  # fenêtre : 2025-10-14 -> 2027-10-14


def raw_event(uid, begins, department="Ille-et-Vilaine", city="Rennes", **extra):
    """Construit un événement OpenAgenda minimal pour les tests de nettoyage."""
    return {
        "uid": uid,
        "title": f"Concert {uid}",
        "longDescription": "<p>Un <b>super</b> concert &amp; plus.</p>",
        "location": {"city": city, "adminLevel2": department, "name": "La Salle"},
        "timings": [{"begin": b, "end": b} for b in begins],
        "canonicalUrl": f"https://openagenda.com/e/{uid}",
        **extra,
    }


def test_future_event_kept_and_flagged_upcoming():
    """Vérifie qu'un événement futur est conservé et marqué comme à venir."""
    df = clean.build_dataset([raw_event(1, ["2026-11-20T20:00:00+01:00"])], today=TODAY)
    assert len(df) == 1 and bool(df.loc[0, "is_upcoming"])


def test_past_event_within_year_kept():
    """Vérifie qu'un événement passé depuis moins d'un an est conservé."""
    df = clean.build_dataset([raw_event(1, ["2026-03-01T20:00:00+01:00"])], today=TODAY)
    assert len(df) == 1 and not bool(df.loc[0, "is_upcoming"])


def test_event_older_than_one_year_dropped():
    """Vérifie qu'un événement antérieur à la fenêtre historique est supprimé."""
    assert clean.build_dataset([raw_event(1, ["2025-01-01T20:00:00+01:00"])], today=TODAY).empty


def test_old_timings_ignored_when_recent_one_exists():
    """Régression : un créneau vieux de >1 an ne doit jamais devenir date_start."""
    ev = raw_event(1, ["2024-05-01T20:00:00+02:00", "2026-12-05T20:00:00+01:00"])
    df = clean.build_dataset([ev], today=TODAY)
    assert df.loc[0, "date_start"].startswith("2026-12-05")


def test_next_upcoming_timing_preferred():
    """Vérifie que le prochain créneau futur est choisi comme date de début."""
    ev = raw_event(1, ["2026-01-10T20:00:00+01:00", "2026-11-01T20:00:00+01:00",
                       "2026-12-01T20:00:00+01:00"])
    df = clean.build_dataset([ev], today=TODAY)
    assert df.loc[0, "date_start"].startswith("2026-11-01")


def test_other_department_dropped_and_accents_tolerated():
    """Vérifie le filtrage des départements et la normalisation des accents."""
    events = [
        raw_event(1, ["2026-11-20T20:00:00+01:00"], department="Paris"),
        raw_event(2, ["2026-11-20T20:00:00+01:00"], department="Finistere"),
    ]
    df = clean.build_dataset(events, today=TODAY)
    assert list(df["event_id"]) == ["2"]


def test_duplicates_removed():
    """Vérifie que deux occurrences identiques ne produisent qu'un seul événement."""
    ev = raw_event(1, ["2026-11-20T20:00:00+01:00"])
    assert len(clean.build_dataset([ev, ev], today=TODAY)) == 1


def test_check_dataset_detects_problems():
    """Vérifie que la validation accepte une ligne saine et signale deux anomalies."""
    df = clean.build_dataset([raw_event(1, ["2026-11-20T20:00:00+01:00"])], today=TODAY)
    assert clean.check_dataset(df, today=TODAY) == []
    df.loc[0, "date_start"] = "2020-01-01T10:00:00+01:00"
    df.loc[0, "department"] = "Paris"
    assert len(clean.check_dataset(df, today=TODAY)) == 2


def test_strip_html():
    """Vérifie la suppression des balises HTML et la gestion d'une valeur nulle."""
    assert clean.strip_html("<p>Un <b>super</b> concert &amp; plus.</p>") == "Un super concert & plus."
    assert clean.strip_html(None) is None


def test_chunk_text_sizes_and_edge_cases():
    """Vérifie le découpage à la bonne taille et les cas vide et court."""
    chunks = embed.chunk_text("mot " * 600)
    assert len(chunks) > 1
    assert all(0 < len(c) <= CHUNK_SIZE for c in chunks)
    assert embed.chunk_text("court") == ["court"]
    assert embed.chunk_text("") == [] and embed.chunk_text(None) == []


def test_embedding_text_contains_context():
    """Vérifie que le texte d'embedding contient les métadonnées de l'événement."""
    row = {"title": "Concert X", "venue": "La Salle", "city": "Rennes",
           "department": "Ille-et-Vilaine", "date_start": "2026-11-20T20:00:00+01:00",
           "chunk_text": "Du rock."}
    text = embed.text_for_embedding(row)
    for expected in ("Concert X", "Rennes", "Ille-et-Vilaine", "2026-11-20", "Du rock."):
        assert expected in text
