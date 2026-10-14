from datetime import datetime, timedelta, timezone

import faiss
import pandas as pd
import pytest

from src.clean import ALLOWED_DEPARTMENTS, norm
from src.config import FAISS_FILE, FUTURE_DAYS, HISTORY_DAYS, METADATA_FILE


@pytest.fixture(scope="module")
def vector_data():
    if not FAISS_FILE.exists() or not METADATA_FILE.exists():
        pytest.skip("Index FAISS absent : exécutez `python src/main.py` avant ce test.")

    index = faiss.read_index(str(FAISS_FILE))
    metadata = pd.read_parquet(METADATA_FILE)
    return index, metadata


def test_metadata_matches_faiss_index(vector_data):
    index, metadata = vector_data

    assert index.ntotal > 0
    assert len(metadata) == index.ntotal
    assert metadata["event_id"].notna().all()
    assert metadata["title"].notna().all()
    assert metadata["date_start"].notna().all()


def test_events_are_in_expected_departments_and_date_window(vector_data):
    _, metadata = vector_data
    dates = pd.to_datetime(metadata["date_start"], utc=True, errors="coerce")

    # La fenêtre « 1 an » est relative à la construction de la base, pas à
    # aujourd'hui : sinon le test échoue dès que la base vieillit.
    built = pd.Timestamp(
        datetime.fromtimestamp(METADATA_FILE.stat().st_mtime, timezone.utc)
    ).normalize()
    slack = timedelta(days=1)  # écart fuseau horaire (jour local vs UTC)

    assert dates.notna().all(), "Toutes les dates doivent être valides."
    assert metadata["department"].notna().all(), "Chaque événement doit avoir un département."
    assert set(metadata["department"].map(norm)) <= ALLOWED_DEPARTMENTS
    assert (dates >= built - timedelta(days=HISTORY_DAYS) - slack).all()
    assert (dates <= built + timedelta(days=FUTURE_DAYS) + slack).all()
