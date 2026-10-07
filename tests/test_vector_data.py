from datetime import timedelta
from pathlib import Path

import faiss
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE = PROJECT_ROOT / "data" / "vector_store"
FAISS_FILE = VECTOR_STORE / "events.faiss"
METADATA_FILE = VECTOR_STORE / "metadata.parquet"

EXPECTED_DEPARTMENTS = {
	"Côtes-d'Armor",
	"Finistère",
	"Ille-et-Vilaine",
	"Morbihan",
	"Loire-Atlantique",
}


@pytest.fixture(scope="module")
def vector_data():
	if not FAISS_FILE.exists() or not METADATA_FILE.exists():
		pytest.skip(
			"Index FAISS absent : exécutez `python src/main.py` avant ce test."
		)

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
	today = pd.Timestamp.now(tz="UTC").normalize()

	assert dates.notna().all(), "Toutes les dates doivent être valides."
	assert set(metadata["department"].dropna()).issubset(EXPECTED_DEPARTMENTS)
	assert (dates >= today - timedelta(days=365)).all()
	assert (dates <= today + timedelta(days=365)).all()
