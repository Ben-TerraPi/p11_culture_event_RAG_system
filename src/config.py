import os
from datetime import date, timedelta
from pathlib import Path

MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]

DEPARTEMENTS = [
	"Côtes-d'Armor",
	"Finistère",
	"Ille-et-Vilaine",
	"Morbihan",
	"Loire-Atlantique",
]

DATASET_API = (
	"https://public.opendatasoft.com/api/explore/v2.1/"
	"catalog/datasets/evenements-publics-openagenda/records/"
)

EXPORT_API = DATASET_API.replace("/records/", "/exports/json")

DUMP_DATE = date(2024, 4, 8)
REFERENCE_DATE = date(2023, 4, 8)
WINDOW_START = REFERENCE_DATE.replace(year=REFERENCE_DATE.year - 1)
WINDOW_END = min(
	REFERENCE_DATE.replace(year=REFERENCE_DATE.year + 1),
	DUMP_DATE,
)
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"
EMBED_BATCH = 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

RAW_FILE = DATA / "raw" / "events.json"
CLEAN_FILE = DATA / "clean" / "events.parquet"
EMBED_FILE = DATA / "embedded" / "events.parquet"