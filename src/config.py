import os
from pathlib import Path

OPENAGENDA_KEY = os.environ["OPENAGENDA_KEY"]
MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]

DEPARTEMENTS = [
	"Côtes-d'Armor",
	"Finistère",
	"Ille-et-Vilaine",
	"Morbihan",
	"Loire-Atlantique",
]

OPENAGENDA_API = "https://api.openagenda.com/v2/agendas"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"
EMBED_BATCH = 64

DATA = Path("data")
RAW_FILE = DATA / "raw" / "agendas.json"
CLEAN_FILE = DATA / "clean" / "agendas.parquet"
EMBED_FILE = DATA / "embedded" / "agendas.parquet"