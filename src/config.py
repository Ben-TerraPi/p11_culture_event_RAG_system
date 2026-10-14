import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAGENDA_KEY = os.getenv("OPENAGENDA_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")


def require(value, name: str):
    """Renvoie `value`, ou lève une erreur claire si la clé API est absente."""
    if not value:
        raise RuntimeError(f"Variable d'environnement manquante : {name} (voir .env.example)")
    return value


# --- Données -----------------------------------------------------------------
DEPARTEMENTS = [
    "Côtes-d'Armor",
    "Finistère",
    "Ille-et-Vilaine",
    "Morbihan",
    "Loire-Atlantique",
]
HISTORY_DAYS = 365        # 1 an d'historique
FUTURE_DAYS = 365         # + événements à venir
SEARCH_QUERY = "Concert"  # None pour récupérer tous les événements culturels
OPENAGENDA_API = "https://api.openagenda.com/v2/events"

# --- Mistral -----------------------------------------------------------------
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"
EMBED_BATCH = 16          # petits lots : l'API limite les tokens par requête
MISTRAL_CHAT_MODEL = "mistral-small-latest"
CHAT_MAX_RETRIES = 1      # les retries sont gérés par api_utils
MIN_INTERVAL = 1.1        # secondes entre deux appels (palier gratuit ~1 req/s)
MAX_RETRIES = 5

# --- Chunking et recherche ---------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_EVENTS = 4          # événements envoyés au LLM
MIN_SCORE = 0.5           # similarité minimale (à calibrer avec evaluate.py)

# --- Fichiers ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

RAW_FILE = DATA / "raw" / "agendas.json"
CLEAN_FILE = DATA / "clean" / "agendas.parquet"
FAISS_FILE = DATA / "vector_store" / "events.faiss"
METADATA_FILE = DATA / "vector_store" / "metadata.parquet"
