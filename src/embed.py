import pandas as pd
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api_utils import with_retry
from config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBED_BATCH, EMBED_MODEL, EMBED_URL, MISTRAL_KEY, require,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def chunk_text(text) -> list[str]:
    """Découpe un texte en chunks de CHUNK_SIZE caractères maximum."""
    return splitter.split_text(text) if isinstance(text, str) else []


def chunk_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par chunk, avec toutes les métadonnées de l'événement."""
    rows = []
    for _, event in df.iterrows():
        text = event["description"] if isinstance(event["description"], str) else event["title"]
        for number, chunk in enumerate(chunk_text(text)):
            rows.append({
                **event.to_dict(),
                "chunk_id": f"{event['event_id']}_{number}",
                "chunk_number": number,
                "chunk_text": chunk,
            })
    return pd.DataFrame(rows)


def event_header(row) -> str:
    """Titre, lieu, département et date, ajoutés à chaque chunk avant vectorisation.

    Sans cet en-tête, le vecteur ne contient ni la ville ni la date de l'événement.
    """
    place = ", ".join(str(row.get(k)) for k in ("venue", "city") if pd.notna(row.get(k)))
    start = row.get("date_start")
    fields = {
        "Titre": row.get("title"),
        "Lieu": place,
        "Département": row.get("department"),
        "Date": str(start)[:10] if pd.notna(start) else None,
    }
    return " | ".join(f"{label} : {value}" for label, value in fields.items()
                      if pd.notna(value) and value)


def text_for_embedding(row) -> str:
    return f"{event_header(row)}\n{row['chunk_text']}"


def _embed_batch(batch: list[str]) -> list[list[float]]:
    key = require(MISTRAL_KEY, "MISTRAL_API_KEY")
    response = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": EMBED_MODEL, "input": batch},
        timeout=60,
    )
    if not response.ok:
        print(f"[embed] HTTP {response.status_code}: {response.text[:300]}")
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Vectorise les textes par lots, avec cadence et réessais sur 429."""
    show_progress = len(texts) > EMBED_BATCH
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        vectors += with_retry(lambda: _embed_batch(batch))
        if show_progress:
            print(f"\r[embed] {len(vectors)} / {len(texts)}", end="", flush=True)
    if show_progress:
        print()
    return vectors


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `embedding` (un vecteur par chunk)."""
    df = df.copy()
    df["embedding"] = get_embeddings([text_for_embedding(row) for _, row in df.iterrows()])
    return df
