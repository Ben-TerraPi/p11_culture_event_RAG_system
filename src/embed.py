import requests
import pandas as pd

from config import (
    MISTRAL_KEY, EMBED_URL, EMBED_MODEL, EMBED_BATCH, CLEAN_FILE, EMBED_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en morceaux de longueur limitée."""
    if not text or pd.isna(text):
        return []
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size doit être supérieur à overlap >= 0")

    text = " ".join(str(text).split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text) and text[end] != " ":
            end = text.find(" ", end)
            if end == -1:
                end = len(text)

        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
        if overlap:
            next_space = text.find(" ", start)
            start = next_space + 1 if next_space != -1 else end
    return chunks


def chunk_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Crée une ligne par chunk tout en conservant les métadonnées événement."""
    rows = []
    for _, row in df.iterrows():
        description = row.get("description")
        source = description if isinstance(description, str) and description.strip() else row.get("title")
        chunks = chunk_text(source)
        for chunk_number, chunk in enumerate(chunks):
            item = row.to_dict()
            item["chunk_id"] = f"{item['event_id']}_{chunk_number}"
            item["chunk_number"] = chunk_number
            item["chunk_text"] = chunk
            rows.append(item)
    return pd.DataFrame(rows)


def text_for_embedding(row) -> str:
    """Texte utile à la recherche sémantique d'un événement."""
    chunk = row.get("chunk_text")
    if isinstance(chunk, str) and chunk.strip():
        return chunk[:8000]
    fields = [
        row["title"], row["description"], row["keywords"], row["venue"],
        row["address"], row["city"], row["department"], row["date_start"],
    ]
    return " ".join(
        str(value) for value in fields
        if value is not None and not pd.isna(value)
    )[:8000]  # 8k caractères maximum


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embeddings mistral-embed par lots."""
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}"}
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = requests.post(EMBED_URL, headers=headers, timeout=60,
                             json={"model": EMBED_MODEL, "input": batch})
        resp.raise_for_status()
        vectors.extend(d["embedding"] for d in resp.json()["data"])
        print(f"[embed] {len(vectors)} / {len(texts)}")
    return vectors


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne embedding (null si texte vide)."""
    texts = df.apply(text_for_embedding, axis=1)
    df["embedding"] = None
    mask = texts.ne("")
    df.loc[mask, "embedding"] = pd.Series(
        get_embeddings(texts[mask].tolist()), index=df.index[mask]
    ).apply(list)
    return df


if __name__ == "__main__":
    df = pd.read_parquet(CLEAN_FILE)
    df = chunk_dataset(df)
    df = add_embeddings(df)
    EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EMBED_FILE, index=False)
    print(f"[embed] {df['embedding'].notna().sum()} vectorisés -> {EMBED_FILE}")