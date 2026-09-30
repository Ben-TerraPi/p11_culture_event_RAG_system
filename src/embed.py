import requests
import pandas as pd

from config import MISTRAL_KEY, EMBED_URL, EMBED_MODEL, EMBED_BATCH, CLEAN_FILE, EMBED_FILE


def text_for_embedding(row) -> str:
    """Texte utile à la recherche sémantique d'un événement."""
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
    df = add_embeddings(df)
    EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EMBED_FILE, index=False)
    print(f"[embed] {df['embedding'].notna().sum()} vectorisés -> {EMBED_FILE}")