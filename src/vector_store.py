"""Construction et interrogation de l'index vectoriel FAISS."""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from embed import get_embeddings


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(df: pd.DataFrame, index_file: Path,
                metadata_file: Path) -> int:
    """Construit un index cosine et persiste ses métadonnées dans le même ordre."""
    valid = df[df["embedding"].notna()].copy().reset_index(drop=True)
    if valid.empty:
        raise ValueError("Aucun embedding disponible pour construire l'index FAISS.")

    vectors = _normalise(np.asarray(valid["embedding"].tolist(), dtype="float32"))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    valid.drop(columns=["embedding"]).to_parquet(metadata_file, index=False)
    return index.ntotal


def search(query: str, index_file: Path, metadata_file: Path,
           limit: int = 5) -> pd.DataFrame:
    """Recherche les chunks les plus proches et renvoie score + métadonnées."""
    if limit < 1:
        raise ValueError("limit doit être supérieur ou égal à 1.")
    index = faiss.read_index(str(index_file))
    metadata = pd.read_parquet(metadata_file)
    query_vector = _normalise(np.asarray(get_embeddings([query]), dtype="float32"))
    scores, positions = index.search(query_vector, min(limit, index.ntotal))

    results = metadata.iloc[positions[0]].copy().reset_index(drop=True)
    results.insert(0, "score", scores[0])
    return results