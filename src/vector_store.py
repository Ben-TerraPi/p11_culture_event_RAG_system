from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from embed import get_embeddings


def _normalise(vectors) -> np.ndarray:
    """Normalise les vecteurs : le produit scalaire devient la similarité cosinus."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(df: pd.DataFrame, index_file: Path, metadata_file: Path) -> int:
    """Indexe les embeddings et enregistre les métadonnées dans le même ordre."""
    if df.empty:
        raise ValueError("Aucun embedding disponible pour construire l'index FAISS.")

    vectors = _normalise(df["embedding"].tolist())
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    for file in (index_file, metadata_file):
        file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    df.drop(columns=["embedding"]).to_parquet(metadata_file, index=False)
    return index.ntotal


@lru_cache(maxsize=2)
def _load(index_file: str, metadata_file: str):
    return faiss.read_index(index_file), pd.read_parquet(metadata_file)


def search(query: str, index_file: Path, metadata_file: Path) -> pd.DataFrame:
    """Classe tous les chunks par similarité avec la question (colonne `score`).

    L'index est exhaustif : le filtrage par date ou par lieu se fait ensuite
    sur les métadonnées (voir rag.retrieve).
    """
    if not (index_file.exists() and metadata_file.exists()):
        raise FileNotFoundError("Base vectorielle absente : lancez `python src/main.py`.")

    index, metadata = _load(str(index_file), str(metadata_file))
    scores, positions = index.search(_normalise(get_embeddings([query])), index.ntotal)

    results = metadata.iloc[positions[0]].reset_index(drop=True)
    results.insert(0, "score", scores[0])
    return results
