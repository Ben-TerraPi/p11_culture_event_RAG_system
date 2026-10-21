import faiss
import numpy as np
import pandas as pd
import pytest

import vector_store
from config import FAISS_FILE, METADATA_FILE


# --------------------------------------------------------------------------
# Index synthétique
# --------------------------------------------------------------------------
@pytest.fixture
def small_store(tmp_path, monkeypatch):
    """Crée un index FAISS temporaire et remplace l'appel d'embeddings."""
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(6, 16)).astype("float32")
    df = pd.DataFrame({
        "event_id": ["1", "1", "2", "3", "4", "5"],
        "chunk_id": [f"c{i}" for i in range(6)],
        "title": [f"Concert {i}" for i in range(6)],
        "embedding": [v.tolist() for v in vectors],
    })
    index_file, meta_file = tmp_path / "t.faiss", tmp_path / "t.parquet"
    assert vector_store.build_index(df, index_file, meta_file) == 6
    # la requête « embedde » vers un vecteur connu, sans appel API
    monkeypatch.setattr(vector_store, "get_embeddings", lambda texts: [vectors[3].tolist()])
    return index_file, meta_file


def test_index_and_metadata_aligned(small_store):
    """Vérifie que le nombre de vecteurs correspond aux lignes de métadonnées."""
    index_file, meta_file = small_store
    assert faiss.read_index(str(index_file)).ntotal == len(pd.read_parquet(meta_file)) == 6


def test_embedding_column_not_persisted_in_metadata(small_store):
    """Vérifie que la colonne volumineuse des embeddings n'est pas persistée."""
    assert "embedding" not in pd.read_parquet(small_store[1]).columns


def test_search_ranks_closest_chunk_first(small_store):
    """Vérifie le classement, le score et l'ordre décroissant d'une recherche."""
    results = vector_store.search("peu importe", *small_store)
    assert len(results) == 6
    assert results.loc[0, "chunk_id"] == "c3"
    assert results.loc[0, "score"] == pytest.approx(1.0, abs=1e-4)
    assert results["score"].is_monotonic_decreasing


def test_search_missing_index_gives_clear_error(tmp_path):
    """Vérifie qu'une recherche sur des fichiers absents lève une erreur claire."""
    with pytest.raises(FileNotFoundError):
        vector_store.search("q", tmp_path / "x.faiss", tmp_path / "x.parquet")


def test_build_index_without_data_fails(tmp_path):
    """Vérifie que la construction échoue lorsque les données sont vides."""
    with pytest.raises(ValueError):
        vector_store.build_index(pd.DataFrame(columns=["embedding"]),
                                 tmp_path / "a.faiss", tmp_path / "a.parquet")


# --------------------------------------------------------------------------
# Base réelle
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_store():
    """Charge l'index réel et ses métadonnées, ou ignore le module s'ils manquent."""
    if not (FAISS_FILE.exists() and METADATA_FILE.exists()):
        pytest.skip("Base vectorielle absente : lancez `python src/main.py`.")
    return faiss.read_index(str(FAISS_FILE)), pd.read_parquet(METADATA_FILE)


def test_real_index_matches_metadata(real_store):
    """Vérifie l'alignement et la non-vacuité de l'index vectoriel réel."""
    index, metadata = real_store
    assert index.ntotal == len(metadata) > 0


def test_real_index_is_cosine_normalised(real_store):
    """Vérifie que les vecteurs réels sont normalisés pour une similarité cosinus."""
    index, _ = real_store
    sample = index.reconstruct_n(0, min(50, index.ntotal))
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)


def test_real_index_retrieves_its_own_vectors(real_store):
    """Chercher le vecteur d'un chunk doit le renvoyer en premier (score ≈ 1)."""
    index, _ = real_store
    for position in {0, index.ntotal // 2, index.ntotal - 1}:
        vector = index.reconstruct(int(position)).reshape(1, -1)
        scores, positions = index.search(vector, 1)
        assert positions[0][0] == position
        assert scores[0][0] == pytest.approx(1.0, abs=1e-3)
