import json
import pandas as pd

import extract
import clean
import embed
import vector_store

from config import EMBED_FILE, RAW_FILE, CLEAN_FILE, FAISS_FILE, METADATA_FILE


def main():
    # 1. Extraction
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    events = extract.fetch_events()
    if not events:
        raise RuntimeError("OpenAgenda n'a retourné aucun événement.")
    RAW_FILE.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

    # 2. Nettoyage
    clean.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.build_dataset(json.loads(RAW_FILE.read_text(encoding="utf-8"))) \
         .to_parquet(CLEAN_FILE, index=False)

    # 3. Vectorisation
    embed.EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = embed.chunk_dataset(pd.read_parquet(CLEAN_FILE))
    df = embed.add_embeddings(df)
    df.to_parquet(EMBED_FILE, index=False)

    # 4. Indexation sémantique
    indexed = vector_store.build_index(df, FAISS_FILE, METADATA_FILE)
    if indexed != len(df):
        raise RuntimeError(f"{len(df) - indexed} chunks n'ont pas été indexés.")

    # Bilan
    print(f"\nTotal : {len(df)} | Historique : {(~df.is_upcoming).sum()} "
          f"| À venir : {df.is_upcoming.sum()}")
    print(f"Chunks indexés : {indexed} -> {FAISS_FILE}")
    print(df["department"].value_counts(dropna=False))


if __name__ == "__main__":
    main()