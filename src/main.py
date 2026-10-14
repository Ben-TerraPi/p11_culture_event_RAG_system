import json

import clean
import embed
import extract
import vector_store
from config import CLEAN_FILE, FAISS_FILE, METADATA_FILE, RAW_FILE


def main() -> None:
    # 1. Extraction
    events = extract.fetch_events()
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

    # 2. Nettoyage + contrôle de la fenêtre de dates et des départements
    df = clean.build_dataset(events)
    problems = clean.check_dataset(df)
    if problems:
        raise RuntimeError("Données non conformes : " + " ; ".join(problems))
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    print(f"[clean] {len(df)} événements -> {CLEAN_FILE}")

    # 3. Chunking + vectorisation
    chunks = embed.add_embeddings(embed.chunk_dataset(df))

    # 4. Index FAISS
    indexed = vector_store.build_index(chunks, FAISS_FILE, METADATA_FILE)

    print(f"\nÉvénements : {len(df)} | passés : {(~df['is_upcoming']).sum()} "
          f"| à venir : {df['is_upcoming'].sum()}")
    print(f"Chunks indexés : {indexed} -> {FAISS_FILE}")
    print(df["department"].value_counts())


if __name__ == "__main__":
    main()
