import json
import pandas as pd

import extract
import clean
import embed

from config import EMBED_FILE, RAW_FILE, CLEAN_FILE


def main():
    # 1. Extraction
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(extract.fetch_events(), ensure_ascii=False), encoding="utf-8")

    # 2. Nettoyage
    clean.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.build_dataset(json.loads(RAW_FILE.read_text(encoding="utf-8"))) \
         .to_parquet(CLEAN_FILE, index=False)

    # 3. Vectorisation
    embed.EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = embed.add_embeddings(pd.read_parquet(CLEAN_FILE))
    df.to_parquet(EMBED_FILE, index=False)

    # Bilan
    print(f"\nTotal : {len(df)} | Historique : {(~df.is_upcoming).sum()} "
          f"| À venir : {df.is_upcoming.sum()}")
    print(df["department"].value_counts(dropna=False))


if __name__ == "__main__":
    main()