import json
import pandas as pd

import extract
import clean
import embed

from config import EMBED_FILE, RAW_FILE, CLEAN_FILE


def main():
    # 1. Extraction
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    agendas = extract.fetch_agendas()
    if not agendas:
        raise RuntimeError("OpenAgenda n'a retourné aucun agenda.")
    RAW_FILE.write_text(json.dumps(agendas, ensure_ascii=False), encoding="utf-8")

    # 2. Nettoyage
    clean.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.build_dataset(json.loads(RAW_FILE.read_text(encoding="utf-8"))) \
         .to_parquet(CLEAN_FILE, index=False)

    # 3. Vectorisation
    embed.EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = embed.add_embeddings(pd.read_parquet(CLEAN_FILE))
    df.to_parquet(EMBED_FILE, index=False)

    # Bilan
    print(f"\nTotal agendas : {len(df)}")
    print(f"Agendas officiels : {df['official'].fillna(False).sum()}")


if __name__ == "__main__":
    main()