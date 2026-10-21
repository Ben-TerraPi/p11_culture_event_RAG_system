"""Évalue le RAG sur data/evaluation/questions.jsonl.

Une ligne JSON par question : `question`, `expected_answer`, `expected_event_ids`, `notes`.

Pour chaque question, affiche la réponse attendue et la réponse du chatbot (à comparer
à la main) et mesure le hit@k : l'un des événements attendus fait-il partie des
événements récupérés ?

    python src/evaluate.py [--upcoming-only]
"""

import argparse
import json

import rag
from config import DATA

EVAL_DIR = DATA / "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upcoming-only", action="store_true",
                        help="ne récupérer que les événements à venir (par défaut : tous)")
    args = parser.parse_args()

    lines = (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    results = []
    for number, item in enumerate(questions, start=1):
        if not item.get("expected_answer", "").strip():
            print(f"[q{number:02d}] ignorée : expected_answer à compléter")
            continue

        documents = rag.retrieve(item["question"], upcoming_only=args.upcoming_only)
        answer = rag.generate_answer(item["question"], documents)

        retrieved = [doc.metadata["event_id"] for doc in documents]
        expected = {str(event_id) for event_id in item.get("expected_event_ids", [])}
        hit = bool(expected & set(retrieved)) if expected else None

        results.append({**item, "answer": answer, "retrieved_event_ids": retrieved, "hit": hit})
        print(f"\n[q{number:02d}] {item['question']}")
        print(f"  attendu : {item['expected_answer']}")
        print(f"  obtenu  : {answer}")
        print(f"  hit@k   : {hit}")

    hits = [r["hit"] for r in results if r["hit"] is not None]
    print(f"\nhit@k : {sum(hits)}/{len(hits)} questions avec événements attendus")

    output = EVAL_DIR / "results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats détaillés : {output}")


if __name__ == "__main__":
    main()
