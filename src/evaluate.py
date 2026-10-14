"""Évalue le RAG sur data/evaluation/questions.jsonl.

Une ligne JSON par question : `question`, `expected_answer`, `expected_event_ids`, `notes`.

Mesures :
  - hit@k : l'un des événements attendus fait-il partie des événements récupérés ?
  - note 0/1/2 d'un LLM-juge qui compare la réponse à `expected_answer`.

    python src/evaluate.py [--upcoming-only]
"""

import argparse
import json
import re
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import rag
from api_utils import with_retry
from config import DATA

EVAL_DIR = DATA / "evaluation"

JUDGE_PROMPT = """Tu es un correcteur strict. Compare la RÉPONSE du chatbot à la RÉPONSE DE RÉFÉRENCE.
Note : 2 = équivalente (mêmes événements et faits clés), 1 = partielle, 0 = fausse, hors-sujet ou inventée.
Si la référence indique qu'aucune information n'existe, la bonne réponse est de le dire clairement.
Réponds uniquement par le chiffre 0, 1 ou 2."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "results.json")
    parser.add_argument("--upcoming-only", action="store_true",
                        help="ne récupérer que les événements à venir (par défaut : tous)")
    args = parser.parse_args()

    lines = args.dataset.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    judge = ChatPromptTemplate.from_messages([
        ("system", JUDGE_PROMPT),
        ("human", "QUESTION : {question}\n\nRÉPONSE DE RÉFÉRENCE : {reference}\n\nRÉPONSE : {answer}"),
    ]) | rag.llm | StrOutputParser()

    results = []
    for number, item in enumerate(questions, start=1):
        if not item.get("expected_answer", "").strip():
            print(f"[eval] q{number:02d} ignorée : expected_answer à compléter")
            continue

        documents = rag.retrieve(item["question"], upcoming_only=args.upcoming_only)
        answer = rag.generate_answer(item["question"], documents)

        retrieved = [doc.metadata["event_id"] for doc in documents]
        expected = {str(event_id) for event_id in item.get("expected_event_ids", [])}
        hit = bool(expected & set(retrieved)) if expected else None

        verdict = with_retry(lambda: judge.invoke({
            "question": item["question"],
            "reference": item["expected_answer"],
            "answer": answer,
        }))
        match = re.search(r"[012]", verdict)
        score = int(match.group()) if match else 0

        results.append({**item, "answer": answer, "retrieved_event_ids": retrieved,
                        "hit": hit, "score": score})
        print(f"[eval] q{number:02d} : note {score}/2 | hit={hit}")

    hits = [r["hit"] for r in results if r["hit"] is not None]
    summary = {
        "questions_evaluees": len(results),
        "hit_at_k": sum(hits) / len(hits) if hits else None,
        "score_reponse_moyen": sum(r["score"] for r in results) / len(results) / 2 if results else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
