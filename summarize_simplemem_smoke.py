"""Summarize SimpleMem smoke retrieval traces without dumping full embeddings."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    qa_rows = {
        str(row.get("question", "")): row
        for row in debug.get("rows", [])
        if row.get("type") == "qa"
    }
    predictions = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = []
    for prediction in predictions:
        row = qa_rows.get(str(prediction.get("question", "")), {})
        retrieval = row.get("retrieval_result") or {}
        items = retrieval.get("items") or []
        metadata = row.get("retrieved_memory_metadata") or []
        meta_by_id = {
            str(item.get("memory_id", "")): item
            for item in metadata
            if isinstance(item, dict)
        }
        retrieved_rounds = [
            str(meta_by_id.get(str(item.get("id", "")), {}).get("round_id", ""))
            for item in items
        ]
        clue_rounds = [str(value) for value in prediction.get("clue_rounds", [])]
        top = []
        for item in items[:10]:
            memory_id = str(item.get("id", ""))
            meta = meta_by_id.get(memory_id, {})
            top.append(
                {
                    "rank": len(top) + 1,
                    "round_id": meta.get("round_id"),
                    "score": item.get("score"),
                    "has_raw_data": item.get("has_raw_data"),
                    "expanded_image": (item.get("raw_content") or {}).get("type") == "image",
                    "summary": item.get("summary"),
                }
            )
        output.append(
            {
                "idx": prediction.get("idx"),
                "question": prediction.get("question"),
                "gt": prediction.get("gt"),
                "pred": prediction.get("pred"),
                "em": prediction.get("em"),
                "answer_elapsed_ms": row.get("elapsed_ms"),
                "retrieved_count": len(items),
                "retrieved_image_count": sum(bool(item.get("has_raw_data")) for item in items),
                "expanded_image_count": sum(
                    (item.get("raw_content") or {}).get("type") == "image"
                    for item in items
                ),
                "clue_rounds": clue_rounds,
                "retrieved_clue_rounds": [
                    round_id for round_id in clue_rounds if round_id in retrieved_rounds
                ],
                "top10": top,
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
