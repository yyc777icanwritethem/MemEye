from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract(source: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    output: list[dict[str, object]] = []
    for row in payload.get("rows", ()):
        if not isinstance(row, dict) or row.get("type") == "stored_memory":
            continue
        metadata = row.get("retrieved_memory_metadata")
        compact_metadata = []
        if isinstance(metadata, list):
            compact_metadata = [
                {
                    "memory_id": item.get("memory_id"),
                    "round_id": item.get("round_id"),
                    "session_id": item.get("session_id"),
                    "raw_pointer": item.get("raw_pointer"),
                    "score": item.get("score"),
                    "mau_tags": item.get("mau_tags"),
                }
                for item in metadata
                if isinstance(item, dict)
            ]
        output.append({
            "type": row.get("type"),
            "keys": sorted(row),
            "question_id": row.get("question_id") or row.get("qa_id"),
            "question": row.get("question"),
            "prediction": row.get("prediction"),
            "elapsed_ms": row.get("elapsed_ms"),
            "retrieved_memory_metadata": compact_metadata,
        })
    return {"dataset_path": payload.get("dataset_path"), "rows": output}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip large SimpleMem debug traces to QA audit fields")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
