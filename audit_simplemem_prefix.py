"""Offline structural audit for an interrupted SimpleMem prefix snapshot."""

import argparse
import json
from pathlib import Path

from benchmark.dataset import MemoryBenchmarkDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialog-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MemoryBenchmarkDataset(args.dialog_json, args.image_root)
    records = []
    for path in sorted((args.data_dir / "index" / "mau_store").glob("mau_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())

    expected = [
        str(dialogue.get("round", "")).strip()
        for session_id in dataset.session_order()
        for dialogue in dataset.get_session(session_id).get("dialogues", [])
        if str(dialogue.get("round", "")).strip() in dataset.rounds
    ]
    observed = []
    for record in records:
        tags = list((record.get("metadata") or {}).get("tags") or [])
        round_ids = [tag.split(":", 1)[1] for tag in tags if tag.startswith("round_id:")]
        if len(round_ids) != 1:
            raise RuntimeError(f"invalid round tags for {record.get('id')}: {tags}")
        observed.append(round_ids[0])

    mapping_path = args.data_dir / "index" / "vectors" / "text" / "id_mapping.json"
    vector_ids = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else []
    result = {
        "status": "passed" if observed == expected[: len(observed)] else "failed",
        "expected_rounds": len(expected),
        "stored_rounds": len(observed),
        "unique_rounds": len(set(observed)),
        "unique_memory_ids": len({str(record.get("id", "")) for record in records}),
        "vector_mapping_count": len(vector_ids),
        "raw_pointer_count": sum(bool(record.get("raw_pointer")) for record in records),
        "first_round": observed[0] if observed else None,
        "last_round": observed[-1] if observed else None,
        "next_round": expected[len(observed)] if len(observed) < len(expected) else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
