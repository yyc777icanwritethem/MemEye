#!/usr/bin/env python3
"""Replay frozen Omni-SimpleMem retrieval items through a new answer model.

The input debug trace is emitted after ``OmniMemoryOrchestrator.answer`` has
finished retrieval and on-demand expansion.  This runner reconstructs only the
official answer prompt from those serialized items; it never opens a memory
store or invokes query processing, vector/BM25/graph retrieval, or expansion.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark.evaluator import bleu_score, f1_score, score_open, summarize_results


RUNNER_VERSION = "simplemem_frozen_answer_replay.v1"
SYSTEM_CONTENT = (
    "You are a professional Q&A assistant. Your task is to extract concise, "
    "accurate answers from the provided memory context. "
    "You should make reasonable inferences from the context when possible. "
    "Do not assume a calendar year or use today's date. Only use dates that "
    "actually appear in the provided memories. "
    "You must output valid JSON format."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_frozen_context(items: Iterable[Dict[str, Any]]) -> str:
    """Exact copy of the official PyramidRetriever.format_for_llm body."""
    frozen = list(items)
    parts = [f"\n[{len(frozen)} memories found]\n"]
    for index, item in enumerate(frozen, 1):
        line_parts = [f"[Context {index}]", f"Content: {item['summary']}"]
        details = item.get("details")
        if isinstance(details, dict):
            for key in ("transcript", "text", "full_text"):
                if details.get(key):
                    line_parts.append(f"Full text: {str(details[key])[:2000]}")
                    break
        timestamp = item.get("timestamp")
        if timestamp and timestamp > 0:
            try:
                value = datetime.fromtimestamp(timestamp)
                if value.year < 2026:
                    line_parts.append(f"Time: {value.strftime('%d %B %Y')}")
            except (ValueError, OSError):
                pass
        tags = [tag for tag in (item.get("tags") or []) if not tag.startswith("locomo_")]
        if tags:
            line_parts.append(f"Tags: {', '.join(tags)}")
        metadata = item.get("metadata") or {}
        if isinstance(metadata, dict):
            if metadata.get("speaker_id"):
                line_parts.append(f"Speaker: {metadata['speaker_id']}")
            for key, label in (("persons", "Persons"), ("entities", "Entities")):
                values = metadata.get(key)
                if isinstance(values, list) and values:
                    line_parts.append(f"{label}: {', '.join(str(value) for value in values)}")
            if metadata.get("location"):
                line_parts.append(f"Location: {metadata['location']}")
            if metadata.get("topic"):
                line_parts.append(f"Topic: {metadata['topic']}")
        parts.append("\n".join(line_parts))
    return "\n\n".join(parts)


def _data_url(path: str) -> str:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"question image is missing: {image_path}")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(row: Dict[str, Any]) -> tuple[List[Dict[str, Any]], bool]:
    retrieval = row.get("retrieval_result") or {}
    items = retrieval.get("items") or []
    context = format_frozen_context(items)
    question = str(row.get("recall_query") or row.get("question") or "")
    instructions = (
        "Based on these memories:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Requirements:\n"
        "1. First, think through the reasoning process\n"
        "2. Provide a CONCISE answer (short phrase, ideally under 10 words). "
        "Use exact words and phrases from the context whenever possible rather than paraphrasing.\n"
        "3. Answer based on the provided context. You may make reasonable inferences "
        "from the information given (e.g., inferring personality traits, likely preferences, "
        "or approximate dates from surrounding context)\n"
        "4. If the question asks for a date, preserve the date supported by the "
        "memories and use the 'Time:' metadata when relevant. Do not substitute the "
        "current date.\n"
        "5. Try your best to answer. Only respond with 'unknown' if the context contains "
        "absolutely NO relevant information about the topic asked\n"
        "6. For counting questions, answer with just the number (e.g., '2' not 'twice')\n"
        "7. For yes/no questions, start with 'Yes', 'No', 'Likely yes', or 'Likely no'\n"
        "8. When listing multiple items, separate them with commas (e.g., 'item1, item2')\n"
        "9. Return your response in JSON format\n\n"
        "Output Format:\n"
        '{"reasoning": "Brief explanation of your thought process", '
        '"answer": "Concise answer in a short phrase"}'
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": instructions}]
    for image_path in row.get("question_images") or []:
        content.append({"type": "image_url", "image_url": {"url": _data_url(str(image_path))}})
    for item in items:
        raw = item.get("raw_content") or {}
        if isinstance(raw, dict) and raw.get("type") == "image" and raw.get("base64"):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{raw['base64']}"},
                }
            )
    multimodal = len(content) > 1
    user_content: Any = content if multimodal else instructions
    return [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ], multimodal


def extract_answer(raw: str) -> str:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "answer" in payload:
            return str(payload["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict) and "answer" in payload:
                return str(payload["answer"]).strip()
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]*)"[^{}]*\}', raw)
    return match.group(1).strip() if match else raw.strip()


def _validate_inputs(debug_path: Path, predictions_path: Path, expected_top_k: int) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    qa_rows = [row for row in payload.get("rows", []) if row.get("type") == "qa"]
    debug_by_id = {str(row.get("question_id")): row for row in qa_rows}
    predictions = _read_jsonl(predictions_path)
    prediction_by_id = {str(row.get("question_id")): row for row in predictions}
    if len(debug_by_id) != len(qa_rows) or len(prediction_by_id) != len(predictions):
        raise ValueError("duplicate question IDs in replay inputs")
    if set(debug_by_id) != set(prediction_by_id):
        raise ValueError("debug and prediction question IDs do not match")
    for question_id, row in debug_by_id.items():
        items = (row.get("retrieval_result") or {}).get("items") or []
        ids = [str(item.get("id") or "") for item in items]
        if len(ids) != expected_top_k or len(set(ids)) != expected_top_k or not all(ids):
            raise ValueError(f"{question_id}: frozen TopK is invalid: {ids}")
    return predictions, debug_by_id


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from openai import OpenAI

    debug_path = args.debug_trace.resolve()
    predictions_path = args.source_predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, debug_by_id = _validate_inputs(debug_path, predictions_path, args.expected_top_k)
    if args.max_questions is not None:
        predictions = predictions[: args.max_questions]

    manifest = {
        "runner_version": RUNNER_VERSION,
        "status": "running",
        "source_debug_trace": str(debug_path),
        "source_debug_sha256": _sha256(debug_path),
        "source_predictions": str(predictions_path),
        "source_predictions_sha256": _sha256(predictions_path),
        "answer_model": args.model,
        "enable_thinking": False,
        "expected_top_k": args.expected_top_k,
        "retrieval_invoked": False,
        "question_order": [str(row["question_id"]) for row in predictions],
    }
    manifest_path = output_dir / "replay_manifest.json"
    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("source_debug_sha256", "source_predictions_sha256", "answer_model", "expected_top_k"):
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"resume manifest mismatch for {key}")
    _write_json(manifest_path, manifest)

    checkpoint_path = output_dir / "answer_checkpoint.jsonl"
    completed = {
        str(row["question_id"]): row
        for row in (_read_jsonl(checkpoint_path) if checkpoint_path.is_file() else [])
    }
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"empty API key environment variable: {args.api_key_env}")
    # Keep retry ownership in this runner.  The OpenAI SDK otherwise performs
    # its own retries inside every outer attempt, which can leave one large
    # multimodal request apparently stuck for close to an hour.
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )

    for index, source in enumerate(predictions, 1):
        question_id = str(source["question_id"])
        if question_id in completed:
            print(f"[frozen-answer] {index}/{len(predictions)} {question_id} resume=skip", flush=True)
            continue
        debug = debug_by_id[question_id]
        messages, multimodal = build_messages(debug)
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(1, args.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "extra_body": {"enable_thinking": False},
                }
                if not multimodal:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage else {}
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[frozen-answer] {index}/{len(predictions)} {question_id} "
                    f"attempt={attempt}/{args.max_retries} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt == args.max_retries:
                    if not args.continue_on_error:
                        raise
                    raw = None
                    usage = {}
                    break
                time.sleep(min(30.0, 2.0 ** attempt))
        else:  # pragma: no cover
            raise RuntimeError(str(last_error))
        if raw is None:
            failure = {
                "question_id": question_id,
                "index": index,
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error or "unknown error"),
                "attempts": args.max_retries,
                "recorded_at": datetime.now().astimezone().isoformat(),
            }
            with (output_dir / "answer_failures.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(
                f"[frozen-answer] {index}/{len(predictions)} {question_id} "
                "status=deferred continuing_after_error",
                file=sys.stderr,
                flush=True,
            )
            continue
        answer = extract_answer(raw)
        exact, contains = score_open(answer, str(source.get("gt", "")))
        result = dict(source)
        result.update(
            {
                "pred": answer,
                "exact_match": exact,
                "em": 1.0 if exact else 0.0,
                "contains_gt": contains,
                "f1": f1_score(answer, str(source.get("gt", ""))),
                "bleu": bleu_score(answer, str(source.get("gt", ""))),
                "bleu_1": bleu_score(answer, str(source.get("gt", "")), weights=(1, 0, 0, 0)),
                "bleu_2": bleu_score(answer, str(source.get("gt", "")), weights=(0.5, 0.5, 0, 0)),
                "bert": None,
                "judge": None,
                "judge_reasoning": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "usage": usage,
                "answer_replay": {
                    "runner_version": RUNNER_VERSION,
                    "answer_model": args.model,
                    "retrieval_invoked": False,
                    "source_answer_model": debug.get("answer_model"),
                    "frozen_memory_ids": [
                        item["id"] for item in debug["retrieval_result"]["items"]
                    ],
                    "frozen_scores": [
                        item.get("score") for item in debug["retrieval_result"]["items"]
                    ],
                    "raw_response": raw,
                },
            }
        )
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        completed[question_id] = result
        print(
            f"[frozen-answer] {index}/{len(predictions)} {question_id} "
            f"em={result['em']:.0f} f1={result['f1']:.3f}",
            flush=True,
        )

    missing_question_ids = [
        str(row["question_id"])
        for row in predictions
        if str(row["question_id"]) not in completed
    ]
    ordered = [
        completed[str(row["question_id"])]
        for row in predictions
        if str(row["question_id"]) in completed
    ]
    predictions_out = output_dir / "predictions.jsonl"
    predictions_out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in ordered),
        encoding="utf-8",
    )
    summary = summarize_results(ordered)
    final_status = "completed" if not missing_question_ids else "incomplete"
    summary.update(
        {
            "status": final_status,
            "count": len(ordered),
            "expected_count": len(predictions),
            "missing_question_ids": missing_question_ids,
        }
    )
    _write_json(output_dir / "summary.json", summary)
    manifest.update(
        {
            "status": final_status,
            "predictions": str(predictions_out),
            "predictions_sha256": _sha256(predictions_out),
            "count": len(ordered),
            "expected_count": len(predictions),
            "missing_question_ids": missing_question_ids,
        }
    )
    _write_json(manifest_path, manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-trace", type=Path, required=True)
    parser.add_argument("--source-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6-plus-2026-04-02")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--expected-top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)
