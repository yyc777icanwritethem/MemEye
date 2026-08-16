"""
SimpleMem / Omni-SimpleMem adapter for MemoryBench.

Wraps `omni_memory.OmniMemoryOrchestrator` (vendored under
`benchmark/simplemem/upstream/OmniSimpleMem`) so it can be selected as a
HistoryMethod via `--method simplemem`.

Design choices:
- **Caption-based ingestion** (mirrors the upstream Mem-Gallery adapter at
  `OmniSimpleMem/benchmarks/memgallery/adapter.py`): we feed each round into
  `orchestrator.add_text()` with the dialogue text + per-image caption lines,
  and tag the MAU with `session_id:` / `image_id:` / `timestamp:`. Raw images
  are not passed to `add_image()` because the upstream image branch would
  re-caption them with `caption_model`, double-counting work and adding cost.
  This matches the `caption_preprocessed: true` convention used by A-MEM.
- **Answer flow**: we call `orchestrator.answer(question, top_k)` directly so
  SimpleMem's intent-aware retrieval planning + its own answer generation
  prompt are exercised end-to-end. The answer LLM is set via
  `OmniMemoryConfig.set_unified_model(model_name)` from the benchmark's
  `--model` flag, keeping it apples-to-apples with other baselines.
- **Visual encoder disabled**: `entropy_trigger.visual_encoder = "none"` so we
  don't pull `openvision-vit-large-patch14-224` (~1.6 GB) for a caption-only
  run. The upstream LoCoMo benchmark uses the same setting.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import REPO_ROOT, write_json
from .dataset import MemoryBenchmarkDataset
from .methods import HistoryMethod


_SIMPLEMEM_ROOT = (
    Path(__file__).resolve().parent / "simplemem" / "upstream" / "OmniSimpleMem"
).resolve()
if str(_SIMPLEMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMPLEMEM_ROOT))


def _load_simplemem_classes() -> Tuple[Any, Any]:
    try:
        from omni_memory import OmniMemoryConfig, OmniMemoryOrchestrator  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Omni-SimpleMem dependencies are not fully installed. Missing module: "
            f"{exc.name}. Verify the upstream clone exists at "
            f"{_SIMPLEMEM_ROOT} and that benchmark/simplemem/upstream/OmniSimpleMem/"
            "omni_memory/core/config.py is present (it is a stopgap reverse-engineered "
            "from tests/test_config.py because upstream gitignored their copy)."
        ) from exc
    return OmniMemoryConfig, OmniMemoryOrchestrator


def _resolve_api_key(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> Optional[str]:
    raw = str(method_config.get("llm_api_key", "")).strip()
    if raw:
        return raw
    raw = str(model_config.get("api_key", "")).strip()
    if raw:
        return raw
    env_name = str(
        method_config.get("llm_api_key_env")
        or model_config.get("api_key_env")
        or "OPENAI_API_KEY"
    ).strip() or "OPENAI_API_KEY"
    return os.getenv(env_name)


def _resolve_base_url(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> Optional[str]:
    raw = str(method_config.get("base_url", "")).strip()
    if raw:
        return raw
    raw = str(model_config.get("base_url", "")).strip()
    return raw or None


def _resolve_model_name(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    explicit = str(method_config.get("llm_model", "")).strip()
    if explicit:
        return explicit
    return (
        str(model_config.get("model", "")).strip()
        or str(model_config.get("name", "")).strip()
        or "gpt-4o-mini"
    )


def _dialogue_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    parts: List[str] = []
    user_text = str(round_payload.get("user", "")).strip()
    assistant_text = str(round_payload.get("assistant", "")).strip()
    if user_text:
        parts.append(f"{speaker_a}: {user_text}")
    if assistant_text:
        parts.append(f"{speaker_b}: {assistant_text}")
    return "\n".join(parts).strip()


def _round_image_blocks(raw_dialogue: Dict[str, Any]) -> List[Tuple[str, str]]:
    input_images = raw_dialogue.get("input_image", []) or []
    captions = raw_dialogue.get("image_caption", []) or []
    image_ids = raw_dialogue.get("image_id", []) or []
    blocks: List[Tuple[str, str]] = []
    for idx, _ in enumerate(input_images):
        image_id = str(image_ids[idx]).strip() if idx < len(image_ids) else ""
        caption = str(captions[idx]).strip() if idx < len(captions) else ""
        blocks.append((image_id, caption))
    return blocks


def _build_round_text(
    round_payload: Dict[str, Any],
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Render one round into a single text blob with caption lines per image."""
    raw_dialogue = round_payload.get("raw", {}) or {}
    text = _dialogue_text(round_payload, speaker_a, speaker_b)
    lines: List[str] = [text] if text else []
    for image_id, caption in _round_image_blocks(raw_dialogue):
        lines.extend(
            [
                "image:",
                f"image_id: {image_id}",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def _build_round_tags(
    round_id: str,
    session_id: str,
    timestamp: Optional[str],
    raw_dialogue: Dict[str, Any],
) -> List[str]:
    tags: List[str] = []
    if session_id:
        tags.append(f"session_id:{session_id}")
    if round_id:
        tags.append(f"round_id:{round_id}")
    if timestamp:
        tags.append(f"timestamp:{timestamp}")
    for image_id, _ in _round_image_blocks(raw_dialogue):
        if image_id:
            tags.append(f"image_id:{image_id}")
    return tags


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    """Append the question's own image caption (if any) to the query string."""
    query = question.strip()
    question_image = str(qa.get("question_image", "")).strip()
    question_caption = qa.get("image_caption")
    if not question_image or not question_caption:
        return query
    if isinstance(question_caption, list):
        caption_text = " ".join(
            str(item).strip() for item in question_caption if str(item).strip()
        )
    else:
        caption_text = str(question_caption).strip()
    if not caption_text:
        return query
    return f"{query}\nquestion's image:\nimage_caption: {caption_text}"


class SimpleMemMethod(HistoryMethod):
    """Omni-SimpleMem as a MemoryBench HistoryMethod.

    Supports two modalities via the ``modality`` config key:

    - ``text_only`` (default): caption-based ingestion.  Images are replaced
      by their ``image_caption`` text.  Requires ``caption_preprocessed: true``.
    - ``multimodal``: text MAUs carry a ``raw_pointer`` to the original image
      and a ``vision_on_demand`` tag so the orchestrator loads them at answer
      time and passes them to the VLM alongside the retrieved text context.
    """

    name = "simplemem"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._orchestrator: Any = None
        self._config_cls: Any = None
        self._orch_cls: Any = None
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._debug_rows: List[Dict[str, Any]] = []
        self._memory_debug_by_id: Dict[str, Dict[str, Any]] = {}
        self._top_k: int = max(1, int(self.config.get("retrieve_k", 20)))
        self._fixed_top_k: bool = bool(self.config.get("fixed_top_k", False))
        self._strict_top_k: bool = bool(self.config.get("strict_top_k", False))
        self._data_dir: Optional[Path] = None
        self._multimodal: bool = str(self.config.get("modality", "text_only")).strip().lower() == "multimodal"
        self._resume_existing: bool = bool(self.config.get("resume_existing", False))
        self._resumed_by_round: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_caption_preprocessed(self, dataset: MemoryBenchmarkDataset) -> None:
        if self._multimodal:
            return  # multimodal mode uses raw images, not captions
        if not bool(self.config.get("caption_preprocessed", True)):
            return
        missing_rounds: List[str] = []
        for round_id, payload in dataset.rounds.items():
            raw = payload.get("raw", {}) or {}
            images = raw.get("input_image", []) or []
            if not images:
                continue
            captions = raw.get("image_caption", []) or []
            if len(captions) != len(images):
                missing_rounds.append(round_id)
        if missing_rounds:
            raise ValueError(
                "SimpleMem caption-based adaptation requires Mem-Gallery-style image captions. "
                "Run the caption preprocess first. Missing/invalid image_caption for rounds: "
                + ", ".join(missing_rounds[:10])
                + ("..." if len(missing_rounds) > 10 else "")
            )

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        safe_task = task_name.lower().replace(" ", "_").replace("/", "_")
        runtime_paths = dict(self.config.get("_runtime_paths", {}))
        output_root = str(runtime_paths.get("output_root", "")).strip()
        if output_root:
            return (Path(output_root) / "_simplemem_debug" / safe_task).resolve()
        return (REPO_ROOT / "output" / safe_task / "simplemem").resolve()

    def _runtime_data_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        safe_task = task_name.lower().replace(" ", "_").replace("/", "_")
        return (REPO_ROOT / "runs" / "simplemem" / safe_task).resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _audit_ingestion(self, dataset: MemoryBenchmarkDataset) -> Dict[str, Any]:
        """Validate the completed Omni snapshot before the first QA call."""
        stored_rows = [r for r in self._debug_rows if r.get("type") == "stored_memory"]
        error_rows = [
            r for r in self._debug_rows
            if r.get("type") in {"ingest_error", "ingest_rejected"}
        ]
        memory_ids = [str(r.get("memory_id", "") or "") for r in stored_rows]
        missing_store_ids = [
            memory_id for memory_id in memory_ids
            if not memory_id or self._orchestrator.mau_store.get(memory_id) is None
        ]
        missing_image_pointers: List[str] = []
        invalid_image_pointers: List[str] = []
        if self._multimodal:
            for row in stored_rows:
                round_id = str(row.get("round_id", "") or "")
                if not (dataset.rounds.get(round_id, {}).get("images", []) or []):
                    continue
                raw_pointer = str(row.get("raw_pointer", "") or "")
                if not raw_pointer:
                    missing_image_pointers.append(round_id)
                elif not Path(raw_pointer).is_file():
                    invalid_image_pointers.append(round_id)
                stored_mau = self._orchestrator.mau_store.get(
                    str(row.get("memory_id", "") or "")
                )
                if (
                    stored_mau is None
                    or str(getattr(stored_mau, "raw_pointer", "") or "") != raw_pointer
                    or "vision_on_demand" not in stored_mau.metadata.tags
                ):
                    invalid_image_pointers.append(round_id)

        vector_count = int(self._orchestrator.vector_store.count())
        graph_checkpoint_ids = self._load_graph_checkpoint_ids()

        audit = {
            "type": "memory_audit",
            "status": "passed",
            "expected_rounds": len(dataset.rounds),
            "stored_memories": len(stored_rows),
            "unique_memory_ids": len(set(memory_ids)),
            "ingest_failures": len(error_rows),
            "missing_store_ids": missing_store_ids,
            "missing_image_pointers": missing_image_pointers,
            "invalid_image_pointers": invalid_image_pointers,
            "vector_count": vector_count,
            "graph_checkpoint_count": len(graph_checkpoint_ids),
        }
        if (
            len(stored_rows) != len(dataset.rounds)
            or len(set(memory_ids)) != len(memory_ids)
            or error_rows
            or missing_store_ids
            or missing_image_pointers
            or invalid_image_pointers
            or vector_count != len(stored_rows)
            or len(graph_checkpoint_ids) != len(stored_rows)
        ):
            audit["status"] = "failed"
        self._debug_rows.append(audit)
        self.runtime_info["memory_audit"] = audit
        self._flush_debug(dataset)
        if audit["status"] != "passed":
            raise RuntimeError(f"SimpleMem memory audit failed: {audit}")
        return audit

    def _graph_checkpoint_path(self) -> Path:
        if self._data_dir is None:
            raise RuntimeError("SimpleMem data directory is not initialized")
        return self._data_dir / "index" / "graph_processed.jsonl"

    def _load_graph_checkpoint_ids(self) -> set[str]:
        path = self._graph_checkpoint_path()
        if not path.exists():
            return set()
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                memory_id = str(row.get("memory_id", "") or "")
                if memory_id:
                    ids.add(memory_id)
        return ids

    def _checkpoint_graph_result(
        self,
        memory_id: str,
        *,
        status: str,
        error: str = "",
        replayed: bool = False,
    ) -> None:
        self._orchestrator.knowledge_graph.save()
        path = self._graph_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "memory_id": memory_id,
                        "status": status,
                        "error": error,
                        "replayed": replayed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    @staticmethod
    def _mau_round_id(mau: Any) -> str:
        tags = list(getattr(getattr(mau, "metadata", None), "tags", []) or [])
        round_ids = [tag.split(":", 1)[1] for tag in tags if tag.startswith("round_id:")]
        if len(round_ids) != 1:
            raise ValueError(
                f"Expected exactly one round_id tag for MAU {getattr(mau, 'id', '')}: {tags}"
            )
        return round_ids[0]

    def _prepare_resume_state(self, dataset: MemoryBenchmarkDataset) -> None:
        """Repair and validate a prefix snapshot without re-running completed writes."""
        existing_maus = list(self._orchestrator.mau_store.iter_all())
        if not existing_maus:
            return
        print(
            f"[SimpleMem] Resume snapshot found: {len(existing_maus)} MAUs; "
            "validating exact dataset prefix..."
        )

        by_round: Dict[str, Any] = {}
        for mau in existing_maus:
            round_id = self._mau_round_id(mau)
            if round_id in by_round:
                raise ValueError(f"Duplicate resumed round_id: {round_id}")
            by_round[round_id] = mau

        expected_order = [
            str(dialogue.get("round", "")).strip()
            for session_id in dataset.session_order()
            for dialogue in dataset.get_session(session_id).get("dialogues", [])
            if str(dialogue.get("round", "")).strip() in dataset.rounds
        ]
        existing_order = [self._mau_round_id(mau) for mau in existing_maus]
        if existing_order != expected_order[: len(existing_order)]:
            raise ValueError(
                "Existing SimpleMem snapshot is not an exact dataset prefix; refusing unsafe resume"
            )

        # Persist the adapter's intended multimodal pointer/tag, including for
        # snapshots created before the post-add_text update fix.
        for round_id, mau in by_round.items():
            images = list(dataset.rounds[round_id].get("images", []) or [])
            changed = False
            if self._multimodal and images and images[0]:
                expected_pointer = str(images[0])
                if mau.raw_pointer != expected_pointer:
                    mau.raw_pointer = expected_pointer
                    changed = True
                if "vision_on_demand" not in mau.metadata.tags:
                    mau.add_tag("vision_on_demand")
                    changed = True
            if changed:
                self._orchestrator.mau_store.update(mau)

        # A hard interruption can leave the append-only MAU store ahead of
        # the periodically saved FAISS index. Rebuild only the derived index.
        vector_items = [
            (str(mau.id), list(mau.embedding))
            for mau in existing_maus
            if mau.embedding
        ]
        expected_vector_ids = [memory_id for memory_id, _ in vector_items]
        current_vector_ids = list(
            self._orchestrator.vector_store._text_store._id_mapping
        )
        if current_vector_ids != expected_vector_ids:
            self._orchestrator.vector_store._text_store.rebuild_index(vector_items)
            self._orchestrator.vector_store._visual_store.rebuild_index([])
            self._orchestrator.save()
        print(
            f"[SimpleMem] Resume vector index repaired: {len(vector_items)} vectors."
        )

        # Knowledge-graph state is now isolated per scenario. Replay only MAUs
        # missing a durable graph checkpoint; summaries/embeddings are reused.
        processed = self._load_graph_checkpoint_ids()
        replayed_count = 0
        for mau in existing_maus:
            memory_id = str(mau.id)
            if memory_id in processed:
                continue
            try:
                entities, relations = self._orchestrator.entity_extractor.extract(mau)
                for entity in entities:
                    self._orchestrator.knowledge_graph.add_extracted_entity(entity)
                for relation in relations:
                    self._orchestrator.knowledge_graph.add_extracted_relation(relation)
                self._checkpoint_graph_result(
                    memory_id, status="completed", replayed=True
                )
            except Exception as exc:
                self._checkpoint_graph_result(
                    memory_id,
                    status="failed",
                    error=str(exc),
                    replayed=True,
                )
            processed.add(memory_id)
            replayed_count += 1
            if replayed_count % 5 == 0 or len(processed) == len(existing_maus):
                print(
                    f"[SimpleMem] Resume graph checkpoints: "
                    f"{len(processed)}/{len(existing_maus)}"
                )

        self._resumed_by_round = by_round
        print(
            f"[SimpleMem] Resume ready: reusing {len(by_round)} completed rounds."
        )

    def _build_orchestrator(self, dataset: MemoryBenchmarkDataset) -> None:
        OmniMemoryConfig, OmniMemoryOrchestrator = _load_simplemem_classes()
        self._config_cls = OmniMemoryConfig
        self._orch_cls = OmniMemoryOrchestrator

        model_config = dict(self.config.get("_model_cfg", {}))
        model_name = _resolve_model_name(self.config, model_config)
        api_key = _resolve_api_key(self.config, model_config)
        if not api_key:
            raise ValueError(
                "OpenAI-compatible API key not found for SimpleMem. "
                "Set OPENAI_API_KEY or pass --model with api_key_env / api_key in the model yaml."
            )
        base_url = _resolve_base_url(self.config, model_config)

        cfg = OmniMemoryConfig.create_default()
        cfg.set_unified_model(model_name)
        cfg.llm.api_key = api_key
        if base_url:
            cfg.llm.api_base_url = base_url
        cfg.llm.temperature = float(self.config.get("temperature", 0.0))
        cfg.llm.max_tokens = int(self.config.get("max_tokens", 1000))

        # Embedding settings (default to MemoryBench's standard)
        cfg.embedding.model_name = str(
            self.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        cfg.embedding.embedding_dim = int(self.config.get("embedding_dim", 384))

        # Caption-only mode: skip CLIP and visual MAUs entirely
        cfg.entropy_trigger.visual_encoder = "none"
        cfg.entropy_trigger.enable_visual_trigger = False

        # Retrieval depth
        cfg.retrieval.default_top_k = self._top_k
        cfg.retrieval.benchmark_fixed_top_k = self._top_k if self._fixed_top_k else None

        # Skip self-evolution (auto-research loop) by default for benchmark fairness
        cfg.enable_self_evolution = bool(self.config.get("enable_self_evolution", False))

        # Per-task data dir; cleared between runs to avoid stale state
        data_dir = self._runtime_data_dir(dataset)
        if data_dir.exists() and not self._resume_existing:
            import shutil

            shutil.rmtree(data_dir, ignore_errors=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir

        self._orchestrator = OmniMemoryOrchestrator(
            config=cfg, data_dir=str(data_dir)
        )
        answer_base_url = str(self.config.get("answer_base_url", "")).strip()
        answer_model = str(self.config.get("answer_model", "")).strip()
        if answer_base_url or answer_model:
            from openai import OpenAI

            answer_key_env = str(
                self.config.get("answer_api_key_env", "LOCAL_QWEN_API_KEY")
            ).strip()
            answer_api_key = os.getenv(answer_key_env, "")
            if not answer_api_key:
                raise ValueError(
                    f"Answer API key environment variable is empty: {answer_key_env}"
                )
            self._orchestrator._benchmark_answer_client = OpenAI(
                api_key=answer_api_key,
                base_url=answer_base_url or base_url,
                timeout=float(self.config.get("answer_timeout", 300)),
            )
            self._orchestrator._benchmark_answer_model = answer_model or model_name
            self.runtime_info["retrieval_model"] = model_name
            self.runtime_info["answer_model"] = answer_model or model_name
            self.runtime_info["answer_base_url"] = answer_base_url or base_url or ""
        if self._resume_existing:
            self._prepare_resume_state(dataset)

        # Caption-only mode: disable modality keyword detection so that words
        # like "image"/"shown"/"visual" in queries don't trigger a VISUAL-only
        # filter that would exclude all TEXT MAUs from retrieval.
        self._orchestrator.query_processor._modality_keywords = {}

    def _ingest_dataset(self, dataset: MemoryBenchmarkDataset) -> None:
        assert self._orchestrator is not None
        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name", "")).strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            timestamp = str(session.get("date", "")).strip() or None
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                raw_dialogue = round_payload.get("raw", {}) or {}
                text = _build_round_text(round_payload, self._speaker_a, self._speaker_b)
                if not text:
                    continue
                tags = _build_round_tags(
                    round_id=round_id,
                    session_id=str(round_payload.get("session_id", "") or session_id),
                    timestamp=timestamp,
                    raw_dialogue=raw_dialogue,
                )
                resumed_mau = self._resumed_by_round.get(round_id)
                if resumed_mau is not None:
                    memory_id = str(resumed_mau.id)
                    mau_tags = list(resumed_mau.metadata.tags or [])
                    self._memory_debug_by_id[memory_id] = {
                        "memory_id": memory_id,
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "raw_pointer": str(resumed_mau.raw_pointer or ""),
                        "mau_tags": mau_tags,
                    }
                    self._debug_rows.append(
                        {
                            "type": "stored_memory",
                            "round_id": round_id,
                            "session_id": round_payload.get("session_id", ""),
                            "timestamp": timestamp or "",
                            "tags": tags,
                            "mau_tags": mau_tags,
                            "text": text,
                            "memory_id": memory_id,
                            "raw_pointer": str(resumed_mau.raw_pointer or ""),
                            "elapsed_ms": 0.0,
                            "reason": "",
                            "resumed": True,
                        }
                    )
                    continue
                started_at = time.perf_counter()
                try:
                    result = self._orchestrator.add_text(text, tags=tags or None, force=True)
                except Exception as exc:
                    self._debug_rows.append(
                        {
                            "type": "ingest_error",
                            "round_id": round_id,
                            "error": str(exc),
                            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                        }
                    )
                    continue
                stored = getattr(result, "success", False) and getattr(result, "mau", None) is not None
                # Multimodal: attach raw image pointer so the orchestrator
                # can load it on-demand at answer time (vision_on_demand flow).
                if stored and self._multimodal:
                    round_images = list(round_payload.get("images", []) or [])
                    if round_images and round_images[0]:
                        result.mau.raw_pointer = str(round_images[0])
                        result.mau.add_tag("vision_on_demand")
                        self._orchestrator.mau_store.update(result.mau)
                if stored:
                    self._orchestrator.save()
                    self._checkpoint_graph_result(
                        str(result.mau.id), status="completed", replayed=False
                    )
                if stored:
                    memory_id = str(getattr(result.mau, "id", "") or "")
                    metadata = getattr(result.mau, "metadata", None)
                    mau_tags = list(getattr(metadata, "tags", []) or [])
                    self._memory_debug_by_id[memory_id] = {
                        "memory_id": memory_id,
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "raw_pointer": str(
                            getattr(result.mau, "raw_pointer", "") or ""
                        ),
                        "mau_tags": mau_tags,
                    }
                self._debug_rows.append(
                    {
                        "type": "stored_memory" if stored else "ingest_rejected",
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp or "",
                        "tags": tags,
                        "text": text,
                        "memory_id": (
                            str(getattr(result.mau, "id", "") or "") if stored else ""
                        ),
                        "raw_pointer": (
                            str(getattr(result.mau, "raw_pointer", "") or "")
                            if stored else ""
                        ),
                        "mau_tags": (
                            list(getattr(getattr(result.mau, "metadata", None), "tags", []) or [])
                            if stored else []
                        ),
                        "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                        "reason": getattr(getattr(result, "trigger_result", None), "reason", "") if not stored else "",
                    }
                )
                if stored:
                    print(
                        f"[SimpleMem] Ingested round {round_id}; "
                        f"new_memory_id={result.mau.id}"
                    )

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._orchestrator is not None and self._dataset_key == dataset_id:
            return

        self._ensure_caption_preprocessed(dataset)
        self._debug_rows = []
        self._memory_debug_by_id = {}
        self._resumed_by_round = {}
        self._build_orchestrator(dataset)
        mode_label = "multimodal" if self._multimodal else "text-only"
        print(
            f"[SimpleMem] Ingesting dataset into Omni-SimpleMem "
            f"(mode={mode_label} model={_resolve_model_name(self.config, dict(self.config.get('_model_cfg', {})))} top_k={self._top_k})..."
        )
        self._ingest_dataset(dataset)
        self._dataset_key = dataset_id
        ingested = sum(1 for r in self._debug_rows if r.get("type") == "stored_memory")
        self.runtime_info["num_memories"] = ingested
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        if self._data_dir is not None:
            self.runtime_info["data_dir"] = str(self._data_dir)
        print(f"[SimpleMem] Memory ready: {ingested} rounds ingested.")
        audit = self._audit_ingestion(dataset)
        print(
            f"[SimpleMem] Memory audit passed: "
            f"{audit['stored_memories']}/{audit['expected_rounds']} rounds."
        )

    # ------------------------------------------------------------------
    # HistoryMethod overrides
    # ------------------------------------------------------------------

    def prepare(self, dataset: MemoryBenchmarkDataset) -> None:
        """Prepare/reuse the complete memory before per-question timing starts."""
        self._ensure_initialized(dataset)

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._orchestrator is not None

        query = _question_with_image_caption(qa, question)
        started_at = time.perf_counter()
        try:
            result = self._orchestrator.answer(
                query,
                top_k=self._top_k,
                question_images=question_images or None,
            )
        except Exception as exc:
            self._debug_rows.append(
                {
                    "type": "answer_error",
                    "question_id": qa.get("question_id", ""),
                    "question": question,
                    "recall_query": query,
                    "question_images": list(question_images or []),
                    "error": str(exc),
                    "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                }
            )
            self._flush_debug(dataset)
            raise

        if isinstance(result, dict):
            answer_text = str(result.get("answer", "") or "").strip()
            sources = result.get("sources") or []
            retrieval_result = result.get("retrieval_result")
        else:
            answer_text = str(result or "").strip()
            sources = []
            retrieval_result = None

        retrieved_memory_metadata: List[Dict[str, Any]] = []
        if isinstance(retrieval_result, dict):
            for item in retrieval_result.get("items", []) or []:
                if not isinstance(item, dict):
                    continue
                memory_id = str(item.get("id", "") or "")
                metadata = dict(self._memory_debug_by_id.get(memory_id, {}))
                metadata["memory_id"] = memory_id
                metadata["score"] = item.get("score")
                retrieved_memory_metadata.append(metadata)

        retrieved_round_ids = list(dict.fromkeys(
            str(item.get("round_id", "") or "")
            for item in retrieved_memory_metadata
            if str(item.get("round_id", "") or "")
        ))
        if self._strict_top_k and len(retrieved_round_ids) != self._top_k:
            raise RuntimeError(
                f"controlled TopK mismatch: requested {self._top_k} unique rounds, "
                f"received {len(retrieved_round_ids)}"
            )

        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": query,
                "question_images": list(question_images or []),
                "num_sources": len(sources) if isinstance(sources, list) else 0,
                "sources": sources if isinstance(sources, list) else [],
                "retrieval_result": retrieval_result,
                "retrieved_memory_metadata": retrieved_memory_metadata,
                "top_k_control": {
                    "requested_unique_rounds": self._top_k,
                    "fixed": self._fixed_top_k,
                    "strict": self._strict_top_k,
                    "actual_memory_items": len(retrieved_memory_metadata),
                    "actual_unique_rounds": len(retrieved_round_ids),
                    "round_ids": retrieved_round_ids,
                },
                "prediction": answer_text,
                "retrieval_model": _resolve_model_name(
                    self.config, dict(self.config.get("_model_cfg", {}))
                ),
                "answer_model": str(
                    getattr(
                        self._orchestrator,
                        "_benchmark_answer_model",
                        self._orchestrator.config.llm.query_model,
                    )
                ),
                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        )
        self._flush_debug(dataset)
        return answer_text

    def build_history(
        self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return []

    def __del__(self) -> None:
        try:
            if self._orchestrator is not None:
                self._orchestrator.close()
        except Exception:
            pass
