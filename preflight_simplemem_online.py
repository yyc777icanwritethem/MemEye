"""One-write server-only preflight for Omni-SimpleMem and DashScope."""

import os
import time
from pathlib import Path

from benchmark.simplemem import _load_simplemem_classes


def main() -> None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    config_cls, orchestrator_cls = _load_simplemem_classes()
    cfg = config_cls.create_default()
    cfg.set_unified_model("qwen3.6-plus-2026-04-02")
    cfg.llm.api_key = api_key
    cfg.llm.api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    cfg.llm.temperature = 0.0
    cfg.llm.max_tokens = 1000
    cfg.embedding.model_name = "all-MiniLM-L6-v2"
    cfg.embedding.embedding_dim = 384
    cfg.entropy_trigger.visual_encoder = "none"
    cfg.entropy_trigger.enable_visual_trigger = False
    cfg.enable_self_evolution = False

    run_dir = Path("runs/preflight") / f"simplemem_online_{int(time.time())}"
    orchestrator = orchestrator_cls(config=cfg, data_dir=str(run_dir))
    try:
        result = orchestrator.add_text(
            "user: We are checking a kitchen renovation memory.\n"
            "assistant: The cabinet finish is matte white.",
            tags=["preflight", "round_id:PREFLIGHT_R1"],
            force=True,
        )
        stored = bool(
            getattr(result, "success", False)
            and getattr(result, "mau", None) is not None
        )
        if not stored:
            raise RuntimeError("Omni add_text did not return a stored MAU")
        memory_id = str(result.mau.id)
        if orchestrator.mau_store.get(memory_id) is None:
            raise RuntimeError("Stored MAU cannot be read back from mau_store")
        print(f"simplemem_online_preflight_ok memory_id={memory_id} run_dir={run_dir}")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
