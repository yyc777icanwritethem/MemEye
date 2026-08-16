# Controlled Omni-SimpleMem adapter for MMDM comparison

This branch keeps the vendored official Omni-SimpleMem orchestrator as the
memory and retrieval implementation while adding the benchmark controls needed
for the MMDM comparison.

The adapter calls `OmniMemoryOrchestrator.add_text()` for memory construction
and `OmniMemoryOrchestrator.answer()` for QA. Native query processing, vector
retrieval, BM25 supplementation, graph and parametric recall, expansion, and
context formatting remain active.

Controlled-run additions are:

- per-scenario storage isolation, prefix-safe resume and durable checkpoints;
- ingestion audit and per-question resume;
- fixed, strict Top10/Top16 unique-round controls;
- source IDs, scores, raw pointers, model identities and debug timing;
- an optional separate answer endpoint so both compared methods can use local
  `Qwen/Qwen2.5-VL-7B-Instruct` without changing Omni's retrieval-side model;
- removal of an incorrect hard-coded assumption that all conversations occur
  in 2023.

The controlled results should therefore be described as an **adapted official
Omni-SimpleMem baseline**, not a bit-for-bit untouched upstream run. Logging and
provenance do not replace retrieval logic; fixed TopK and answer-model splitting
are intentional experimental interventions.

Generated `runs/`, `output/`, logs, archives, predictions and debug traces are
excluded from Git. Reproduction uses the committed configs and server scripts;
aggregate results and server artifact paths live in the MMDM repository's
`docs/MEMEYE_OMNI_MMDM_FULL_RESULTS_20260817.md`.
