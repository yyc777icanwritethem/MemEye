#!/usr/bin/env bash
set -euo pipefail

cd /data/yangyicheng/MemEye_omni_mmdm
mkdir -p runs
source /data/yangyicheng/conda/miniconda3/etc/profile.d/conda.sh
conda activate mmdm
set -a
source /home/yangyicheng/.config/mmdm/writer_qwen.env
set +a
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/data/yangyicheng/huggingface_cache
export HF_HUB_CACHE=/data/yangyicheng/huggingface_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for K in 10 16; do
  LOG="runs/simplemem_two_scenario_full_controlled_top${K}.log"
  METHOD="config/methods/simplemem_multimodal_qwen_top${K}.yaml"
  OUTPUT="runs/omni_two_scenario_full_controlled_top${K}"
  {
    echo "stage=full_controlled_topk status=started top_k=${K} at=$(date -Iseconds)"
    python -u run_benchmark.py \
      --task-config config/tasks/home_renovation_interior_design.yaml \
      --model-config config/models/qwen3_6_plus_dashscope.yaml \
      --method-config "$METHOD" \
      --output-root "$OUTPUT" \
      --mode open
    python -u run_benchmark.py \
      --task-config config/tasks/personal_health_dashboard_assistant.yaml \
      --model-config config/models/qwen3_6_plus_dashscope.yaml \
      --method-config "$METHOD" \
      --output-root "$OUTPUT" \
      --mode open
    echo "stage=full_controlled_topk status=completed top_k=${K} at=$(date -Iseconds)"
  } >>"$LOG" 2>&1
done
