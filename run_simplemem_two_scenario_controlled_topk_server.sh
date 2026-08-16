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

# Top10 is reused from omni_two_scenario_smoke_promptfix_v2: its audit proves
# all eight contexts contain exactly ten distinct rounds.  Only Top16 is new.
for K in 16; do
  LOG="runs/simplemem_two_scenario_controlled_top${K}.log"
  METHOD="config/methods/simplemem_multimodal_qwen_top${K}.yaml"
  OUTPUT="runs/omni_two_scenario_controlled_top${K}"
  {
    echo "stage=controlled_topk status=started top_k=${K} at=$(date -Iseconds)"
    python -u run_benchmark.py \
      --task-config config/tasks/home_renovation_interior_design_smoke.yaml \
      --model-config config/models/qwen3_6_plus_dashscope.yaml \
      --method-config "$METHOD" \
      --output-root "$OUTPUT" \
      --mode open
    python -u run_benchmark.py \
      --task-config config/tasks/personal_health_dashboard_assistant_smoke.yaml \
      --model-config config/models/qwen3_6_plus_dashscope.yaml \
      --method-config "$METHOD" \
      --output-root "$OUTPUT" \
      --mode open
    echo "stage=controlled_topk status=completed top_k=${K} at=$(date -Iseconds)"
  } >>"$LOG" 2>&1
done
