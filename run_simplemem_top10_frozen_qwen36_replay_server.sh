#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yangyicheng/MemEye_omni_mmdm
PYTHON=/data/yangyicheng/conda/miniconda3/envs/mmdm/bin/python
SOURCE="$ROOT/runs/omni_two_scenario_full_controlled_top10_qwen25vl7b_split_v2_answers"
OUTPUT="$ROOT/runs/omni_top10_frozen_retrieval_qwen36_v1"

set -a
source /home/yangyicheng/.config/mmdm/writer_qwen.env
set +a
mkdir -p "$OUTPUT/logs"

run_home() {
  "$PYTHON" -u "$ROOT/replay_simplemem_frozen_answers.py" \
    --debug-trace "$SOURCE/_simplemem_debug/home_renovation_interior_design/debug_trace.json" \
    --source-predictions "$SOURCE/home_renovation_interior_design/20260816_225711_qwen3_6_plus_dashscope_simplemem__multimodal/predictions.jsonl" \
    --output-dir "$OUTPUT/home_renovation_interior_design"
}

run_health() {
  "$PYTHON" -u "$ROOT/replay_simplemem_frozen_answers.py" \
    --debug-trace "$SOURCE/_simplemem_debug/personal_health_dashboard_assistant/debug_trace.json" \
    --source-predictions "$SOURCE/personal_health_dashboard_assistant/20260816_225711_qwen3_6_plus_dashscope_simplemem__multimodal/predictions.jsonl" \
    --output-dir "$OUTPUT/personal_health_dashboard_assistant"
}

echo "stage=frozen_qwen36_top10 status=started at=$(date -Iseconds)"
run_home >"$OUTPUT/logs/home.log" 2>&1 &
home_pid=$!
run_health >"$OUTPUT/logs/health.log" 2>&1 &
health_pid=$!
echo "home_pid=$home_pid health_pid=$health_pid"

home_status=0
health_status=0
wait "$home_pid" || home_status=$?
wait "$health_pid" || health_status=$?
echo "stage=home status=$home_status at=$(date -Iseconds)"
echo "stage=health status=$health_status at=$(date -Iseconds)"
if [[ "$home_status" -ne 0 || "$health_status" -ne 0 ]]; then
  exit 1
fi
echo "stage=frozen_qwen36_top10 status=completed at=$(date -Iseconds)"
