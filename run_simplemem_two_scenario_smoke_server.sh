#!/usr/bin/env bash
set -euo pipefail

cd /data/yangyicheng/MemEye_omni_mmdm
mkdir -p runs
exec > runs/simplemem_two_scenario_smoke_v1.log 2>&1
source /data/yangyicheng/conda/miniconda3/etc/profile.d/conda.sh
conda activate mmdm
set -a
source /home/yangyicheng/.config/mmdm/writer_qwen.env
set +a
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/data/yangyicheng/huggingface_cache

echo "stage=home status=started"
python -u run_benchmark.py \
  --task-config config/tasks/home_renovation_interior_design_smoke.yaml \
  --model-config config/models/qwen3_6_plus_dashscope.yaml \
  --method-config config/methods/simplemem_multimodal_qwen.yaml \
  --output-root runs/omni_two_scenario_smoke_v1 \
  --mode open
echo "stage=home status=completed"

echo "stage=health status=started"
python -u run_benchmark.py \
  --task-config config/tasks/personal_health_dashboard_assistant_smoke.yaml \
  --model-config config/models/qwen3_6_plus_dashscope.yaml \
  --method-config config/methods/simplemem_multimodal_qwen.yaml \
  --output-root runs/omni_two_scenario_smoke_v1 \
  --mode open
echo "stage=health status=completed"

echo "stage=all status=completed"
