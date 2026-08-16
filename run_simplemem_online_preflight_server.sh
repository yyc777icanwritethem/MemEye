#!/usr/bin/env bash
set -euo pipefail

cd /data/yangyicheng/MemEye_omni_mmdm
mkdir -p runs
exec > runs/simplemem_online_preflight_v1.log 2>&1
source /data/yangyicheng/conda/miniconda3/etc/profile.d/conda.sh
conda activate mmdm
set -a
source /home/yangyicheng/.config/mmdm/writer_qwen.env
set +a
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/data/yangyicheng/huggingface_cache
python -u preflight_simplemem_online.py
