#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键模型量化脚本 —— 静态量化 PyTorch 模型为 int8
# 用法: bash scripts/quantize.sh --model_dir exp/ds_tcn [options]

set -euo pipefail

# ======================== 默认参数 ========================
model_dir=exp/ds_tcn          # 模型目录 (含 config.yaml)
checkpoint=                   # 待量化 checkpoint, 默认取 avg_30.pt
test_data=data/test/data.list # 校准用数据集 (量化需代表性数据)
num_workers=8

help_message="Usage: $0 --model_dir exp/ds_tcn [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ -z "${checkpoint}" ]; then
  if [ -f "${model_dir}/avg_30.pt" ]; then
    checkpoint="${model_dir}/avg_30.pt"
  elif [ -f "${model_dir}/final.pt" ]; then
    checkpoint="${model_dir}/final.pt"
  else
    echo "[ERROR] 未指定 checkpoint 且找不到默认模型文件"
    exit 1
  fi
fi

output_model="${model_dir}/$(basename "${checkpoint}" .pt)_int8.zip"

echo "=========================================="
echo "  模型目录: ${model_dir}"
echo "  Checkpoint: ${checkpoint}"
echo "  校准数据: ${test_data}"
echo "  输出模型: ${output_model}"
echo "=========================================="

echo ""
echo "[Quantize] 开始静态量化 ..."

python "${PROJECT_DIR}/wekws/bin/static_quantize.py" \
  --config "${model_dir}/config.yaml" \
  --test_data "${test_data}" \
  --checkpoint "${checkpoint}" \
  --num_workers "${num_workers}" \
  --script_model "${output_model}"

echo ""
echo "[SUCCESS] 量化完成！"
echo "         int8 模型: ${output_model}"
echo "         文件大小: $(ls -lh "${output_model}" | awk '{print $5}')"
