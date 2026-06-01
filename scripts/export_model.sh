#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键模型导出脚本 —— 导出 TorchScript JIT 和 ONNX 格式
# 用法: bash scripts/export_model.sh --model_dir exp/ds_tcn [options]

set -euo pipefail

# ======================== 默认参数 ========================
model_dir=exp/ds_tcn          # 模型目录 (含 config.yaml)
checkpoint=                   # 待导出 checkpoint，默认取 avg_30.pt
export_jit=true               # 导出 JIT 模型
export_onnx=true              # 导出 ONNX 模型

help_message="Usage: $0 --model_dir exp/ds_tcn [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

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

echo "=========================================="
echo "  模型目录: ${model_dir}"
echo "  Checkpoint: ${checkpoint}"
echo "  导出 JIT: ${export_jit}"
echo "  导出 ONNX: ${export_onnx}"
echo "=========================================="

# ======================== 导出 JIT ========================
if $export_jit; then
  jit_model="${model_dir}/$(basename "${checkpoint}" | sed -e 's:.pt$:.zip:g')"
  echo ""
  echo "[Export] 导出 TorchScript JIT 模型 ..."
  python "${PROJECT_DIR}/wekws/bin/export_jit.py" \
    --config "${model_dir}/config.yaml" \
    --checkpoint "${checkpoint}" \
    --jit_model "${jit_model}"
  echo "[Export] JIT 模型已导出: ${jit_model}"
fi

# ======================== 导出 ONNX ========================
if $export_onnx; then
  onnx_model="${model_dir}/$(basename "${checkpoint}" | sed -e 's:.pt$:.onnx:g')"
  echo ""
  echo "[Export] 导出 ONNX 模型 ..."
  python "${PROJECT_DIR}/wekws/bin/export_onnx.py" \
    --config "${model_dir}/config.yaml" \
    --checkpoint "${checkpoint}" \
    --onnx_model "${onnx_model}"

  echo "[Export] ONNX 模型已导出: ${onnx_model}"

  # 打印模型信息
  python -c "
import onnx
m = onnx.load('${onnx_model}')
print('[Export] ONNX 模型信息:')
print(f'  IR 版本: {m.ir_version}')
print(f'  Opset 版本: {m.opset_import[0].version}')
for p in m.metadata_props:
    print(f'  {p.key}: {p.value}')
  "
fi

echo ""
echo "[SUCCESS] 模型导出完成！"
echo "         JIT:  ${jit_model:-未导出}"
echo "         ONNX: ${onnx_model:-未导出}"
