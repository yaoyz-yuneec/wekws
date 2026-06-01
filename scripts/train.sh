#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键模型训练脚本 —— 支持单/多 GPU，支持断点续训
# 支持 max-pooling loss 和 CTC loss 两种训练方式
# 用法: bash scripts/train.sh --config conf/ds_tcn.yaml [options]

set -euo pipefail

# ======================== 默认参数 ========================
config=conf/ds_tcn.yaml     # 模型配置文件
train_data=data/train/data.list
cv_data=data/dev/data.list
gpus="0"                     # GPU 列表, "," 分隔, "-1" 表示 CPU
model_dir=exp/ds_tcn         # 模型保存目录
num_keywords=1               # 唤醒词个数
num_workers=8                # DataLoader 线程数
min_duration=50              # 最小唤醒词时长 (帧)
seed=666                     # 随机种子
cmvn_file=data/train/global_cmvn
norm_mean=true
norm_var=true
checkpoint=                  # 断点续训: 指定 checkpoint
reverb_lmdb=                 # 混响 LMDB (增强)
noise_lmdb=                  # 噪声 LMDB (增强)

help_message="Usage: $0 [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${PROJECT_DIR}/tools:${PATH}"

mkdir -p "${model_dir}"

echo "=========================================="
echo "  配置文件: ${config}"
echo "  模型目录: ${model_dir}"
echo "  训练数据: ${train_data}"
echo "  GPU: ${gpus}"
echo "  Checkpoint: ${checkpoint:-无 (从头训练)}"
echo "=========================================="

# ======================== 训练 ========================
echo ""
echo "[Train] 开始训练 ..."

cmvn_opts=
$norm_mean && cmvn_opts="--cmvn_file ${cmvn_file}"
$norm_var && cmvn_opts="${cmvn_opts} --norm_var"

num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')

if [ "${gpus}" = "-1" ] || [ "${num_gpus}" -eq 0 ]; then
  # ---------- CPU 训练 (单进程) ----------
  echo "[Train] CPU 模式训练 ..."
  python "${PROJECT_DIR}/wekws/bin/train.py" \
    --gpus -1 \
    --config "${config}" \
    --train_data "${train_data}" \
    --cv_data "${cv_data}" \
    --model_dir "${model_dir}" \
    --num_workers "${num_workers}" \
    --num_keywords "${num_keywords}" \
    --min_duration "${min_duration}" \
    --seed "${seed}" \
    --dict ./dict \
    ${cmvn_opts} \
    ${checkpoint:+--checkpoint "${checkpoint}"} \
    ${reverb_lmdb:+--reverb_lmdb "${reverb_lmdb}"} \
    ${noise_lmdb:+--noise_lmdb "${noise_lmdb}"}
else
  # ---------- GPU 训练 (分布式) ----------
  echo "[Train] GPU 模式训练 (${num_gpus} GPUs) ..."
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    "${PROJECT_DIR}/wekws/bin/train.py" \
    --gpus "${gpus}" \
    --config "${config}" \
    --train_data "${train_data}" \
    --cv_data "${cv_data}" \
    --model_dir "${model_dir}" \
    --num_workers "${num_workers}" \
    --num_keywords "${num_keywords}" \
    --min_duration "${min_duration}" \
    --seed "${seed}" \
    --dict ./dict \
    ${cmvn_opts} \
    ${reverb_lmdb:+--reverb_lmdb "${reverb_lmdb}"} \
    ${noise_lmdb:+--noise_lmdb "${noise_lmdb}"} \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

echo ""
echo "[SUCCESS] 训练完成！"
echo "         模型保存目录: ${model_dir}"
echo "         使用 bash scripts/evaluate.sh 评估模型"
