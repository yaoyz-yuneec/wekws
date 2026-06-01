#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键全流程脚本 —— 数据准备 → 训练 → 评估 → 导出 → 量化 → 部署
# 用法: bash scripts/pipeline.sh --dataset hey_snips [options]

set -euo pipefail

# ======================== 默认参数 ========================
# ---------- 数据集 ----------
dataset=hey_snips              # hey_snips | mobvoi | speech_commands
dl_dir=./data/local

# ---------- 模型 ----------
config=conf/ds_tcn.yaml
model_dir=exp/ds_tcn
num_keywords=1

# ---------- 训练 ----------
gpus="0"
num_workers=8
norm_mean=true
norm_var=true
seed=666

# ---------- 评估 ----------
num_average=30
batch_size=256

# ---------- 流程控制 ----------
stage=0                        # 起始阶段
stop_stage=6                   # 终止阶段
                               # 0: 数据准备
                               # 1: 训练
                               # 2: 模型平均
                               # 3: 评分 & DET
                               # 4: 模型导出 (JIT + ONNX)
                               # 5: 模型量化
                               # 6: C++ 运行时编译

help_message="
Usage: bash scripts/pipeline.sh [options]

One-click pipeline for WeKws keyword spotting.

Options:
  --dataset <str>       数据集 (hey_snips|mobvoi|speech_commands)  [default: hey_snips]
  --config <path>       模型配置文件                               [default: conf/ds_tcn.yaml]
  --model_dir <path>    模型保存目录                               [default: exp/ds_tcn]
  --num_keywords <int>  唤醒词个数                                 [default: 1]
  --gpus <str>          GPU 列表 (','分隔, -1 为 CPU)              [default: 0]
  --stage <int>         起始阶段                                   [default: 0]
  --stop_stage <int>    终止阶段                                   [default: 6]
  -h, --help            打印帮助信息

各阶段说明:
  0: 数据准备 (下载 + CMVN + data.list)
  1: 模型训练
  2: 模型平均 (Top-k 平均)
  3: 评分 & DET 曲线
  4: 模型导出 (JIT + ONNX)
  5: 静态量化 (int8)
  6: C++ Runtime 编译
"

# ======================== 参数解析 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${PROJECT_DIR}/tools:${PATH}"

# 解析 stage/stop_stage 等参数
. "${PROJECT_DIR}/tools/parse_options.sh" || exit 1

# 自动设置 checkpoint 路径
score_checkpoint="${model_dir}/avg_${num_average}.pt"

echo "============================================"
echo "  WeKws 一键全流程脚本"
echo "  数据集: ${dataset}  |  模型: ${config}"
echo "  阶段: ${stage} → ${stop_stage}"
echo "============================================"

# ======================== Stage 0: 数据准备 ========================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo ""
  echo "========== Stage 0: 数据准备 =========="

  # 调用 prepare_data 脚本
  bash "${SCRIPT_DIR}/prepare_data.sh" \
    --dataset "${dataset}" \
    --dl_dir "${dl_dir}" \
    --config "${config}" \
    --num_workers "${num_workers}"

  echo "========== Stage 0 完成 =========="
fi

# ======================== Stage 1: 训练 ========================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo ""
  echo "========== Stage 1: 模型训练 =========="

  bash "${SCRIPT_DIR}/train.sh" \
    --config "${config}" \
    --train_data "data/train/data.list" \
    --cv_data "data/dev/data.list" \
    --gpus "${gpus}" \
    --model_dir "${model_dir}" \
    --num_keywords "${num_keywords}" \
    --num_workers "${num_workers}" \
    --seed "${seed}" \
    --norm_mean "${norm_mean}" \
    --norm_var "${norm_var}"

  echo "========== Stage 1 完成 =========="
fi

# ======================== Stage 2: 模型平均 ========================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo ""
  echo "========== Stage 2: 模型平均 =========="

  # 只做平均步骤
  if [ ! -f "${score_checkpoint}" ]; then
    python "${PROJECT_DIR}/wekws/bin/average_model.py" \
      --dst_model "${score_checkpoint}" \
      --src_path "${model_dir}" \
      --num "${num_average}" \
      --val_best
    echo "[Pipeline] 模型平均完成: ${score_checkpoint}"
  else
    echo "[Pipeline] 平均模型已存在，跳过。"
  fi

  echo "========== Stage 2 完成 =========="
fi

# ======================== Stage 3: 评分 & DET ========================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo ""
  echo "========== Stage 3: 评估 =========="

  result_dir="${model_dir}/test_avg_${num_average}"
  mkdir -p "${result_dir}"

  # 评分
  python "${PROJECT_DIR}/wekws/bin/score.py" \
    --config "${model_dir}/config.yaml" \
    --test_data "data/test/data.list" \
    --gpu "${gpu:-0}" \
    --batch_size "${batch_size}" \
    --checkpoint "${score_checkpoint}" \
    --score_file "${result_dir}/score.txt" \
    --dict ./dict \
    --num_workers "${num_workers}"

  # DET 曲线 (每个关键词)
  for keyword in $(tail -n +2 dict/words.txt); do
    python "${PROJECT_DIR}/wekws/bin/compute_det.py" \
      --keyword "${keyword}" \
      --test_data "data/test/data.list" \
      --window_shift 50 \
      --score_file "${result_dir}/score.txt" \
      --stats_file "${result_dir}/stats.${keyword}.txt"
  done

  # 绘制 DET 图
  python "${PROJECT_DIR}/wekws/bin/plot_det_curve.py" \
    --keywords_dict dict/dict.txt \
    --stats_dir "${result_dir}" \
    --figure_file "${result_dir}/det.png" \
    --xlim 10 --x_step 2 --ylim 10 --y_step 2

  echo "[Pipeline] DET 曲线: ${result_dir}/det.png"
  echo "========== Stage 3 完成 =========="
fi

# ======================== Stage 4: 模型导出 ========================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo ""
  echo "========== Stage 4: 模型导出 =========="

  bash "${SCRIPT_DIR}/export_model.sh" \
    --model_dir "${model_dir}" \
    --checkpoint "${score_checkpoint}"

  echo "========== Stage 4 完成 =========="
fi

# ======================== Stage 5: 模型量化 ========================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo ""
  echo "========== Stage 5: 模型量化 =========="

  bash "${SCRIPT_DIR}/quantize.sh" \
    --model_dir "${model_dir}" \
    --checkpoint "${score_checkpoint}" \
    --test_data "data/test/data.list"

  echo "========== Stage 5 完成 =========="
fi

# ======================== Stage 6: 运行时编译 ========================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo ""
  echo "========== Stage 6: C++ Runtime 编译 =========="

  bash "${SCRIPT_DIR}/deploy_runtime.sh" \
    --build_type Release \
    --with_streaming true

  echo "========== Stage 6 完成 =========="
fi

echo ""
echo "============================================"
echo "  全流程执行完毕！"
echo "  模型目录: ${model_dir}/"
echo "  导出模型:"
ls -lh "${model_dir}/"*.onnx "${model_dir}/"*.zip "${model_dir}/"*_int8.zip 2>/dev/null || true
echo "============================================"
