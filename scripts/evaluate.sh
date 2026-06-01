#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键模型评估脚本 —— 模型平均、评分、DET 曲线
# 支持 max-pooling loss 和 CTC loss 两种评估方式
# 用法: bash scripts/evaluate.sh --model_dir exp/ds_tcn [options]

set -euo pipefail

# ======================== 默认参数 ========================
model_dir=exp/ds_tcn        # 模型目录 (含 config.yaml)
test_data=data/test/data.list
gpu=0
batch_size=256
num_workers=8
num_average=30              # 平均检查点个数
score_checkpoint=            # 若为空则自动为 ${model_dir}/avg_${num_average}.pt
criterion=max_pooling        # max_pooling | ctc
window_shift=50              # 滑窗步长 (帧)
dict=./dict
token_file=                  # CTC 模式: tokens.txt
lexicon_file=                # CTC 模式: lexicon.txt
keywords=                    # CTC 模式: 关键词 (逗号分隔)

help_message="Usage: $0 --model_dir exp/ds_tcn [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${PROJECT_DIR}/tools:${PATH}"

# 自动设置评分 checkpoint
if [ -z "${score_checkpoint}" ]; then
  score_checkpoint="${model_dir}/avg_${num_average}.pt"
fi

result_dir="${model_dir}/test_$(basename "${score_checkpoint}")"
mkdir -p "${result_dir}"

echo "=========================================="
echo "  模型目录: ${model_dir}"
echo "  Checkpoint: ${score_checkpoint}"
echo "  评估数据: ${test_data}"
echo "  损失类型: ${criterion}"
echo "=========================================="

# ======================== Stage 1: 模型平均 ========================
echo ""
echo "[Eval] Stage 1: 模型平均 (Top-${num_average}) ..."
if [ ! -f "${score_checkpoint}" ]; then
  python "${PROJECT_DIR}/wekws/bin/average_model.py" \
    --dst_model "${score_checkpoint}" \
    --src_path "${model_dir}" \
    --num "${num_average}" \
    --val_best
  echo "[Eval] 模型平均完成: ${score_checkpoint}"
else
  echo "[Eval] 平均模型已存在，跳过。"
fi

# ======================== Stage 2: 评分 ========================
echo ""
echo "[Eval] Stage 2: 评分 (生成 score.txt) ..."

score_prefix=""
score_script="score.py"
if [ "${criterion}" = "ctc" ]; then
  score_prefix="stream_"
  score_script="score_ctc.py"
fi

python "${PROJECT_DIR}/wekws/bin/${score_prefix}${score_script}" \
  --config "${model_dir}/config.yaml" \
  --test_data "${test_data}" \
  --gpu "${gpu}" \
  --batch_size "${batch_size}" \
  --num_workers "${num_workers}" \
  --checkpoint "${score_checkpoint}" \
  --score_file "${result_dir}/score.txt" \
  --dict "${dict}" \
  ${keywords:+--keywords "${keywords}"} \
  ${token_file:+--token_file "${token_file}"} \
  ${lexicon_file:+--lexicon_file "${lexicon_file}"}

echo "[Eval] 评分完成: ${result_dir}/score.txt"

# ======================== Stage 3: 计算 DET / 准确率 ========================
echo ""
echo "[Eval] Stage 3: 计算检测指标 ..."

if [ "${criterion}" = "ctc" ]; then
  # CTC 模式: compute_det_ctc.py
  python "${PROJECT_DIR}/wekws/bin/compute_det_ctc.py" \
    --keywords "${keywords}" \
    --test_data "${test_data}" \
    --window_shift "${window_shift}" \
    --step 0.001 \
    --score_file "${result_dir}/score.txt" \
    --token_file "${token_file}" \
    --lexicon_file "${lexicon_file}"
else
  # Max-pooling 模式: 逐个关键词计算 DET
  for keyword in $(tail -n +2 "${dict}/words.txt"); do
    echo "[Eval]   关键词: ${keyword}"
    python "${PROJECT_DIR}/wekws/bin/compute_det.py" \
      --keyword "${keyword}" \
      --test_data "${test_data}" \
      --window_shift "${window_shift}" \
      --score_file "${result_dir}/score.txt" \
      --stats_file "${result_dir}/stats.${keyword}.txt"
  done

  # 绘制 DET 曲线
  python "${PROJECT_DIR}/wekws/bin/plot_det_curve.py" \
    --keywords_dict "${dict}/dict.txt" \
    --stats_dir "${result_dir}" \
    --figure_file "${result_dir}/det.png" \
    --xlim 10 --x_step 2 --ylim 10 --y_step 2

  echo "[Eval] DET 曲线: ${result_dir}/det.png"
fi

echo ""
echo "[SUCCESS] 评估完成！"
echo "         结果目录: ${result_dir}/"
echo "         Score 文件: ${result_dir}/score.txt"
echo "         DET 曲线: ${result_dir}/det.png"
