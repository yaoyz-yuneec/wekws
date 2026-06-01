#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键安装脚本 —— 配置 WeKws 完整运行环境
# 用法: bash scripts/install.sh [--conda-env wekws] [--python 3.10] [--cuda 12.4]

set -euo pipefail

# 默认参数
CONDA_ENV="wekws"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"
INSTALL_TORCH=true
INSTALL_SOX=true
INSTALL_PRE_COMMIT=true

help_message="Usage: $0 [--conda-env wekws] [--python 3.10] [--cuda 12.4] [--no-torch] [--no-sox] [--no-pre-commit]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ---------- 0. 检查 conda ----------
if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda 未找到，请先安装 Miniconda:"
  echo "        https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
echo "[INFO] conda 已找到: $(conda --version)"

# ---------- 1. 创建/激活 conda 环境 ----------
if conda env list | grep -q "^${CONDA_ENV} "; then
  echo "[INFO] conda 环境 '${CONDA_ENV}' 已存在，跳过创建"
else
  echo "[INFO] 创建 conda 环境 '${CONDA_ENV}' (Python ${PYTHON_VERSION}) ..."
  conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
fi

# 后续命令在 conda 环境下执行
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "[INFO] Python: $(python --version)"

# ---------- 2. 安装 sox ----------
if $INSTALL_SOX; then
  if command -v sox &>/dev/null; then
    echo "[INFO] sox 已安装，跳过"
  else
    echo "[INFO] 安装 sox ..."
    conda install -y conda-forge::sox
  fi
fi

# ---------- 3. 安装 PyTorch ----------
if $INSTALL_TORCH; then
  echo "[INFO] 安装 PyTorch (CUDA ${CUDA_VERSION}) ..."
  case "${CUDA_VERSION}" in
    12.4)
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
      ;;
    12.1)
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
      ;;
    11.8)
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
      ;;
    cpu)
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
      ;;
    *)
      echo "[WARN] 不支持的 CUDA 版本 '${CUDA_VERSION}'，安装最新 PyTorch ..."
      pip install torch torchaudio
      ;;
  esac
  echo "[INFO] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
  echo "[INFO] CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
fi

# ---------- 4. 安装其他 Python 依赖 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[INFO] 安装 Python 依赖 ..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- 5. 安装 pre-commit ----------
if $INSTALL_PRE_COMMIT; then
  echo "[INFO] 安装 pre-commit hooks ..."
  pre-commit install 2>/dev/null || echo "[WARN] pre-commit install 失败，跳过"
fi

echo ""
echo "[SUCCESS] WeKws 环境配置完成！"
echo "        激活环境: conda activate ${CONDA_ENV}"
echo "        查看用法: bash scripts/pipeline.sh --help"
