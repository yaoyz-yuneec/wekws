#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键安装脚本 —— 配置 WeKws 完整运行环境
# 自动检测当前设备的 CUDA 版本，若 jetson_local_wheels/ 有本地 whl 则优先使用
# 用法: bash scripts/install.sh [--conda-env wekws] [--python 3.10] [--cuda auto]

set -euo pipefail

# 默认参数
CONDA_ENV="wekws"
PYTHON_VERSION="3.10"
CUDA_VERSION="auto"           # auto: 自动检测 | 12.4 | 12.1 | 11.8 | cpu
INSTALL_TORCH=true
INSTALL_SOX=true
INSTALL_PRE_COMMIT=true

help_message="Usage: $0 [--conda-env wekws] [--python 3.10] [--cuda auto] [--no-torch] [--no-sox] [--no-pre-commit]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_WHEEL_DIR="${PROJECT_DIR}/jetson_local_wheels"

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

# ---------- 3. 自动检测 CUDA 版本 (当 --cuda auto 时) ----------
if [ "${CUDA_VERSION}" = "auto" ]; then
  # 优先从 nvcc 检测
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | \
      sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "[INFO] 从 nvcc 检测到 CUDA ${CUDA_VERSION}"
  # 回退: 从 nvidia-smi 检测
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
      2>/dev/null | head -1 || true)
    if [ -z "${CUDA_VERSION}" ]; then
      CUDA_VERSION="cpu"
      echo "[WARN] 未检测到 CUDA，将安装 CPU 版本 PyTorch"
    fi
  else
    CUDA_VERSION="cpu"
    echo "[WARN] 未检测到 CUDA，将安装 CPU 版本 PyTorch"
  fi
fi

# ---------- 4. 安装 PyTorch (优先使用本地 whl) ----------
if $INSTALL_TORCH; then
  echo "[INFO] 安装 PyTorch (CUDA ${CUDA_VERSION}) ..."

  # 检查本地 whl 目录
  LOCAL_TORCH=$(ls "${LOCAL_WHEEL_DIR}/torch-"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)
  LOCAL_TORCHAUDIO=$(ls "${LOCAL_WHEEL_DIR}/torchaudio-"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)
  LOCAL_TORCHVISION=$(ls "${LOCAL_WHEEL_DIR}/torchvision-"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)

  if [ -n "${LOCAL_TORCH}" ] && [ -n "${LOCAL_TORCHAUDIO}" ]; then
    echo "[INFO] 发现本地 Jetson whl 包，优先从本地安装 (跳过 PyTorch 官方源)"
    PIP_CMDS=()
    PIP_CMDS+=("${LOCAL_TORCH}")
    PIP_CMDS+=("${LOCAL_TORCHAUDIO}")
    [ -n "${LOCAL_TORCHVISION}" ] && PIP_CMDS+=("${LOCAL_TORCHVISION}")
    pip install "${PIP_CMDS[@]}"
  else
    # 无本地 whl，从 PyTorch 官方源安装 (仅 x86_64 有效)
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
  fi

  echo "[INFO] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
  echo "[INFO] CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
fi

# ---------- 5. 安装其他 Python 依赖 ----------
echo "[INFO] 安装 Python 依赖 ..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- 6. 安装本地 onnxruntime (若存在) ----------
LOCAL_ONNX=$(ls "${LOCAL_WHEEL_DIR}/onnxruntime"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)
if [ -n "${LOCAL_ONNX}" ]; then
  echo "[INFO] 发现本地 onnxruntime whl，从本地安装 ..."
  pip install "${LOCAL_ONNX}"
fi

# ---------- 7. 安装 pre-commit ----------
if $INSTALL_PRE_COMMIT; then
  echo "[INFO] 安装 pre-commit hooks ..."
  pre-commit install 2>/dev/null || echo "[WARN] pre-commit install 失败，跳过"
fi

echo ""
echo "[SUCCESS] WeKws 环境配置完成！"
echo "        激活环境: conda activate ${CONDA_ENV}"
echo "        查看用法: bash scripts/pipeline.sh --help"
