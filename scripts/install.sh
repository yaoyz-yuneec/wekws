#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键安装脚本 —— 配置 WeKws 完整运行环境
# 自动检测当前设备的 CUDA 版本，若 jetson_local_wheels/ 有本地 whl 则优先使用
# 环境管理: conda > venv, 优先使用 conda, 不存在则自动创建 venv
# 用法: bash scripts/install.sh [--env-name wekws] [--python 3.10] [--cuda auto]

set -euo pipefail

# 默认参数
ENV_NAME="wekws"             # conda 环境名 或 venv 目录名
PYTHON_VERSION="3.10"
CUDA_VERSION="auto"          # auto: 自动检测 | 12.4 | 12.1 | 11.8 | cpu
INSTALL_TORCH=true
INSTALL_SOX=true
INSTALL_PRE_COMMIT=true
USE_CONDA="auto"             # auto: 有 conda 则用, 否则 venv | true: 强制 conda | false: 强制 venv

help_message="Usage: $0 [--env-name wekws] [--python 3.10] [--cuda auto] [--use-conda auto] [--no-torch] [--no-sox] [--no-pre-commit]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_WHEEL_DIR="${PROJECT_DIR}/jetson_local_wheels"

# ---------- 0. 确定环境管理方式: conda or venv ----------
HAVE_CONDA=false
if command -v conda &>/dev/null; then
  HAVE_CONDA=true
  echo "[INFO] conda 已找到: $(conda --version)"
fi

USE_CONDA_BOOL=false
if [ "${USE_CONDA}" = "true" ]; then
  USE_CONDA_BOOL=true
elif [ "${USE_CONDA}" = "auto" ] && $HAVE_CONDA; then
  USE_CONDA_BOOL=true
fi

ENV_DIR="${PROJECT_DIR}/${ENV_NAME}"  # venv 时使用

if $USE_CONDA_BOOL; then
  # ==================== Conda 模式 ====================
  if ! $HAVE_CONDA; then
    echo "[ERROR] 强制 conda 模式但未找到 conda，请先安装 Miniconda:"
    echo "        https://docs.conda.io/en/latest/miniconda.html"
    exit 1
  fi

  if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] conda 环境 '${ENV_NAME}' 已存在，跳过创建"
  else
    echo "[INFO] 创建 conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
  fi

  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
  echo "[INFO] Python: $(python --version)"
  PYTHON_BIN="$(which python)"

  # ---------- Conda 模式下安装 sox ----------
  if $INSTALL_SOX; then
    if command -v sox &>/dev/null; then
      echo "[INFO] sox 已安装，跳过"
    else
      echo "[INFO] 安装 sox ..."
      conda install -y conda-forge::sox
    fi
  fi
else
  # ==================== Venv 模式 ====================
  echo "[INFO] 使用 Python venv 模式"

  # 检查系统 Python 版本
  SYS_PYTHON=$(command -v python3 || command -v python)
  if [ -z "${SYS_PYTHON}" ]; then
    echo "[ERROR] 未找到 python3"
    exit 1
  fi
  SYS_PY_VER=$("${SYS_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "[INFO] 系统 Python: ${SYS_PY_VER} (${SYS_PYTHON})"

  # 检查 venv 模块
  if ! "${SYS_PYTHON}" -c 'import venv' &>/dev/null; then
    echo "[INFO] venv 模块未安装，尝试安装 python3-venv ..."
    sudo apt-get install -y python3-venv 2>/dev/null || {
      echo "[WARN] 无法安装 python3-venv，使用 pip install virtualenv 替代"
      pip3 install virtualenv --user
      VIRTUALENV_INSTALLED=true
    }
  fi

  if [ -d "${ENV_DIR}" ]; then
    echo "[INFO] venv '${ENV_DIR}' 已存在，跳过创建"
  else
    echo "[INFO] 创建 venv '${ENV_DIR}' ..."
    if [ "${VIRTUALENV_INSTALLED:-false}" = "true" ]; then
      python3 -m virtualenv "${ENV_DIR}"
    else
      "${SYS_PYTHON}" -m venv "${ENV_DIR}"
    fi
  fi

  source "${ENV_DIR}/bin/activate"
  echo "[INFO] Python: $(python --version)"
  PYTHON_BIN="$(which python)"

  # ---------- Venv 模式下安装 sox (系统 apt) ----------
  if $INSTALL_SOX; then
    if command -v sox &>/dev/null; then
      echo "[INFO] sox 已安装，跳过"
    else
      echo "[INFO] 尝试安装 sox (系统 apt) ..."
      sudo apt-get install -y sox libsox-dev 2>/dev/null || \
        echo "[WARN] sox 安装失败，请手动安装: sudo apt install sox"
    fi
  fi
fi

# ---------- 自动检测 CUDA 版本 (当 --cuda auto 时) ----------
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

# ---------- 安装 PyTorch (优先使用本地 whl) ----------
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

# ---------- 安装其他 Python 依赖 ----------
echo "[INFO] 安装 Python 依赖 ..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- 安装本地 onnxruntime (若存在) ----------
LOCAL_ONNX=$(ls "${LOCAL_WHEEL_DIR}/onnxruntime"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)
if [ -n "${LOCAL_ONNX}" ]; then
  echo "[INFO] 发现本地 onnxruntime whl，从本地安装 ..."
  pip install "${LOCAL_ONNX}"
fi

# ---------- 安装 pre-commit ----------
if $INSTALL_PRE_COMMIT; then
  echo "[INFO] 安装 pre-commit hooks ..."
  pre-commit install 2>/dev/null || echo "[WARN] pre-commit install 失败，跳过"
fi

echo ""
if $USE_CONDA_BOOL; then
  echo "[SUCCESS] WeKws 环境配置完成！(conda)"
  echo "        激活环境: conda activate ${ENV_NAME}"
else
  echo "[SUCCESS] WeKws 环境配置完成！(venv)"
  echo "        激活环境: source ${ENV_DIR}/bin/activate"
fi
echo "        查看用法: bash scripts/pipeline.sh --help"
