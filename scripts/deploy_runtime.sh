#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键 C++ 运行时部署脚本 —— 编译 ONNX Runtime + WeKws 推理引擎
# 支持 x86_64 和 aarch64 交叉编译
# 用法: bash scripts/deploy_runtime.sh [options]

set -euo pipefail

# ======================== 默认参数 ========================
build_type=Release            # Release | Debug
build_dir=build               # 构建目录
onnx_version=auto             # ONNX Runtime 版本 (auto: 从本地 whl 提取, 否则用 1.12.0)
with_streaming=true           # 编译流式 KWS (依赖 portaudio)
toolchain=                    # 交叉编译 toolchain 文件路径 (aarch64 需指定)

help_message="Usage: $0 [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/runtime"
BUILD_DIR="${RUNTIME_DIR}/${build_dir}"
LOCAL_WHEEL_DIR="${PROJECT_DIR}/jetson_local_wheels"

# ---------- 自动检测本地 onnxruntime whl 并提取版本 ----------
LOCAL_ONNX=$(ls "${LOCAL_WHEEL_DIR}/onnxruntime"*"linux_aarch64.whl" 2>/dev/null | head -1 || true)
if [ "${onnx_version}" = "auto" ]; then
  if [ -n "${LOCAL_ONNX}" ]; then
    # 从 whl 文件名提取版本号: onnxruntime_gpu-1.18.0-cp310-...
    onnx_version=$(basename "${LOCAL_ONNX}" | sed -n 's/.*-\([0-9]*\.[0-9]*\.[0-9]*\)-.*/\1/p')
    echo "[INFO] 从本地 whl 检测到 ONNX Runtime v${onnx_version}"
  else
    onnx_version="1.12.0"
    echo "[INFO] 使用默认 ONNX Runtime v${onnx_version}"
  fi
fi

echo "=========================================="
echo "  构建类型: ${build_type}"
echo "  ONNX 版本: ${onnx_version}"
echo "  流式 KWS: ${with_streaming}"
echo "  Build Dir: ${BUILD_DIR}"
echo "=========================================="

# ======================== 检查依赖 ========================
echo ""
echo "[Deploy] 检查依赖 ..."

if ! command -v cmake &>/dev/null; then
  echo "[ERROR] cmake 未安装. 请先安装: sudo apt install cmake"
  exit 1
fi
echo "[INFO] cmake: $(cmake --version | head -1)"

if ! command -v make &>/dev/null && ! command -v ninja &>/dev/null; then
  echo "[ERROR] 未找到 make 或 ninja 构建工具"
  exit 1
fi

# ======================== 配置 CMake ========================
echo ""
echo "[Deploy] 配置 CMake ..."

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${build_type}"
  -DONNX_VERSION="${onnx_version}"
)

if [ -n "${toolchain}" ]; then
  if [ ! -f "${toolchain}" ]; then
    echo "[ERROR] Toolchain 文件不存在: ${toolchain}"
    exit 1
  fi
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${toolchain}")
  echo "[INFO] 使用交叉编译 toolchain: ${toolchain}"
fi

if $with_streaming; then
  echo "[INFO] 启用流式 KWS (portaudio)"
  # portaudio 所需系统库
  if [ -z "${toolchain}" ]; then
    echo "[INFO] 检查 portaudio 系统依赖 ..."
    for pkg in libasound2-dev; do
      if ! dpkg -s "${pkg}" &>/dev/null; then
        echo "[WARN] 推荐安装 ${pkg}: sudo apt install ${pkg}"
      fi
    done
  fi
fi

# ---------- 以 runtime/onnxruntime/CMakeLists.txt 作为根 ----------
cmake -S "${RUNTIME_DIR}/onnxruntime" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

# ======================== 编译 ========================
echo ""
echo "[Deploy] 编译 WeKws Runtime ..."
NUM_PROC=$(nproc 2>/dev/null || echo 4)
cmake --build "${BUILD_DIR}" -- -j "${NUM_PROC}"

echo ""
echo "[Deploy] 编译完成。生成的可执行文件:"
echo "----------------------------------------"
ls -lh "${BUILD_DIR}/bin/" 2>/dev/null || ls -lh "${BUILD_DIR}/"*.out 2>/dev/null || true

# ======================== 验证 ========================
echo ""
echo "[Deploy] 验证 ..."
if [ -f "${BUILD_DIR}/bin/kws_main" ]; then
  echo "[INFO] kws_main (离线 KWS) 编译成功"
fi
if [ -f "${BUILD_DIR}/bin/stream_kws_main" ]; then
  echo "[INFO] stream_kws_main (在线流式 KWS) 编译成功"
fi
if [ -f "${BUILD_DIR}/fc_base/onnxruntime-src/lib/libonnxruntime.so*" ] || \
   [ -f "${BUILD_DIR}/_deps/onnxruntime-src/lib/libonnxruntime.so*" ]; then
  echo "[INFO] ONNX Runtime 库已下载并链接"
fi

echo ""
echo "[SUCCESS] C++ Runtime 部署完成！"
echo "         构建目录: ${BUILD_DIR}/"
echo "         kws_main 用法: ${BUILD_DIR}/bin/kws_main <fbank_dim> <batch_size> <onnx_model_path>"
echo "         stream_kws_main 用法: ${BUILD_DIR}/bin/stream_kws_main <fbank_dim> <batch_size> <onnx_model_path>"
