# C++ 运行时与 CUDA 部署说明

## 一、ONNX 模型与设备的关系

**导出的 ONNX 模型本身是设备无关的。**

ONNX 只描述计算图（operators 和 tensors），不绑定任何硬件平台。同一个 `.onnx` 文件可以：
- 在 CPU 上运行（ONNX Runtime CPU）
- 在 CUDA GPU 上运行（ONNX Runtime GPU）
- 在 TensorRT、OpenVINO、CoreML 等后端上运行

**CUDA 加速由 ONNX Runtime 的 Execution Provider（执行提供者）在运行时决定**，与模型文件本身的导出方式无关。

---

## 二、项目 C++ 运行时的现状

### 2.1 当前实现

项目的 C++ 运行时位于 `runtime/core/kws/`，核心类是 [`KeywordSpotting`](file:///home/unix_ai/nlp/wekws/runtime/core/kws/keyword_spotting.h)。

关键代码分析：

```cpp
// runtime/core/kws/keyword_spotting.cc - 构造时加载模型
KeywordSpotting::KeywordSpotting(const std::string& model_path) {
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
  // ... 读取 cache_dim / cache_len 元数据
  Reset();
}

// runtime/core/kws/keyword_spotting.cc - 重置缓存（CPU 内存）
void KeywordSpotting::Reset() {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  cache_.resize(cache_dim_ * cache_len_, 0.0);
  const int64_t cache_shape[] = {1, cache_dim_, cache_len_};
  cache_ort_ = Ort::Value::CreateTensor<float>(memory_info, cache_.data(),
                                               cache_.size(), cache_shape, 3);
}
```

当前实现**仅使用 CPU 内存和 CPU Execution Provider**。

### 2.2 CMake 构建配置

[`runtime/core/cmake/onnxruntime.cmake`](file:///home/unix_ai/nlp/wekws/runtime/core/cmake/onnxruntime.cmake) 从 GitHub 下载 **ONNX Runtime 1.12.0 CPU 版本**：

```cmake
set(ONNX_VERSION "1.12.0")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
else()
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
endif()
```

### 2.3 可执行文件用法

**离线推理**（[`kws_main.cc`](file:///home/unix_ai/nlp/wekws/runtime/core/bin/kws_main.cc)）：
```bash
./build/bin/kws_main <fbank_dim> <batch_size> <model.onnx> <test.wav>
# 示例: ./build/bin/kws_main 80 100 avg_30.onnx test.wav
```

**流式推理**（[`stream_kws_main.cc`](file:///home/unix_ai/nlp/wekws/runtime/core/bin/stream_kws_main.cc)）：
```bash
./build/bin/stream_kws_main <fbank_dim> <batch_size> <model.onnx>
# 示例: ./build/bin/stream_kws_main 80 100 avg_30.onnx
```

---

## 三、如何启用 CUDA 加速

### 3.1 修改 C++ 代码

需要修改 [`keyword_spotting.cc`](file:///home/unix_ai/nlp/wekws/runtime/core/kws/keyword_spotting.cc)，在创建 Session 前添加 CUDA Execution Provider。

```cpp
// keyword_spotting.cc
#include "onnxruntime_cxx_api.h"
// 新增 CUDA 头文件
#include "cuda_provider_factory.h"  // onnxruntime-gpu 附带

KeywordSpotting::KeywordSpotting(const std::string& model_path) {
  // 在创建 Session 前添加 CUDA EP
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
      session_options_, 0));  // 0 = GPU device ID

  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
  // ... 其余代码不变
}
```

**注意点**：
- CUDA Execution Provider 的添加必须**在 Session 创建之前**
- 如果在 CUDA 和 CPU 上都可运行，ONNX Runtime 会自动选择 CUDA EP
- Session Options 是 `static` 成员变量，需注意多线程安全

### 3.2 修改 CMake 构建脚本

将 [`onnxruntime.cmake`](file:///home/unix_ai/nlp/wekws/runtime/core/cmake/onnxruntime.cmake) 中的下载链接改为 **onnxruntime-gpu** 版本：

```cmake
# 修改前 (CPU 版本)
set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")

# 修改后 (GPU 版本)
set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz")
```

### 3.3 完整修改示例

以下是 `keyword_spotting.h` 和 `keyword_spotting.cc` 的修改要点：

**keyword_spotting.h** 新增：
```cpp
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace wekws {

class KeywordSpotting {
 public:
  explicit KeywordSpotting(const std::string& model_path,
                           bool use_cuda = false);  // 新增参数
  // ...
};
```

**keyword_spotting.cc** 修改构造：
```cpp
#include "cuda_provider_factory.h"  // 新增

KeywordSpotting::KeywordSpotting(const std::string& model_path,
                                 bool use_cuda) {
  if (use_cuda) {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
        session_options_, 0));
    std::cout << "Using CUDA Execution Provider" << std::endl;
  }
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
  // ...
}
```

### 3.4 编译脚本参考

```bash
# 使用 onnxruntime-gpu 编译
mkdir build && cd build
cmake .. -DONNX_USE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

---

## 四、部署环境要求

### 4.1 软件依赖

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| **onnxruntime-gpu** | ≥ 1.12.0 | 与项目 CMake 配置兼容 |
| **CUDA Toolkit** | ≥ 11.4 | 需与 onnxruntime-gpu 的 CUDA 版本匹配 |
| **cuDNN** | ≥ 8.2 | 需与 CUDA 版本匹配 |

### 4.2 版本兼容性速查

| onnxruntime-gpu | CUDA | cuDNN |
|-----------------|------|-------|
| 1.12.x | 11.4 / 11.6 | 8.2 / 8.4 |
| 1.13.x - 1.15.x | 11.6 / 11.7 / 11.8 | 8.4 / 8.5 |
| 1.16.x - 1.17.x | 11.8 / 12.x | 8.7 / 8.9 |
| 1.18.x+ | 12.x | 8.9+ |

### 4.3 Jetson 平台特殊说明

在 Jetson 设备（Orin、Xavier）上，ONNX Runtime 使用 **TensorRT** 作为 Execution Provider，而非直接使用 CUDA：

```cpp
// Jetson 平台推荐使用 TensorRT EP
#include "tensorrt_provider_factory.h"

Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_TensorRT(
    session_options_, 0));
```

Jetson 上需安装：
- JetPack SDK（自带 CUDA、cuDNN、TensorRT）
- 预编译的 `onnxruntime-gpu` aarch64 wheel（可从项目的 `jetson_local_wheels/` 目录找到）

---

## 五、性能对比参考

| 部署方式 | 延迟（单帧） | 吞吐量 | 适用场景 |
|----------|-------------|--------|----------|
| CPU (ONNX Runtime) | ~3-8 ms | 100-300 帧/秒 | 低成本 CPU 设备 |
| CUDA GPU | ~0.5-2 ms | 1000+ 帧/秒 | 服务器 / 带 GPU 的边缘设备 |
| TensorRT (Jetson) | ~0.3-1 ms | 2000+ 帧/秒 | Jetson Orin / Xavier |

> 以上数据为 FSMN-CTC 模型（2599 输出）的参考值，实际性能取决于具体硬件和 batch size。

---

## 六、FAQ

### Q: 导出的 ONNX 文件是否需要为 CUDA 特别处理？

**不需要。** 导出的 ONNX 是设备无关的，CUDA 加速由 ONNX Runtime 在运行时决定。`export_onnx.py` 的导出结果可直接用于 CPU 和 CUDA 两种场景。

### Q: 为什么导出时设置 `CUDA_VISIBLE_DEVICES=-1`？

这是为了确保 PyTorch 在 CPU 上导出 ONNX。导出的 ONNX 本身不包含设备信息，CPU 导出避免了不必要的 GPU 内存占用，且更稳定。最终 ONNX 文件在 CUDA 上运行时表现一致。

### Q: 如何确认 ONNX 模型在 CUDA 上正常运行？

```bash
# Python 端验证 CUDA 推理
python3 -c "
import onnxruntime as ort
# 创建 CUDA 会话
cuda_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession('model.onnx', providers=cuda_providers)
print('Provider:', sess.get_providers())
"
```

### Q: 当前 C++ 代码的 cache 处理在 CUDA 下需要修改吗？

需要。当前 `keyword_spotting.cc` 使用 CPU 内存分配 cache（`OrtMemTypeCPU`）。在 CUDA 模式下，建议将 cache 分配在 GPU 内存中以避免 CPU-GPU 拷贝开销：

```cpp
// CUDA 模式下使用 GPU 内存
Ort::MemoryInfo memory_info("Cuda", OrtAllocatorType::OrtArenaAllocator);
```

但对于 FSMN 模型（cache 形状为 `[1, hdim, padding, num_layers]`），当前代码仅支持 3 维 cache `[1, dim, len]`。需要在 `Reset()` 和 `Forward()` 中适配 4 维 cache。
