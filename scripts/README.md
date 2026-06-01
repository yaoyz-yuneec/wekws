# WeKws 一键脚本使用指南

## 目录概览

```
scripts/
├── install.sh           # 环境安装（conda/venv + PyTorch + 依赖）
├── prepare_data.sh      # 数据准备（下载 + CMVN + data.list）
├── train.sh             # 模型训练（单/多 GPU，断点续训）
├── evaluate.sh          # 模型评估（平均 + 评分 + DET 曲线）
├── export_model.sh      # 模型导出（TorchScript JIT + ONNX）
├── quantize.sh          # 静态量化（float32 → int8）
├── deploy_runtime.sh    # C++ 运行时编译（ONNX Runtime）
├── pipeline.sh          # 🔥 一键全流程（0→6 阶段依次执行）
└── README.md            # 本文件
```

所有脚本均使用项目统一的 `tools/parse_options.sh` 解析参数，可通过 `--help` 查看完整选项。

---

## 快速开始（全流程）

```bash
# 1. 进入项目根目录
cd /path/to/wekws

# 2. 安装环境（自动检测 CUDA，有本地 whl 则优先使用）
bash scripts/install.sh

# 3. 激活虚拟环境
#    - 若使用 conda:  conda activate wekws_env
#    - 若使用 venv:   source wekws_env/bin/activate

# 4. 一键全流程（以 Hey Snips 数据集为例）
bash scripts/pipeline.sh --dataset hey_snips

# 仅运行特定阶段（如训练→评估）
bash scripts/pipeline.sh --stage 1 --stop_stage 3
```

---

## 各脚本详细说明

### 1. install.sh — 环境安装

自动配置 Python 运行环境，支持 conda 和 venv 两种模式。

```bash
bash scripts/install.sh [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--env-name` | `wekws_env` | conda 环境名 或 venv 目录名 |
| `--python` | `3.10` | Python 版本 |
| `--cuda` | `auto` | CUDA 版本（auto: 自动检测 / 12.4 / 12.1 / 11.8 / cpu） |
| `--use-conda` | `auto` | 环境管理方式（auto: 优先 conda / true: 强制 conda / false: 强制 venv） |
| `--no-torch` | — | 跳过 PyTorch 安装 |
| `--no-sox` | — | 跳过 sox 安装 |
| `--no-pre-commit` | — | 跳过 pre-commit 安装 |

**特性说明：**
- 自动检测 CUDA 版本（`nvcc` → `nvidia-smi` → 回退 CPU）
- 优先安装 `jetson_local_wheels/` 下的本地 whl 包（适用于 Jetson 等 aarch64 设备）
- 若本地有 `onnxruntime_gpu` whl 也会自动安装

---

### 2. prepare_data.sh — 数据准备

下载数据集、提取音频、生成 Kaldi 格式数据和 data.list。

```bash
bash scripts/prepare_data.sh --dataset <数据集> [选项]
```

**支持的数据集：**

| 数据集 | `--dataset` 参数 | 说明 |
|--------|-------------------|------|
| Hey Snips | `hey_snips` | 需手动下载，填入 Google Form 后获取 |
| Mobvoi（Hi Xiaowen） | `mobvoi` | 自动从 OpenSLR 下载（openslr.org/87） |
| Google Speech Commands v1 | `speech_commands` | 自动下载 |

**选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `hey_snips` | 数据集名称 |
| `--dl_dir` | `./data/local` | 下载目录 |
| `--output_dir` | `./data` | 输出目录 |
| `--config` | `./conf/ds_tcn.yaml` | 模型配置文件（用于 CMVN） |
| `--num_workers` | `16` | CMVN 计算并行数 |
| `--nj` | `8` | wav_to_duration 并行数 |

**执行流程：**
1. 下载并解压数据集
2. 生成 dictionary（dict/dict.txt, dict/words.txt）
3. 转换为 Kaldi 格式（wav.scp, text）
4. 计算全局 CMVN 统计量
5. 生成 data.list（供训练/评估使用）

---

### 3. train.sh — 模型训练

支持单/多 GPU 分布式训练、断点续训、数据增强。

```bash
bash scripts/train.sh [选项]
```

**选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `conf/ds_tcn.yaml` | 模型配置文件路径 |
| `--train_data` | `data/train/data.list` | 训练数据 |
| `--cv_data` | `data/dev/data.list` | 验证数据 |
| `--gpus` | `0` | GPU 列表（逗号分隔，`-1` 表示 CPU） |
| `--model_dir` | `exp/ds_tcn` | 模型保存目录 |
| `--num_keywords` | `1` | 唤醒词个数 |
| `--num_workers` | `8` | DataLoader 线程数 |
| `--seed` | `666` | 随机种子 |
| `--checkpoint` | — | 断点续训 checkpoint |
| `--cmvn_file` | `data/train/global_cmvn` | CMVN 文件 |
| `--norm_mean` | `true` | 是否做均值归一化 |
| `--norm_var` | `true` | 是否做方差归一化 |
| `--reverb_lmdb` | — | 混响增强 LMDB |
| `--noise_lmdb` | — | 噪声增强 LMDB |

**配置文件说明（YAML）：**

```yaml
# conf/ds_tcn.yaml 示例
dataset_conf:
    fbank_conf:
        num_mel_bins: 40          # FBank 维度
        frame_shift: 10           # 帧移 (ms)
        frame_length: 25          # 帧长 (ms)
    batch_conf:
        batch_size: 256

model:
    hidden_dim: 64                # 隐藏层维度
    backbone:
        type: tcn                 # 骨干网络类型
        ds: true                  # 深度可分离卷积
        num_layers: 4

optim: adam
optim_conf:
    lr: 0.001
training_config:
    max_epoch: 80
```

**示例：**

```bash
# 单 GPU 训练
bash scripts/train.sh --config conf/ds_tcn.yaml --gpus "0"

# 多 GPU 训练
bash scripts/train.sh --config conf/ds_tcn.yaml --gpus "0,1,2,3"

# 从 checkpoint 恢复训练
bash scripts/train.sh --config conf/ds_tcn.yaml --checkpoint exp/ds_tcn/30.pt

# CPU 训练
bash scripts/train.sh --config conf/ds_tcn.yaml --gpus "-1"

# CTC 模式训练（更多关键词）
bash scripts/train.sh --config conf/ds_tcn_ctc.yaml --num_keywords 2599
```

---

### 4. evaluate.sh — 模型评估

模型平均、评分、DET 曲线绘制。

```bash
bash scripts/evaluate.sh --model_dir exp/ds_tcn [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `exp/ds_tcn` | 模型目录（含 config.yaml） |
| `--test_data` | `data/test/data.list` | 测试数据 |
| `--gpu` | `0` | GPU ID |
| `--batch_size` | `256` | 推理 batch size |
| `--num_average` | `30` | 平均检查点个数 |
| `--criterion` | `max_pooling` | 损失类型（`max_pooling` / `ctc`） |
| `--window_shift` | `50` | 滑窗步长（帧） |
| `--dict` | `./dict` | 字典目录 |

**CTC 模式额外选项：**

| 选项 | 说明 |
|------|------|
| `--token_file` | tokens.txt 路径 |
| `--lexicon_file` | lexicon.txt 路径 |
| `--keywords` | 关键词列表（逗号分隔，如 "你好小问,你好问问"） |

**执行流程：**
1. **模型平均** — 取验证集最优的 top-k 检查点进行平均
2. **评分** — 对测试集推理，生成逐帧 score
3. **DET 曲线** — 计算检测错误权衡曲线（Detection Error Tradeoff）

---

### 5. export_model.sh — 模型导出

将训练好的 PyTorch 模型导出为 TorchScript JIT 和 ONNX 格式。

```bash
bash scripts/export_model.sh --model_dir exp/ds_tcn [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `exp/ds_tcn` | 模型目录 |
| `--checkpoint` | `avg_30.pt` | 待导出 checkpoint |
| `--export_jit` | `true` | 导出 JIT（TorchScript） |
| `--export_onnx` | `true` | 导出 ONNX |

**输出文件：**
- `{model_dir}/avg_30.zip` — TorchScript JIT 模型
- `{model_dir}/avg_30.onnx` — ONNX 模型（含 cache_dim, cache_len 元信息）

ONNX 导出后会自动验证 PyTorch 输出与 ONNX Runtime 输出的一致性。

---

### 6. quantize.sh — 模型量化

将 float32 模型静态量化为 int8，适用于边缘设备部署。

```bash
bash scripts/quantize.sh --model_dir exp/ds_tcn [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `exp/ds_tcn` | 模型目录 |
| `--checkpoint` | `avg_30.pt` | 待量化 checkpoint |
| `--test_data` | `data/test/data.list` | 校准数据集 |
| `--num_workers` | `8` | 并行数 |

**输出：** `{model_dir}/avg_30_int8.zip` — int8 量化后的 TorchScript 模型

---

### 7. deploy_runtime.sh — C++ 运行时编译

编译基于 ONNX Runtime 的 C++ 推理引擎。

```bash
bash scripts/deploy_runtime.sh [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--build_type` | `Release` | 构建类型（Release / Debug） |
| `--build_dir` | `build` | 构建目录（在 runtime/ 下） |
| `--onnx_version` | `auto` | ONNX Runtime 版本（auto: 从本地 whl 提取版本） |
| `--with_streaming` | `true` | 编译流式 KWS（依赖 portaudio） |
| `--toolchain` | — | 交叉编译 toolchain 路径（aarch64） |

**编译产物（在 `runtime/build/bin/` 下）：**
- `kws_main` — 离线 KWS 推理
- `stream_kws_main` — 在线流式 KWS（麦克风实时检测）

**用法：**

```bash
# 离线推理
runtime/build/bin/kws_main <fbank_dim> <batch_size> <onnx_model_path>

# 在线流式检测
runtime/build/bin/stream_kws_main <fbank_dim> <batch_size> <onnx_model_path>
```

**交叉编译（aarch64）：**

```bash
bash scripts/deploy_runtime.sh \
  --toolchain runtime/toolchains/aarch64-linux-gnu.toolchain.cmake
```

---

### 8. pipeline.sh — 🔥 一键全流程

串行执行所有阶段，覆盖数据准备到部署的完整链条。

```bash
bash scripts/pipeline.sh [选项]
```

各阶段编号：

| 阶段 | 说明 | 调用脚本 |
|------|------|----------|
| 0 | 数据准备（下载 + CMVN + data.list） | `prepare_data.sh` |
| 1 | 模型训练 | `train.sh` |
| 2 | 模型平均（Top-k） | — |
| 3 | 评分 & DET 曲线 | `evaluate.sh` |
| 4 | 模型导出（JIT + ONNX） | `export_model.sh` |
| 5 | 静态量化（int8） | `quantize.sh` |
| 6 | C++ Runtime 编译 | `deploy_runtime.sh` |

**示例：**

```bash
# 全流程（0→6）
bash scripts/pipeline.sh --dataset hey_snips

# 只做训练到评估（1→3）
bash scripts/pipeline.sh --dataset hey_snips --stage 1 --stop_stage 3

# 在 Mobvoi 数据集上使用 CTC 模型
bash scripts/pipeline.sh \
  --dataset mobvoi \
  --config conf/ds_tcn_ctc.yaml \
  --model_dir exp/ds_tcn_ctc \
  --num_keywords 2599
```

---

## 典型工作流示例

### 流程 A：Hey Snips 单唤醒词（入门）

```bash
# 1. 安装环境
bash scripts/install.sh

# 2. 激活环境
source wekws_env/bin/activate

# 3. 下载 Hey Snips 数据（需手动下载到 data/local/）
bash scripts/prepare_data.sh --dataset hey_snips

# 4. 训练模型
bash scripts/train.sh --config conf/ds_tcn.yaml

# 5. 评估
bash scripts/evaluate.sh --model_dir exp/ds_tcn

# 6. 导出
bash scripts/export_model.sh --model_dir exp/ds_tcn
```

### 流程 B：Mobvoi 多唤醒词（CTC）

```bash
bash scripts/install.sh
source wekws_env/bin/activate
bash scripts/prepare_data.sh --dataset mobvoi
bash scripts/train.sh --config conf/ds_tcn_ctc.yaml --num_keywords 2599
bash scripts/evaluate.sh --model_dir exp/ds_tcn_ctc --criterion ctc \
  --keywords "你好小问,你好问问" \
  --token_file data/tokens.txt \
  --lexicon_file data/lexicon.txt
```

### 流程 C：C++ 部署

```bash
# 先完成训练和导出
bash scripts/export_model.sh --model_dir exp/ds_tcn
# 编译 C++ 运行时
bash scripts/deploy_runtime.sh
# 运行离线推理
runtime/build/bin/kws_main 40 256 exp/ds_tcn/avg_30.onnx
```

---

## 项目目录结构说明

```
wekws/
├── scripts/               ← 一键脚本（本目录）
├── tools/                 ← 工具脚本（parse_options.sh 等）
├── wekws/                 ← Python 训练代码
│   ├── bin/               ←   命令行入口（train.py, score.py 等）
│   ├── dataset/           ←   数据加载
│   └── model/             ←   模型定义
├── runtime/               ← C++ 推理引擎
│   ├── core/              ←   核心代码
│   └── onnxruntime/       ←   ONNX Runtime 集成
├── examples/              ← 示例（各数据集 run.sh 参考）
├── conf/                  ← 模型配置文件（YAML）
├── dict/                  ← 字典（由 prepare_data.sh 生成）
├── data/                  ← 数据（由 prepare_data.sh 生成）
├── exp/                   ← 训练输出（模型、日志）
└── jetson_local_wheels/   ← Jetson 本地 whl 包（可选）
```
