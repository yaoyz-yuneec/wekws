# ONNX 模型导出说明

## 一、`export_model.sh` 会导出哪些内容？

### 直接导出的文件

`export_model.sh` 只导出**模型文件本身**，不包含其他辅助文件。具体输出如下：

| 文件 | 格式 | 触发条件 | 说明 |
|------|------|----------|------|
| `{model_dir}/avg_30.onnx` | ONNX | `--export_onnx true` | 用于 C++ 运行时（ONNX Runtime）部署的流式推理模型 |
| `{model_dir}/avg_30.zip` | TorchScript JIT | `--export_jit true` | 用于 Python 端流式推理的 JIT 模型 |

文件名取决于 checkpoint 名称，例如：
- `avg_30.pt` → `avg_30.onnx` + `avg_30.zip`
- `final.pt` → `final.onnx` + `final.zip`

### 不会导出的文件（但部署时需要）

| 文件 | 用途 | 来源 |
|------|------|------|
| `{model_dir}/config.yaml` | 模型配置（input_dim、backbone 类型等） | 训练脚本自动保存 |
| `dict/dict.txt` | 关键词 → 索引映射表，评估时使用 | `prepare_data.sh` 生成 |
| `dict/words.txt` | 关键词列表，循环遍历每个关键词生成 DET | `prepare_data.sh` 生成 |
| `data/train/global_cmvn` | CMVN 归一化参数（已固化到模型中，一般无需额外处理） | `prepare_data.sh` 生成 |

> **注意**：CTC loss 模式下，评估脚本（`score_ctc.py`、`compute_det_ctc.py`）还需要 `tokens.txt` 和 `lexicon.txt`。但这些是数据准备阶段的产物，**不在导出脚本的职责范围内**，导出后的 ONNX 模型本身不依赖它们。

### 关于 tokens.txt 的说明

**`tokens.txt` 不是导出脚本生成的**，它属于 CTC 训练/评估流程的产物：

- **CTC 模式**（`run_ctc.sh` / `run_fsmn_ctc.sh`）：在数据准备阶段从外部资源（如 `mobvoi_kws_transcription/tokens.txt`）复制或生成，路径通常在 `data/tokens.txt`，内容形如：
  ```
  <SILENCE> 0
  <EPS> 1
  <BLK> 2
  a 3
  ai 4
  ...
  ```
- **Max-pooling 模式**（默认，如 `conf/ds_tcn.yaml`）：不需要 tokens.txt，只依赖 `dict/dict.txt` 和 `dict/words.txt`

**关键理解**：导出后的 ONNX 模型本身**已经包含了完整的计算图**（含 softmax）。推理时：
1. 将音频转为 fbank 特征 → 送入 ONNX 模型 → 得到概率输出
2. 如果使用 CTC 模式，需要对概率输出做 CTC beam search 解码 → **这时**才需要 `tokens.txt` / `lexicon.txt`
3. 如果使用 max-pooling 模式，直接取模型输出即可，不需要任何辅助文件

因此 `tokens.txt` / `lexicon.txt` 是**后处理解码阶段**需要的文件，而非 ONNX 模型推理阶段需要的文件。

### 完整导出后的目录结构示例

```
exp/ds_tcn/
├── config.yaml              # [已有] 模型配置
├── avg_30.pt                # [已有] 平均后的 checkpoint
├── avg_30.onnx              # [新增] ONNX 模型 ← export_model.sh
├── avg_30.zip               # [新增] JIT 模型   ← export_model.sh
├── checkpoint.pt            # [已有] 训练过程保存的 checkpoint
├── train.log                # [已有] 训练日志
├── tensorboard/             # [已有] 训练曲线
├── test_avg_30/             # [已有] 评估结果目录
│   ├── score.txt
│   ├── stats.*.txt
│   └── det.png
└── ...
```

---

## 二、导出 ONNX 格式的关键注意事项

WeKws 模型的 ONNX 导出与一般的图像分类/语音识别模型不同，因为它采用了**带状态缓存（state cache）的流式推理架构**。以下是导出时必须注意的关键点：

### 1. 输入输出约定

模型接受 **2 个输入**，返回 **2 个输出**：

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `input` | float32 | `(1, T, feature_dim)` | 音频特征，T 为时间维度（动态） |
| `cache` | float32 | `(1, hdim, padding)` 或 `(1, hdim, padding, num_layers)` | 流式推理的状态缓存 |
| `output` | float32 | `(1, T, output_dim)` | 模型输出，T 为时间维度（动态） |
| `r_cache` | float32 | 与 `cache` 同形 | 更新后的缓存，需传入下一次推理 |

> C++ 运行时代码（`runtime/core/kws/keyword_spotting.cc`）硬编码了这 4 个名称，**不可更改**。

### 2. Cache 机制

- 非 FSMN 模型：cache 形状为 `(1, hdim, padding)`
- FSMN 模型：cache 需要额外扩展最后一维表示层数，形状为 `(1, hdim, padding, num_layers)`
- cache 的 `padding` 值来自 `model.backbone.padding`

### 3. 元数据（Metadata）

ONNX 模型中**必须**嵌入以下自定义元数据，C++ 运行时通过它们初始化缓存：

```python
meta = onnx_model.metadata_props.add()
meta.key, meta.value = 'cache_dim', str(model.hdim)

meta = onnx_model.metadata_props.add()
meta.key, meta.value = 'cache_len', str(model.backbone.padding)
```

运行时在 `keyword_spotting.cc` 中通过 `metadata.LookupCustomMetadataMap("cache_dim")` 和 `metadata.LookupCustomMetadataMap("cache_len")` 读取。

### 4. 动态轴配置

时间维度（axis=1）必须设为动态，以支持不同长度的输入：

```python
dynamic_axes = {'input': {1: 'T'}, 'output': {1: 'T'}}
```

### 5. CTC Loss 的特殊处理

如果训练配置中使用了 CTC loss（`criterion == 'ctc'`），导出前必须将模型的 forward 替换为 `forward_softmax`，将 softmax 纳入导出的计算图中：

```python
if configs['training_config'].get('criterion', 'max_pooling') == 'ctc':
    model.forward = model.forward_softmax
```

### 6. Opset 版本

项目固定使用 `opset_version=13`。

### 7. 精度验证

导出后建议验证 PyTorch 和 ONNX Runtime 的输出一致性，脚本中使用的精度阈值为 `atol=1e-5`。

---

## 三、能否使用其他项目的脚本导出？

**强烈建议使用本项目自带的导出工具**，原因如下：

### 不建议使用其他项目脚本的原因

1. **输入输出约定不通用**
   - 大多数语音项目（如 WeNet、wenet）导出的 ONNX 是编码器-解码器或标准 CTC 结构
   - WeKws 的 `(input, cache) → (output, r_cache)` 接口是该项目独有的

2. **Cache 机制与 backbone 类型强耦合**
   - 项目支持 TCN、GRU、FSMN、MDTC 多种 backbone
   - 每种 backbone 对 cache 的形状和语义要求不同，其他项目无法适配

3. **元数据是 C++ 运行时的硬依赖**
   - 缺少 `cache_dim` / `cache_len` 元数据会导致 C++ 运行时直接崩溃

### 什么情况下可以用自己的脚本

如果**理解上述所有约束**，可以自行编写导出脚本，但必须保证：

```
输入: ["input"]  (dynamic shape: [1, T, feature_dim])
      ["cache"]  (shape: [1, hdim, padding])  # FSMN 时为 [1, hdim, padding, num_layers]
输出: ["output"] (dynamic shape: [1, T, output_dim])
      ["r_cache"](shape: [1, hdim, padding])
元数据: cache_dim / cache_len 必须写入 ONNX Metadata
CTC: softmax 必须包含在计算图中，不能依赖后处理
```

---

## 四、推荐做法

```bash
# 最稳妥的方式（前提是 checkpoint 位于 exp/ds_tcn/ 下）
bash scripts/export_model.sh --model_dir exp/ds_tcn

# 仅导出 ONNX
bash scripts/export_model.sh --model_dir exp/ds_tcn --export_jit false

# 指定 checkpoint
bash scripts/export_model.sh --model_dir exp/ds_tcn --checkpoint exp/ds_tcn/final.pt
```

即使 checkpoint 是从其他项目下载的，**只要模型架构是 WeKws 的 `KWSModel`**，就可以使用本项目的导出脚本。如果模型架构不同，则无法直接转换。
