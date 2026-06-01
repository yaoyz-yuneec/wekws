# data/local 目录说明

本目录用于存放**原始数据集**的下载和提取文件，是整个 WeKws 数据处理流程的源头。数据处理脚本会从本目录中读取原始音频，然后生成 `data/` 目录下的 Kaldi 格式文件。

---

## 目录结构

```
data/local/
├── speech_commands_v0.01.tar.gz       # 原始数据集压缩包
└── speech_commands_v1/                # 解压后的数据集目录
    ├── .extracted                     # 解压标记文件（空文件）
    ├── train/                         # 训练集音频文件
    │   ├── bed/                       #   → bed 类别的 .wav 文件
    │   ├── bird/                      #   → bird 类别的 .wav 文件
    │   ├── cat/                       #   → cat 类别的 .wav 文件
    │   ├── dog/                       #   → ... 共 31 个关键词子目录
    │   └── ...
    ├── valid/                         # 验证集音频文件
    │   └── ...                        #   → 同 train 结构，共 30 个关键词子目录
    └── test/                          # 测试集音频文件
        └── ...                        #   → 同 train 结构，共 30 个关键词子目录
```

---

## 各文件/目录说明

### 1. `speech_commands_v0.01.tar.gz`

- **来源**：Google Speech Commands v0.01 数据集
- **下载链接**：`http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz`
- **作用**：原始音频压缩包，包含约 6.5 万条 1 秒长度的语音命令片段，涵盖 30 个关键词（如 "yes"、"no"、"up"、"down"、"on"、"off"、"go"、"stop"、"left"、"right" 等）。
- **大小**：约 1.4 GB
- **下载脚本**：由 `examples/speechcommand_v1/s0/local/data_download.sh` 自动下载到本目录。

### 2. `speech_commands_v1/`

解压后的数据集根目录，包含音频文件及数据集划分结果。

#### 2.1 `.extracted`

- **类型**：空标记文件（0 字节）
- **作用**：表示数据集已经成功解压。`data_download.sh` 脚本通过检查该文件是否存在，来决定是否跳过重复解压。

#### 2.2 `train/` / `valid/` / `test/`

- **划分来源**：数据集自带的 `validation_list.txt` 和 `testing_list.txt` 定义了验证集和测试集的音频文件列表。
- **划分方式**：由 `examples/speechcommand_v1/s0/local/split_dataset.py` 执行：
  1. 先将 `tar.gz` 解压到 `speech_commands_v1/audio/` 目录
  2. 根据 `validation_list.txt` 和 `testing_list.txt`，将对应文件移动到 `valid/` 和 `test/` 目录
  3. 剩余的 `audio/` 目录重命名为 `train/`
- **内部结构**：每个子目录中按关键词名称（如 `bed/`、`bird/`、`cat/`、`dog/` 等）进一步分类，每个关键词子目录中存放对应类别的 `.wav` 音频文件。

---

## 数据流向

```
data/local/speech_commands_v1/{train,valid,test}/
    │   (原始音频文件)
    │
    ▼
data/{train,valid,test}/wav.list
    │   (音频文件路径列表，由 `find` 命令生成)
    │
    ▼
data/{train,valid,test}/wav.scp   +   data/{train,valid,test}/text
    │   (Kaldi 格式：utterance_id + 音频绝对路径)    (标注：utterance_id + 词符)
    │
    ▼
模型训练 / 评估
```

数据处理涉及的关键脚本：

| 脚本 | 说明 |
|------|------|
| `data_download.sh` | 下载并解压 `speech_commands_v0.01.tar.gz` |
| `split_dataset.py` | 按官方划分文件将数据集拆分为 train/valid/test |
| `prepare_speech_command.py` | 从 wav.list 生成 Kaldi 格式的 wav.scp 和 text 文件 |
| `scripts/prepare_data.sh` | 一键数据准备脚本，整合上述步骤 |

---

## 与唤醒词训练的关系

`data/local` 是 WeKws **唤醒词（Keyword Spotting / Wake Word Detection）** 模型训练的数据源头。整个数据处理到模型训练的流程如下：

### 完整数据流向

```
data/local/speech_commands_v1/{train,valid,test}/
  ├── on/  (关键词 "on" 的音频)
  ├── off/ (关键词 "off" 的音频)
  ├── yes/ (关键词 "yes" 的音频)
  ├── no/  (关键词 "no" 的音频)
  ├── ...  (共 30 个关键词类别)
  │
  │  [Step 1] prepare_speech_command.py 为每个音频分配标签
  │   - 10 个目标任务词 → 对应词符 <YES>, <NO>, <UP>, <DOWN>, ...
  │   - 其余 20 个非目标词 → <UNKNOWN> (filler/填充类)
  ▼

data/{train,valid,test}/
  ├── wav.list   # 音频路径列表
  ├── wav.scp    # Kaldi 格式: utterance_id  /path/to/wav
  └── text       # 标注文件: utterance_id  <TOKEN>
  │                # 例如: on_xxx  <ON>
  │                #        yes_yyy <YES>
  │                #        bed_zzz <UNKNOWN>
  │
  │  [Step 2] compute_cmvn_stats.py 计算全局 CMVN 统计量
  │  [Step 3] wav_to_duration.sh + make_list.py
  ▼

data/{train,valid,test}/data.list
  # JSON 行格式: {"key":"...", "txt":"<TOKEN>", "duration":1.0, "wav":"..."}
  │
  │  [Step 4] wekws/bin/train.py 加载 data.list
  ▼

模型训练 (wekws/bin/train.py)
  ├── 1. 读取 wav 文件 → waveform
  ├── 2. 提取 Fbank/MFCC 特征
  ├── 3. 根据 dict/dict.txt 将文本标签转为数字 ID
  │    dict.txt:
  │      <UNKNOWN> 0   ← 非目标任务词
  │      <YES>     1   ← 目标任务词 1
  │      <NO>      2   ← 目标任务词 2
  │      ...
  │      <GO>      10  ← 目标任务词 10
  ├── 4. 定义模型结构（KWSModel）
  │    - preprocessing 层: 特征维度投影
  │    - backbone 层: TCN/MDTC/GRU/FSMN 等时序建模网络
  │    - classifier 层: 输出每个关键词的概率
  │    - activation 层: Sigmoid (唤醒词) / Identity (语音命令)
  ├── 5. 计算损失 (支持多种损失函数)
  │    - ce: 交叉熵 (用于语音命令分类)
  │    - max_pooling: 最大池化损失 (用于唤醒词，关键字取最高帧响应，非关键字取最难帧)
  │    - ctc: CTC 损失 (支持流式关键词检测)
  └── 6. 反向传播，优化模型参数

模型导出 (export_jit.py / export_onnx.py)
  ├── JIT 模型 (.zip)：用于 Python 推理
  └── ONNX 模型 (.onnx)：用于 C++/Android/嵌入式部署

运行时推理 (runtime/)
  ├── keyword_spotting.cc: C++ 推理引擎
  ├── stream_kws_ctc.py: Python 流式关键词检测
  └── Android App: 移动端唤醒词检测
```

### 核心概念说明

#### 关键词 vs 非关键词（Filler）

唤醒词模型的核心是区分**目标任务词（Keywords）**和**非目标任务词（Filler/UNKNOWN）**。

- **目标任务词**：用户想要唤醒的关键词，如 "yes"、"no"、"hi_xiaowen" 等
- **非目标词（Filler）**：所有其他语音内容，模型应将其归类为 `<UNKNOWN>`（ID=0）
- 训练时，模型学习让目标任务词对应的输出概率趋近 1，非目标词趋近 0

#### 损失函数的选择

| 损失函数 | 适用场景 | 特点 |
|---------|---------|------|
| `ce` (交叉熵) | Speech Commands 多分类 | 对整个句子做单标签分类，适合短句命令识别 |
| `max_pooling` | 唤醒词检测 | 帧级别损失：关键字取最大帧概率，非关键字取最小帧概率；适合长音频中的关键词检测 |
| `ctc` | 流式唤醒词+转录 | 支持 CTC prefix beam search 解码，可同时检测关键词和转录语音内容 |

#### 模型结构简介

模型采用 **特征提取 → 时序建模 → 分类输出** 的三段式架构：

```
Fbank/MFCC 特征
      │
      ▼
┌────────────────┐
│ GlobalCMVN     │  ← 全局均值方差归一化 (可选)
├────────────────┤
│ Preprocessing  │  ← 特征维度投影 + 下采样
├────────────────┤
│ Backbone       │  ← 核心时序网络：TCN / MDTC / GRU / FSMN
├────────────────┤
│ Classifier     │  ← 全连接层 → 输出每个关键词的概率
└────────────────┘
      │
      ▼
关键词概率分布 (如: UNKNOWN=0.1, YES=0.8, NO=0.05, ...)
```

#### 训练与推理的区别

- **训练阶段**：数据来自 `data/list`，使用整个句子提取特征，计算损失并更新模型
- **推理阶段**：`stream_kws_ctc.py` 或 C++ runtime 以**流式方式**处理音频（每 300ms 滑动窗口），实时输出检测结果
- **部署场景**：训练好的模型可以导出为 ONNX/JIT 格式，部署到 Android、树莓派、Jetson 等边缘设备

---

## 其他数据集

本目录的设计同样适用于其他数据集（如 `hey_snips`、`mobvoi/hi_xiaowen`）。不同数据集的数据准备流程在 `scripts/prepare_data.sh` 中统一管理，通过 `--dataset` 参数选择：

- `hey_snips`：下载到 `dl_dir/hey_snips_research_6k_en_train_eval_clean_ter/` 目录
- `mobvoi`：下载到 `dl_dir/mobvoi_hotword_dataset/` 和 `dl_dir/mobvoi_hotword_dataset_resources/` 目录
- `speech_commands`：下载到 `dl_dir/speech_commands_v1/` 目录（即本目录）
