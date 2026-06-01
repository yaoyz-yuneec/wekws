#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#                 WeKws Contributors
#
# 一键数据准备脚本 —— 下载、提取、预处理数据集
# 支持: hey_snips | mobvoi(hi_xiaowen) | speech_commands
# 用法: bash scripts/prepare_data.sh --dataset hey_snips [options]

set -euo pipefail

# ======================== 默认参数 ========================
dataset=hey_snips          # 数据集名称: hey_snips / mobvoi / speech_commands
dl_dir=./data/local        # 下载目录
output_dir=./data          # 输出目录 (Kaldi 格式数据)
config=./conf/ds_tcn.yaml  # 模型配置文件 (CMVN 需用)
num_workers=16             # CMVN 计算并行数
nj=8                       # wav_to_duration 并行数

help_message="Usage: $0 --dataset [hey_snips|mobvoi|speech_commands] [options]"
. "$(dirname "$0")/../tools/parse_options.sh" || exit 1

# ======================== 路径设置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PATH="${PROJECT_DIR}/tools:${PATH}"

mkdir -p "${dl_dir}" "${output_dir}"

echo "=========================================="
echo "  数据集: ${dataset}"
echo "  下载目录: ${dl_dir}"
echo "  输出目录: ${output_dir}"
echo "=========================================="

# ======================== Stage -1: 下载数据 ========================
case "${dataset}" in
  hey_snips)
    echo "[Stage -1] Hey Snips 数据集需要手动填写 Google Form 下载"
    echo "            表单地址: https://forms.gle/JtmFYM7xK1SaMfZYA"
    echo "            下载后将 hey_snips_kws_4.0.tar.gz 放入 ${dl_dir}"
    read -r -p "是否已下载完毕并放入指定目录? [y/N]: " confirm
    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
      echo "[INFO] 请下载后再运行。跳过数据提取。"
      exit 0
    fi
    if [ -f "${dl_dir}/hey_snips_kws_4.0.tar.gz" ]; then
      echo "[INFO] 解压数据 ..."
      tar -xvzf "${dl_dir}/hey_snips_kws_4.0.tar.gz" -C "${dl_dir}" || exit 1
    else
      echo "[ERROR] ${dl_dir}/hey_snips_kws_4.0.tar.gz 不存在"
      exit 1
    fi
    ;;

  mobvoi)
    echo "[Stage -1] 下载 Mobvoi Hotword 数据集 ..."
    data_url=http://www.openslr.org/resources/87
    for f in mobvoi_hotword_dataset.tgz mobvoi_hotword_dataset_resources.tgz; do
      if [ ! -f "${dl_dir}/${f}" ]; then
        wget --no-check-certificate -O "${dl_dir}/${f}" "${data_url}/${f}"
      fi
      echo "[INFO] 解压 ${f} ..."
      tar -xvzf "${dl_dir}/${f}" -C "${dl_dir}" || exit 1
    done
    ;;

  speech_commands)
    echo "[Stage -1] 下载 Google Speech Commands v1 ..."
    local_dir="${SCRIPT_DIR}/../examples/speechcommand_v1/s0/local"
    if [ -f "${local_dir}/data_download.sh" ]; then
      bash "${local_dir}/data_download.sh" --dl_dir "${dl_dir}"
      python "${local_dir}/split_dataset.py" "${dl_dir}/speech_commands_v1"
    else
      echo "[ERROR] 找不到 ${local_dir}/data_download.sh"
      exit 1
    fi
    ;;

  *)
    echo "[ERROR] 不支持的数据集: ${dataset}，可选: hey_snips / mobvoi / speech_commands"
    exit 1
    ;;
esac

# ======================== Stage 0: 准备 Kaldi 格式数据 ========================
echo ""
echo "[Stage 0] 准备 Kaldi 格式数据 ..."

mkdir -p dict

case "${dataset}" in
  hey_snips)
    echo "<FILLER> -1" > dict/dict.txt
    echo "<HEY_SNIPS> 0" >> dict/dict.txt
    awk '{print $1}' dict/dict.txt > dict/words.txt

    for folder in train dev test; do
      mkdir -p "${output_dir}/${folder}"
      json_path="${dl_dir}/hey_snips_research_6k_en_train_eval_clean_ter/${folder}.json"
      python "${PROJECT_DIR}/examples/hey_snips/s0/local/prepare_data.py" \
        "${dl_dir}/hey_snips_research_6k_en_train_eval_clean_ter/audio_files" \
        "${json_path}" dict/dict.txt "${output_dir}/${folder}"
    done
    ;;

  mobvoi)
    echo "<FILLER> -1" > dict/dict.txt
    echo "<HI_XIAOWEN> 0" >> dict/dict.txt
    echo "<NIHAO_WENWEN> 1" >> dict/dict.txt
    awk '{print $1}' dict/dict.txt > dict/words.txt

    for folder in train dev test; do
      mkdir -p "${output_dir}/${folder}"
      for prefix in p n; do
        mkdir -p "${output_dir}/${prefix}_${folder}"
        json_path="${dl_dir}/mobvoi_hotword_dataset_resources/${prefix}_${folder}.json"
        python "${PROJECT_DIR}/examples/hi_xiaowen/s0/local/prepare_data.py" \
          "${dl_dir}/mobvoi_hotword_dataset" "${json_path}" \
          dict/dict.txt "${output_dir}/${prefix}_${folder}"
      done
      cat "${output_dir}/p_${folder}/wav.scp" "${output_dir}/n_${folder}/wav.scp" \
        > "${output_dir}/${folder}/wav.scp"
      cat "${output_dir}/p_${folder}/text" "${output_dir}/n_${folder}/text" \
        > "${output_dir}/${folder}/text"
      rm -rf "${output_dir}/p_${folder}" "${output_dir}/n_${folder}"
    done
    ;;

  speech_commands)
    echo "<UNKNOWN> 0"  > dict/dict.txt
    echo "<YES> 1"    >> dict/dict.txt
    echo "<NO> 2"     >> dict/dict.txt
    echo "<UP> 3"     >> dict/dict.txt
    echo "<DOWN> 4"   >> dict/dict.txt
    echo "<LEFT> 5"   >> dict/dict.txt
    echo "<RIGHT> 6"  >> dict/dict.txt
    echo "<ON> 7"     >> dict/dict.txt
    echo "<OFF> 8"    >> dict/dict.txt
    echo "<STOP> 9"   >> dict/dict.txt
    echo "<GO> 10"    >> dict/dict.txt
    awk '{print $1}' dict/dict.txt > dict/words.txt

    for x in train test valid; do
      data_dir="${output_dir}/${x}"
      mkdir -p "${data_dir}"
      find "${dl_dir}/speech_commands_v1/${x}" -name "*.wav" \
        | grep -v "_background_noise_" > "${data_dir}/wav.list"
      python "${PROJECT_DIR}/examples/speechcommand_v1/s0/local/prepare_speech_command.py" \
        --wav_list="${data_dir}/wav.list" --data_dir="${data_dir}"
    done
    ;;
esac

echo "[Stage 0] 数据准备完成。"

# ======================== Stage 1: CMVN + 格式转换 ========================
echo ""
echo "[Stage 1] 计算 CMVN 并生成 data.list ..."

# 计算全局 CMVN
python "${PROJECT_DIR}/tools/compute_cmvn_stats.py" \
  --num_workers "${num_workers}" \
  --train_config "${config}" \
  --in_scp "${output_dir}/train/wav.scp" \
  --out_cmvn "${output_dir}/train/global_cmvn"

# 生成 wav.dur + data.list
for x in train dev test valid; do
  if [ -d "${output_dir}/${x}" ]; then
    bash "${PROJECT_DIR}/tools/wav_to_duration.sh" --nj "${nj}" \
      "${output_dir}/${x}/wav.scp" "${output_dir}/${x}/wav.dur"
    python "${PROJECT_DIR}/tools/make_list.py" \
      "${output_dir}/${x}/wav.scp" "${output_dir}/${x}/text" \
      "${output_dir}/${x}/wav.dur" "${output_dir}/${x}/data.list"
    echo "[INFO] ${x}: $(wc -l < "${output_dir}/${x}/data.list") 条样本"
  fi
done

echo ""
echo "[SUCCESS] 数据准备完成！"
echo "         数据目录: ${output_dir}/"
echo "         字典目录: $(pwd)/dict/"
echo "         使用 bash scripts/train.sh 开始训练"
