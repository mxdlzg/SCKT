#!/bin/bash

# 默认参数
save_dir=""
use_pred=""
use_wandb=""
start_ratio=0.9
end_ratio=0.2
step_ratio=0.1
ratios=""

# 解析命名参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --save_dir)
      save_dir="$2"
      shift 2
      ;;
    --use_pred)
      use_pred="$2"
      shift 2
      ;;
    --use_wandb)
      use_wandb="$2"
      shift 2
      ;;
    --start_ratio)
      start_ratio="$2"
      shift 2
      ;;
    --end_ratio)
      end_ratio="$2"
      shift 2
      ;;
    --step_ratio)
      step_ratio="$2"
      shift 2
      ;;
    --ratios)
      ratios="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# 检查必要参数是否提供
if [[ -z $save_dir || -z $use_pred || -z $use_wandb ]]; then
  echo "Error: Missing required parameters."
  echo "Usage: $0 --save_dir <path> --use_pred <value> --use_wandb <value> [--start_ratio <value>] [--end_ratio <value>] [--step_ratio <value>] [--ratios <comma-separated list>]"
  exit 1
fi

# 获取当前时间，格式化为 yyyyMMdd
timestamp=$(date +"%Y%m%d-%H%M%S")

# 构建总日志文件名
log_file="${save_dir}/pred_${use_pred}_all_ratios_eval_result_time_${timestamp}.log"

# 写入日志文件头部
echo "Starting evaluation. Logs will be saved to ${log_file}" > "${log_file}"

# 如果提供了 --ratios 参数，则按列表循环
if [[ -n $ratios ]]; then
  IFS=',' read -r -a ratio_array <<< "$ratios"  # 将逗号分隔的字符串转成数组
  for train_ratio in "${ratio_array[@]}"; do
    # 记录开始时间
    start_time=$(date +%s)  

    # 执行 python 脚本，并将输出追加到日志文件
    # echo "Running with train_ratio: ${train_ratio}"
    echo "Running with train_ratio: ${train_ratio}" | tee -a "${log_file}"
    python wandb_eval.py \
        --save_dir "${save_dir}" \
        --use_pred ${use_pred} \
        --train_ratio ${train_ratio} \
        --use_wandb ${use_wandb} 2>&1 | tee -a "${log_file}"

    end_time=$(date +%s)  # 记录结束时间
    elapsed=$((end_time - start_time))
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    echo "Time taken: ${minutes} minutes and ${seconds} seconds" | tee -a "${log_file}"
    echo "-----------------------------------------------------" | tee -a "${log_file}"
  done
else
  # 否则按 start_ratio、end_ratio 和 step_ratio 循环
  train_ratio=$start_ratio
  while (( $(echo "$train_ratio >= $end_ratio" | bc -l) )); do
    # 记录开始时间
    start_time=$(date +%s)  

    # 执行 python 脚本，并将输出追加到日志文件
    echo "Running with train_ratio: ${train_ratio}" | tee -a "${log_file}"
    python wandb_eval.py \
        --save_dir "${save_dir}" \
        --use_pred ${use_pred} \
        --train_ratio ${train_ratio} \
        --use_wandb ${use_wandb} 2>&1 | tee -a "${log_file}"

    end_time=$(date +%s)  # 记录结束时间
    elapsed=$((end_time - start_time))
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    echo "Time taken: ${minutes} minutes and ${seconds} seconds" | tee -a "${log_file}"
    echo "-----------------------------------------------------" | tee -a "${log_file}"

    # 增加 train_ratio
    train_ratio=$(echo "$train_ratio - $step_ratio" | bc)
  done
fi

echo "All runs finished. Logs saved in ${log_file}" | tee -a "${log_file}"
