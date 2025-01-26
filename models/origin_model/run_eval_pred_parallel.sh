#!/bin/bash

# 默认参数
save_dir=""
use_pred=""
use_wandb=""
start_ratio=0.2
end_ratio=0.9
step_ratio=0.1
ratios=""
max_processes=4 # 默认最大进程数

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
     --max_processes)
      max_processes="$2"
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
  echo "Usage: $0 --save_dir <path> --use_pred <value> --use_wandb <value> [--start_ratio <value>] [--end_ratio <value>] [--step_ratio <value>] [--ratios <comma-separated list>] [--max_processes <integer>]"
  exit 1
fi

# 检查MPS
if ! pgrep -f "nvidia-cuda-mps-control -d" > /dev/null; then
  echo "错误： NVIDIA MPS (Multi-Process Service) 未启动。"
  echo "请先启动 MPS: sudo nvidia-cuda-mps-control -d"
  exit 1 # 返回非 0 表示错误
else
  echo "NVIDIA MPS 正在运行。"
fi

# 获取当前时间，格式化为 yyyyMMdd
timestamp=$(date +"%Y%m%d-%H%M%S")

# 构建总日志文件名
log_file="${save_dir}/pred_${use_pred}_all_ratios_eval_result_time_${timestamp}.log"

# 写入日志文件头部
echo "Starting evaluation. Logs will be saved to ${log_file}" > "${log_file}"

# 声明一个用于存储后台进程ID的数组
declare -a pids

# 定义一个函数来启动后台任务并记录 PID
run_eval() {
  local train_ratio="$1"
  
  # 记录开始时间
  local start_time=$(date +%s)
  
  # 执行 python 脚本，并将输出追加到日志文件
  echo "Running with train_ratio: ${train_ratio}" | tee -a "${log_file}"
  python wandb_eval.py \
      --save_dir "${save_dir}" \
      --use_pred "${use_pred}" \
      --train_ratio "${train_ratio}" \
      --use_wandb "${use_wandb}" 2>&1 | tee -a "${log_file}" &
  
  local pid=$!
  pids+=("$pid")  # 将 PID 添加到数组中
  echo "启动的任务pid：$pid"
  
  end_time=$(date +%s) # 记录结束时间
  elapsed=$((end_time - start_time))
  minutes=$((elapsed / 60))
  seconds=$((elapsed % 60))
  
  # echo "Time taken: ${minutes} minutes and ${seconds} seconds" | tee -a "${log_file}"
  echo "-----------------------------------------------------" | tee -a "${log_file}"
}


# 定义一个函数来等待进程完成
wait_for_processes() {
  local has_next="$1"  # 将第一个参数赋值给局部变量 has_next
  while true; do
    if [[ ${#pids[@]} -eq 0 ]]; then
      break  # 如果没有后台进程，则退出循环
    fi

    # 等待任意一个后台进程结束，不关心是哪个进程
    wait -n
    
    echo "当前并行任务达到最大数量$max_processes，pids:$pids, 进入wait"
    # 扫描 pids 数组，找出已结束的进程并移除
    for i in "${!pids[@]}"; do
        if ! kill -0 "${pids[$i]}" 2>/dev/null; then #判断进程是否还存在，如果不存在了，则移除pid
            echo "任务结束，pid：$pids[$i]"
        unset pids[$i]
        pids=("${pids[@]}")
        echo "检测到任务结束，剩余pids：$pids。"
            if [[ "$has_next" != "false" ]]; then
              echo "存在后续任务，退出wait~"
              return
            else
              echo "不存在后续任务，继续wait"
              break
            fi
        fi
    done
  done
}


# 如果提供了 --ratios 参数，则按列表循环
if [[ -n $ratios ]]; then
  IFS=',' read -r -a ratio_array <<< "$ratios"  # 将逗号分隔的字符串转成数组
  for train_ratio in "${ratio_array[@]}"; do
     run_eval "$train_ratio"

    # 控制并发数量
    if [[ ${#pids[@]} -ge "$max_processes" ]]; then
       wait_for_processes
    fi

  done
else
  # 否则按 start_ratio、end_ratio 和 step_ratio 循环
  train_ratio=$start_ratio
  while (( $(echo "$train_ratio <= $end_ratio" | bc -l) )); do

    run_eval "$train_ratio"
    # 控制并发数量
    if [[ ${#pids[@]} -ge "$max_processes" ]]; then
      wait_for_processes
    fi

    # 增加 train_ratio
    train_ratio=$(echo "$train_ratio + $step_ratio" | bc)
  done
fi

# 等待所有剩余的后台进程完成
wait_for_processes "false"


echo "All runs finished. Logs saved in ${log_file}" | tee -a "${log_file}"

