import argparse
import subprocess
import time

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluations with different train ratios.")
    parser.add_argument("--save_dir", required=True, help="Directory to save results.")
    parser.add_argument("--use_pred", required=True, help="Prediction mode (1 or 0).")
    parser.add_argument("--use_wandb", required=True, help="WandB mode (1 or 0).")
    parser.add_argument("--start_ratio", type=float, default=0.2, help="Starting train ratio.")
    parser.add_argument("--end_ratio", type=float, default=0.9, help="Ending train ratio.")
    parser.add_argument("--step_ratio", type=float, default=0.1, help="Step size for train ratio.")
    parser.add_argument("--ratios", type=str, help="Comma-separated list of train ratios.")
    return parser.parse_args()

# 类似于 Unix `tee` 的功能
class Tee:
    def __init__(self, logfile):
        self.logfile = logfile

    def write(self, message):
        print(message, end="")  # 输出到控制台
        self.logfile.write(message)  # 同时写入日志文件
        self.flush()

    def flush(self):
        self.logfile.flush()

# 主函数
def main():
    args = parse_args()

    # 获取当前时间戳，创建总日志文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = f"{args.save_dir}/pred_{args.use_pred}_all_ratios_eval_result_time_{timestamp}.log"

    # 打印日志文件路径
    print(f"Starting evaluation. Logs will be saved to {log_file_path}")
    with open(log_file_path, "w") as log:
        tee = Tee(log)  # 创建 Tee 对象

        tee.write(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"save_dir={args.save_dir}, use_pred={args.use_pred}, use_wandb={args.use_wandb}\n")

        # 确定 ratio 列表
        if args.ratios:
            ratio_list = [float(r.strip()) for r in args.ratios.split(",")]
        else:
            ratio_list = []
            ratio = args.start_ratio
            while ratio <= args.end_ratio:
                ratio_list.append(round(ratio, 2))
                ratio += args.step_ratio

        # 循环处理每个 ratio
        for train_ratio in ratio_list:
            start_time = time.time()
            tee.write(f"\nRunning with train_ratio: {train_ratio}\n")

            # 调用外部命令
            cmd = [
                "python", "wandb_eval.py",
                "--save_dir", args.save_dir,
                "--use_pred", str(args.use_pred),
                "--train_ratio", str(train_ratio),
                "--use_wandb", str(args.use_wandb)
            ]
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tee.write(process.stdout)

            # 记录运行时间
            end_time = time.time()
            duration = end_time - start_time
            minutes, seconds = divmod(duration, 60)
            tee.write(f"Finished train_ratio {train_ratio} in {int(minutes)}m {int(seconds)}s\n")
            tee.write("-----------------------------------------------------\n")

        tee.write(f"\nAll runs finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"All runs finished. Logs saved in {log_file_path}")

if __name__ == "__main__":
    main()
