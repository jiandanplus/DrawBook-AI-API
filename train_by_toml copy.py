import os
import subprocess
import sys
from pathlib import Path

# Configuration parameters
sample_prompts = "./lora-scripts/config/sample_prompts.txt"          # prompt file for sample | 采样 prompts 文件, 留空则不启用采样功能

flux = 0        # train sdxl LoRA | 训练 SDXL LoRA (1 -> flux, 0 -> sdxl/lork)

config_file = "./toml/lork.toml" if flux else "./toml/lork.toml"
multi_gpu = 0   # multi gpu | 多显卡训练 该参数仅限在显卡数 >= 2 使用

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================

# Set environment variables
os.environ["HF_HOME"] = "./lora-scripts/huggingface"
os.environ["PYTHONUTF8"] = "1"

ext_args = []
launch_args = []

# Multi GPU configuration
if multi_gpu:
    launch_args.extend(["--multi_gpu", "--num_processes=2"])

# Select script based on sdxl flag
script_name = "flux_train_network.py" if flux else "sdxl_train_network.py"
script_path = f"./lora-scripts/scripts/dev/{script_name}"

# Check if script exists
if not Path(script_path).exists():
    print(f"Error: Script file not found: {script_path}")
    print("Available scripts in ./lora-scripts/scripts/dev/:")
    if Path("./scripts/dev/").exists():
        for file in Path("./lora-scripts/scripts/dev/").glob("*.py"):
            print(f"  {file.name}")
    sys.exit(1)

# Build command
cmd = [
    sys.executable, "-m", "accelerate.commands.launch",
    *launch_args,
    "--num_cpu_threads_per_process=8",
    script_path,
    f"--config_file={config_file}",
    f"--sample_prompts={sample_prompts}",
    *ext_args
]

# Run training
try:
    print("Starting LoRA training...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print("Training finished successfully")
except subprocess.CalledProcessError as e:
    print(f"Training failed with return code {e.returncode}")
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(1)