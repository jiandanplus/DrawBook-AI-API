import os
import subprocess

# ========== 配置区 ==========
pretrained_model = "./models/white5v_ultra.safetensors"
clip_l = ""
t5xxl = ""
ae = "./models/sdxl_vae_fp16fix.safetensors"
model_type = "sdxl"  # 可选: sd1.5, sd2.0, sdxl, flux
parameterization = 0  # 仅sd2.0时有效

train_data_dir = "./train_database/test1"
reg_data_dir = ""
shuffle_caption = "False"  # True or False

network_weights = ""
network_dim = 100000
network_alpha = 100000

resolution = "1024,1024"
batch_size = 1
max_train_epoches = 1
save_every_n_epochs = 1

train_unet_only = 0
train_text_encoder_only = 0
stop_text_encoder_training = 0

noise_offset = 0
keep_tokens = 0
min_snr_gamma = 0

lr = "1e-4"
unet_lr = "1e-4"
text_encoder_lr = "1e-5"
lr_scheduler = "constant"
lr_warmup_steps = 0
lr_restart_cycles = 1

optimizer_type = "AdamW8bit"

output_name = "煎蛋"
save_model_as = "safetensors"

save_state = 0
resume = ""

min_bucket_reso = 512
max_bucket_reso = 2048
persistent_data_loader_workers = 1
clip_skip = 2
multi_gpu = 0
lowram = 0

algo = "lora"
conv_dim = 4
conv_alpha = 4
dropout = "0"

use_wandb = 0
wandb_api_key = ""
log_tracker_name = ""

# ========== 环境变量 ==========
os.environ["HF_HOME"] = "./lora-scripts/huggingface"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

# ========== 参数组装 ==========
ext_args = []
launch_args = []

if model_type == "sd1.5":
    ext_args.append(f"--clip_skip={clip_skip}")
elif model_type == "sd2.0":
    ext_args.append("--v2")

if model_type == "flux":
    network_module = "networks.lora_flux"
else:
    network_module = "lycoris.kohya"

if model_type == "sdxl":
    trainer_file = "./lora-scripts/scripts/stable/sdxl_train_network.py"
elif model_type == "flux":
    trainer_file = "./lora-scripts/scripts/dev/flux_train_network.py"
else:
    trainer_file = "./lora-scripts/scripts/stable/train_network.py"

if multi_gpu:
    launch_args.extend(["--multi_gpu", "--num_processes=8"])

if lowram:
    ext_args.append("--lowram")

if parameterization:
    ext_args.append("--v_parameterization")

if train_unet_only:
    ext_args.append("--network_train_unet_only")

if train_text_encoder_only:
    ext_args.append("--network_train_text_encoder_only")

if network_weights:
    ext_args.append(f"--network_weights={network_weights}")

if reg_data_dir:
    ext_args.append(f"--reg_data_dir={reg_data_dir}")

if optimizer_type:
    ext_args.append(f"--optimizer_type={optimizer_type}")

if optimizer_type == "DAdaptation":
    ext_args.extend(["--optimizer_args", "decouple=True"])

if network_module == "lycoris.kohya":
    ext_args.extend([
        "--network_args",
        f"conv_dim={conv_dim}",
        f"conv_alpha={conv_alpha}",
        f"algo={algo}",
        f"dropout={dropout}"
    ])

if noise_offset != 0:
    ext_args.append(f"--noise_offset={noise_offset}")

if stop_text_encoder_training != 0:
    ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")

if save_state == 1:
    ext_args.append("--save_state")

if resume:
    ext_args.append(f"--resume={resume}")

if min_snr_gamma != 0:
    ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

if persistent_data_loader_workers:
    ext_args.append("--persistent_data_loader_workers")

if shuffle_caption.lower() == "true":
    ext_args.append("--shuffle_caption")

if use_wandb == 1:
    ext_args.append("--log_with=all")
    if wandb_api_key:
        ext_args.append(f"--wandb_api_key={wandb_api_key}")
    if log_tracker_name:
        ext_args.append(f"--log_tracker_name={log_tracker_name}")
else:
    ext_args.append("--log_with=tensorboard")

# ========== 训练命令组装 ==========
import sys
cmd = [
    sys.executable, "-m", "accelerate.commands.launch", *launch_args, "--num_cpu_threads_per_process=4", trainer_file,
    "--enable_bucket",
    f"--pretrained_model_name_or_path={pretrained_model}",
    f"--train_data_dir={train_data_dir}",
    f"--clip_l={clip_l}",
    f"--t5xxl={t5xxl}",
    f"--ae={ae}",
    "--output_dir=./output",
    "--logging_dir=./logs",
    f"--log_prefix={output_name}",
    f"--resolution={resolution}",
    f"--network_module={network_module}",
    f"--max_train_epochs={max_train_epoches}",
    f"--learning_rate={lr}",
    f"--unet_lr={unet_lr}",
    f"--text_encoder_lr={text_encoder_lr}",
    f"--lr_scheduler={lr_scheduler}",
    f"--lr_warmup_steps={lr_warmup_steps}",
    f"--lr_scheduler_num_cycles={lr_restart_cycles}",
    f"--network_dim={network_dim}",
    f"--network_alpha={network_alpha}",
    f"--output_name={output_name}",
    f"--train_batch_size={batch_size}",
    f"--save_every_n_epochs={save_every_n_epochs}",
    '--mixed_precision=fp16',
    '--save_precision=fp16',
    '--seed=1337',
    '--cache_latents',
    '--prior_loss_weight=1',
    '--max_token_length=225',
    '--caption_extension=.txt',
    f'--save_model_as={save_model_as}',
    f'--min_bucket_reso={min_bucket_reso}',
    f'--max_bucket_reso={max_bucket_reso}',
    f'--keep_tokens={keep_tokens}',
    '--xformers'
] + ext_args

# ========== 执行训练 ==========
print("Running training command:")
print(" ".join(cmd))
import sys
print(sys.executable)
subprocess.run(cmd, check=True)
print("Train finished")
input("Press Enter to exit...")