#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    # 构建训练命令
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "FLUX_trainer/flux_train_network.py",
        "--pretrained_model_name_or_path=models\\flux1-dev.safetensors",
        "--clip_l=models\\clip_l.safetensors",
        "--t5xxl=models\\t5xxl_fp16.safetensors",
        "--ae=models\\ae.safetensors",
        "--dataset_config=my_flux_dataset_config.toml",
        "--output_dir=output_model",
        "--output_name=test_flux_lora",
        "--save_model_as=safetensors",
        "--network_module=networks.lora_flux",
        "--network_dim=16",
        "--network_alpha=1",
        "--learning_rate=1e-4",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=constant",
        "--sdpa",
        "--max_train_epochs=10",
        "--save_every_n_epochs=1",
        "--mixed_precision=fp16",
        "--gradient_checkpointing",
        "--guidance_scale=1.0",
        "--timestep_sampling=flux_shift",
        "--model_prediction_type=raw",
        "--blocks_to_swap=18",
        "--cache_text_encoder_outputs",
        "--cache_latents"
    ]
    
    print("启动 FLUX LoRA 训练...")
    print("执行命令:", " ".join(cmd))
    
    try:
        # 执行命令
        result = subprocess.run(cmd, check=True)
        print("训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"训练出错: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)

if __name__ == "__main__":
    main()