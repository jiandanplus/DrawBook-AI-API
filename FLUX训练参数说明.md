# FLUX训练参数说明

本文档详细说明了使用`flux_train_network.py`进行LoRA训练时可用的所有参数及其作用。

## 基础模型参数

| 参数 | 作用 |
|------|------|
| `--pretrained_model_name_or_path` | FLUX.1模型文件路径（如 `flux1-dev.safetensors`） |
| `--clip_l` | CLIP-L模型文件路径（如 `clip_l.safetensors`） |
| `--t5xxl` | T5-XXL模型文件路径（如 `t5xxl_fp16.safetensors`） |
| `--ae` | AutoEncoder模型文件路径（如 `ae.safetensors`） |
| `--model_type` | 模型类型，可选 "flux" 或 "chroma"，默认为 "flux" |

## 数据集参数

| 参数 | 作用 |
|------|------|
| `--dataset_config` | 数据集配置文件路径（如 `lora.toml`） |
| `--train_data_dir` | 训练图像数据目录 |
| `--resolution` | 训练分辨率（如 "1024,1024"） |
| `--enable_bucket` | 启用分桶训练 |
| `--min_bucket_reso` | 最小分桶分辨率 |
| `--max_bucket_reso` | 最大分桶分辨率 |

## 网络参数

| 参数 | 作用 |
|------|------|
| `--network_module` | 网络模块（如 `networks.lora_flux`） |
| `--network_dim` | 网络维度（LoRA rank） |
| `--network_alpha` | LoRA权重缩放因子 |
| `--network_args` | 网络额外参数 |
| `--network_train_unet_only` | 仅训练UNet部分 |
| `--network_train_text_encoder_only` | 仅训练文本编码器部分 |

## 训练参数

| 参数 | 作用 |
|------|------|
| `--output_dir` | 输出目录 |
| `--output_name` | 输出文件名 |
| `--save_model_as` | 保存模型格式（如 safetensors） |
| `--max_train_epochs` | 最大训练轮数 |
| `--max_train_steps` | 最大训练步数 |
| `--train_batch_size` | 训练批次大小 |
| `--gradient_accumulation_steps` | 梯度累积步数 |
| `--save_every_n_epochs` | 每N轮保存一次模型 |
| `--save_every_n_steps` | 每N步保存一次模型 |

## 优化器参数

| 参数 | 作用 |
|------|------|
| `--learning_rate` | 学习率 |
| `--unet_lr` | UNet学习率 |
| `--text_encoder_lr` | 文本编码器学习率 |
| `--optimizer_type` | 优化器类型（如 AdamW8bit） |
| `--lr_scheduler` | 学习率调度器（如 cosine_with_restarts） |
| `--lr_warmup_steps` | 学习率预热步数 |

## 混合精度和加速参数

| 参数 | 作用 |
|------|------|
| `--mixed_precision` | 混合精度类型（如 fp16, bf16） |
| `--full_fp16` | 全FP16训练 |
| `--full_bf16` | 全BF16训练 |
| `--gradient_checkpointing` | 启用梯度检查点 |
| `--xformers` | 使用xformers优化 |
| `--sdpa` | 使用SDPA优化 |

## FLUX特有参数

| 参数 | 作用 |
|------|------|
| `--guidance_scale` | 指导比例，默认3.5 |
| `--timestep_sampling` | 时间步采样方法（如 flux_shift） |
| `--model_prediction_type` | 模型预测类型（如 raw） |
| `--discrete_flow_shift` | 离散流位移，默认3.0 |
| `--t5xxl_max_token_length` | T5-XXL最大token长度 |
| `--apply_t5_attn_mask` | 应用T5注意力掩码 |

## 内存优化参数

| 参数 | 作用 |
|------|------|
| `--cache_latents` | 缓存潜在空间 |
| `--cache_text_encoder_outputs` | 缓存文本编码器输出 |
| `--blocks_to_swap` | 交换块数量以节省显存 |
| `--fp8_base` | 使用FP8基础模型 |

## 样本生成参数

| 参数 | 作用 |
|------|------|
| `--sample_prompts` | 样本生成提示文件 |
| `--sample_sampler` | 样本采样器（如 euler_a） |
| `--sample_every_n_epochs` | 每N轮生成一次样本 |
| `--sample_every_n_steps` | 每N步生成一次样本 |

## 其他参数

| 参数 | 作用 |
|------|------|
| `--seed` | 随机种子 |
| `--logging_dir` | 日志目录 |
| `--log_with` | 日志工具（如 tensorboard, wandb） |
| `--lowram` | 低内存优化 |
| `--highvram` | 高显存优化 |

## 示例训练命令

```bash
accelerate launch --num_cpu_threads_per_process 1 FLUX_trainer/flux_train_network.py \
  --pretrained_model_name_or_path=models/flux1-dev.safetensors \
  --clip_l=models/clip_l.safetensors \
  --t5xxl=models/t5xxl_fp16.safetensors \
  --ae=models/ae.safetensors \
  --dataset_config=lora.toml \
  --network_module=networks.lora_flux \
  --sdpa \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --guidance_scale=1.0 \
  --timestep_sampling=flux_shift \
  --model_prediction_type=raw \
  --blocks_to_swap=18 \
  --cache_text_encoder_outputs \
  --cache_latents
```