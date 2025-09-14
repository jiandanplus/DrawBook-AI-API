该仓库包含用于Stable Diffusion的训练、生成和实用脚本。

## FLUX.1 和 SD3 训练（WIP）

此功能是实验性的。选项和训练脚本可能会在未来发生变化。如果您有任何改进训练的想法，请告诉我们。

__请将PyTorch更新到2.4.0。我们使用`torch==2.4.0`和`torchvision==0.19.0`以及CUDA 12.4进行了测试。我们还为了安全起见将`accelerate`更新到0.33.0。`requirements.txt`也已更新，所以请更新依赖项。__

安装PyTorch的命令如下：
`pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124`

如果使用DeepSpeed，请使用`pip install deepspeed==0.16.7`安装DeepSpeed。

- [FLUX.1 训练](#flux1-训练)
- [SD3 训练](#sd3-训练)
 
### 最新更新

2025年7月30日：
- **重大变更**：对于FLUX.1和Chroma训练，在训练期间生成样本图像时，CFG（无分类器指导，使用负面提示）比例选项已从`--g`更改为`--l`。现在`--g`选项用于嵌入的指导比例。请相应地更新您的提示。详情请参见[训练期间的样本图像生成](#训练期间的样本图像生成)。

- 在PR [#2157](https://github.com/kohya-ss/sd-scripts/pull/2157)中添加了对[Chroma](https://huggingface.co/lodestones/Chroma)的支持。感谢lodestones提供的高质量模型。
    - Chroma是基于FLUX.1 schnell的新模型。在此仓库中，使用`flux_train_network.py`通过`--model_type chroma`为Chroma训练LoRA。Chroma训练还需要`--apply_t5_attn_mask`。
    - 请参考[FLUX.1 LoRA训练文档](./docs/flux_train_network.md)了解更多详情。

2025年7月21日：
- 在PR [#1927](https://github.com/kohya-ss/sd-scripts/pull/1927)和[#2138](https://github.com/kohya-ss/sd-scripts/pull/2138)中添加了对[Lumina-Image 2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0)的支持。特别感谢sdbds和RockerBOO的贡献。
    - 请参考[Lumina-Image 2.0文档](./docs/lumina_train_network.md)了解更多详情。
- 我们开始向[docs](./docs)添加全面的训练相关文档。这些文档是在生成式AI的帮助下创建的，并将随着时间推移进行更新。虽然在此阶段仍有许多不足，但我们计划逐步改进。

    目前，以下文档可用：
    - train_network.md
    - sdxl_train_network.md
    - sdxl_train_network_advanced.md
    - flux_train_network.md
    - sd3_train_network.md
    - lumina_train_network.md
    
2025年7月10日：
- README中添加了[AI编码代理](#面向使用-ai-编码代理的开发者)部分。此部分为使用Claude和Gemini等AI编码代理的开发者提供了项目上下文和编码标准的说明。

2025年5月1日：
- 解决了在启用DeepSpeed的情况下使用混合精度在flux_train.py中训练FLUX.1时的错误。感谢sharlynxy的PR [#2060](https://github.com/kohya-ss/sd-scripts/pull/2060)。详情请参见PR。
  - 如果启用DeepSpeed，请使用`pip install deepspeed==0.16.7`安装DeepSpeed。

2025年4月27日：
- FLUX.1训练现在支持在训练期间样本生成中的CFG比例。请使用`--g`选项指定CFG比例（注意`--l`用于嵌入的指导比例。）PR [#2064](https://github.com/kohya-ss/sd-scripts/pull/2064)。
    - 详情请参见[这里](#训练期间的样本图像生成)。
    - 如果您遇到任何问题，请告知我们。

// 这是省略的部分

该功能尚未完全测试，因此可能存在错误。如果您发现任何问题，请提交Issue。

使用ControlNet数据集指定掩码。掩码图像应为RGB图像。R通道中像素值255被视为掩码（仅对带有掩码的像素计算损失），0被视为非掩码。像素值0-255转换为0-1（即像素值128被视为损失的一半权重）。有关数据集规范的详细信息，请参见[LLLite文档](./docs/train_lllite_README.md#preparing-the-dataset)。

#### 关于Scheduled Huber Loss

Scheduled Huber Loss已引入到每个训练脚本中。这是一种提高对训练数据中异常值或异常（数据损坏）鲁棒性的方法。

使用传统的MSE（L2）损失函数时，异常值的影响可能很大，可能导致生成图像质量下降。另一方面，虽然Huber损失函数可以抑制异常值的影响，但它往往会损害图像细节的再现。

为了解决这个问题，所提出的方法巧妙地应用了Huber损失函数。通过在训练早期阶段（噪声较高时）使用Huber损失，在后期阶段使用MSE进行调度，它在异常值鲁棒性和细节再现之间取得了平衡。

实验结果证实，与纯Huber损失或MSE相比，该方法在包含异常值的数据上实现了更高的准确性。计算成本的增加微乎其微。

新添加的参数loss_type、huber_schedule和huber_c允许选择损失函数类型（Huber、smooth L1、MSE）、调度方法（指数、常数、SNR）和Huber参数。这使得可以根据数据集的特征进行优化。

详情请参见PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/)。

- `loss_type`：指定损失函数类型。选择`huber`表示Huber损失，`smooth_l1`表示smooth L1损失，`l2`表示MSE损失。默认为`l2`，与之前相同。
- `huber_schedule`：指定调度方法。选择`exponential`、`constant`或`snr`。默认为`snr`。
- `huber_c`：指定Huber参数。默认为`0.1`。

请阅读[Releases](https://github.com/kohya-ss/sd-scripts/releases)了解最近的更新。

#### 主要变更点

- 依赖库已更新。请参见[升级](./README-ja.md#升级)并更新库。
  - 特别是`imagesize`已新添加，请立即使用`pip install imagesize==1.4.1`单独安装，如果无法立即更新库。
  - `bitsandbytes==0.43.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6`已包含在requirements.txt中。
    - 由于`bitsandbytes`官方支持Windows，因此不再需要复杂的步骤。
  - 此外，PyTorch版本已更新到2.1.2。PyTorch不需要立即更新。更新时，升级步骤不会更新PyTorch，因此请手动安装torch、torchvision、xformers。
- 启用向wandb的日志输出时，整个命令行将被公开。因此，如果命令行包含wandb的API密钥或HuggingFace的令牌等，请建议在配置文件（`.toml`）中记录。感谢提出问题的bghira。
  - 在这种情况下，训练开始时会显示警告。
  - 此外，如果有绝对路径指定，该路径可能会被公开，因此建议指定相对路径或在配置文件中记录。在这种情况下会显示INFO日志。
  - 详情请参见[#1123](https://github.com/kohya-ss/sd-scripts/pull/1123)以及PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240)。
- 在Colab中运行时，日志输出似乎会导致停止。请在训练脚本中指定`--console_log_simple`选项以禁用rich日志记录并进行尝试。
- 其他还包括添加掩码损失、添加Scheduled Huber Loss、DeepSpeed支持、数据集设置改进、图像标记改进等。详情请参见以下内容。

#### 训练脚本

- 在`train_network.py`和`sdxl_train_network.py`中，修改了训练模型的元数据以记录部分数据集设置（`caption_prefix`、`caption_suffix`、`keep_tokens_separator`、`secondary_separator`、`enable_wildcard`）。
- 在`train_network.py`和`sdxl_train_network.py`中，修复了state中包含U-Net和Text Encoder的错误。state的保存和加载速度加快，文件大小减小，加载时的内存使用量也减少。
- 支持DeepSpeed。PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)、[#1139](https://github.com/kohya-ss/sd-scripts/pull/1139)感谢BootsofLagrangian。详情请参见PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)。
- 在各训练脚本中支持掩码损失。PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207)详情请参见[关于掩码损失](#关于掩码损失)。
- 在各训练脚本中添加了Scheduled Huber Loss。PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/)感谢提出建议的kabachuha，以及深化讨论的cheald、drhead等各位。详情请参见该PR以及[Scheduled Huber Loss关于](#scheduled-huber-loss-关于)。
- 在各训练脚本中添加了`--noise_offset_random_strength`和`--ip_noise_gamma_random_strength`选项，使noise offset和ip noise gamma在0到指定值范围内变动。PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177)感谢KohakuBlueleaf。
- 在各训练脚本中添加了训练结束时保存state的`--save_state_on_train_end`选项。PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168)感谢gesen2egee。
- 在各训练脚本中，当`--sample_every_n_epochs`和`--sample_every_n_steps`选项指定0以下数值时，修改为显示警告并忽略它们。感谢提出问题的S-Del。

#### 数据集设置

- 数据集设置的`.toml`文件现在以UTF-8编码读取。PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167)感谢Horizon1704。
- 修复了在数据集设置中指定多个正则化图像子集时，最后一个子集的各种设置应用于所有子集图像的错误。现在每个子集的设置正确应用于各自的图像。PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205)感谢feffy380。
- 在数据集的子集设置中添加了一些功能。
  - 添加了不参与混洗的标签分割标识符指定`secondary_separator`。例如指定`secondary_separator=";;;"`。通过`secondary_separator`分割，该部分在混洗、drop时会被一起处理。
  - 添加了`enable_wildcard`。设为`true`时可以使用通配符表示法`{aaa|bbb|ccc}`。还启用多行标题。
  - `keep_tokens_separator`现在可以在标题中使用两次。例如指定`keep_tokens_separator="|||"`时，标题为`1girl, hatsune miku, vocaloid ||| stage, mic ||| best quality, rating: general`时，第二个`|||`分割的部分在混洗、drop时不会被处理并保留在末尾。
  - 可以与现有功能`caption_prefix`和`caption_suffix`一起使用。`caption_prefix`和`caption_suffix`首先处理，然后按顺序处理通配符、`keep_tokens_separator`、混洗和drop、`secondary_separator`。
  - 详情请参见[数据集设置](./docs/config_README-ja.md)。
- 在DreamBooth方式的DataSet中添加了缓存图像信息（大小、标题）的功能。PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178)、[#1206](https://github.com/kohya-ss/sd-scripts/pull/1206)感谢KohakuBlueleaf。详情请参见[数据集设置](./docs/config_README-ja.md#dreambooth-方式专用的选项)。
- 添加了数据集设置的[英文文档](./docs/config_README-en.md)。PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175)感谢darkstorm2150。

#### 图像标记

- `tag_image_by_wd14_tagger.py`支持v3仓库（仅在指定`--onnx`时有效）。PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192)感谢sdbds。
  - 可能需要升级Onnx版本。默认情况下未安装Onnx，请使用`pip install onnx==1.15.0 onnxruntime-gpu==1.17.1`等安装、升级。请同时查看`requirements.txt`的注释。
- `tag_image_by_wd14_tagger.py`现在将模型保存在`--repo_id`的子目录中。这样可以缓存多个模型文件。请删除`--model_dir`直接下的不必要文件。
- 在`tag_image_by_wd14_tagger.py`中添加了一些选项。
  - 一些在PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216)中添加。感谢Disty0。
  - 输出评级标签的`--use_rating_tags`和`--use_rating_tags_as_last_tag`
  - 将角色标签首先输出的`--character_tags_first`
  - 展开角色标签和系列的`--character_tag_expand`
  - 指定始终首先输出的标签的`--always_first_tags`
  - 替换标签的`--tag_replacement`
  - 详情请参见[标记相关文档](./docs/wd14_tagger_README-ja.md)。
- 修复了在`make_captions.py`中指定`--beam_search`且`--num_beams`指定2以上值时的错误。

#### 关于掩码损失

各训练脚本支持掩码损失。要启用掩码损失，请指定`--masked_loss`选项。

该功能尚未完全测试，因此可能存在错误。如果出现错误，请提交Issue将非常有帮助。

使用ControlNet数据集指定掩码。掩码图像必须是RGB图像。R通道像素值255为损失计算对象，0为损失计算对象外。0-255的值转换为0-1范围（即像素值128的部分损失权重为一半）。详情请参见[LLLite文档](./docs/train_lllite_README-ja.md#数据集的准备)。

#### 关于Scheduled Huber Loss

各训练脚本引入了提高对训练数据中异常值或异常（数据损坏）鲁棒性的方法Scheduled Huber Loss。

传统的MSE（L2）损失函数会受到异常值的很大影响，可能导致生成图像质量下降。另一方面，Huber损失函数可以抑制异常值的影响，但往往会损害图像的细节再现性。

该方法通过巧妙地应用Huber损失函数，在训练的早期阶段（噪声较大时）使用Huber损失，在后期阶段使用MSE进行调度，从而在异常值耐受性和细节再现性之间取得平衡。

实验结果表明，与纯Huber损失或MSE相比，该方法在包含异常值的数据中实现了更高的精度。计算成本的增加微乎其微。

具体来说，通过新增的参数loss_type、huber_schedule、huber_c，可以选择损失函数的类型（Huber、smooth L1、MSE）和调度方法（指数、常数、SNR）。这使得可以根据数据集进行优化。

详情请参见PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/)。

- `loss_type`：指定损失函数的类型。`huber`表示Huber损失，`smooth_l1`表示smooth L1损失，`l2`表示MSE损失。默认为`l2`，与之前相同。
- `huber_schedule`：指定调度方法。`exponential`表示指数函数，`constant`表示恒定，`snr`表示基于信噪比的调度。默认为`snr`。
- `huber_c`：指定Huber损失的参数。默认为`0.1`。

PR内分享了一些比较。如果要尝试此功能，最初可以尝试使用`--loss_type smooth_l1 --huber_schedule snr --huber_c 0.1`等。

请查看[Release](https://github.com/kohya-ss/sd-scripts/releases)了解最近的更新信息。

## 附加信息

### LoRA的命名

为了避免混淆，`train_network.py`支持的LoRA已被命名。文档已更新。以下是该仓库中LoRA类型的名称。

1. __LoRA-LierLa__：（用于__Li__n__e__a__r__ __La__yers的LoRA）

    用于Linear层和1x1内核的Conv2d层的LoRA

2. __LoRA-C3Lier__：（用于具有__3__x3内核的__C__onvolutional层和__Li__n__e__a__r__层的LoRA）

    除了1.之外，还用于具有3x3内核的Conv2d层
    
LoRA-LierLa是`train_network.py`的默认LoRA类型（不带`conv_dim`网络参数）。
<!-- 
LoRA-LierLa可以与[AUTOMATIC1111的Web UI扩展](https://github.com/kohya-ss/sd-webui-additional-networks)一起使用，或与Web UI的内置LoRA功能一起使用。

要将LoRA-C3Lier与Web UI一起使用，请使用我们的扩展。
-->

### 训练期间的样本图像生成
  例如，提示文件可能如下所示

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  以`#`开头的行是注释。您可以在提示后使用`--n`等选项指定生成图像的选项。可以使用以下选项。

  * `--n` 负面提示直到下一个选项。当CFG比例为`1.0`时忽略。
  * `--w` 指定生成图像的宽度。
  * `--h` 指定生成图像的高度。
  * `--d` 指定生成图像的种子。
  * `--l` 指定生成图像的CFG比例。对于FLUX.1模型，默认为`1.0`，表示无CFG。对于Chroma模型，设置为约`4.0`以启用CFG。
  * `--g` 指定嵌入指导比例对于具有嵌入指导的模型（FLUX.1），默认为`3.5`。对于Chroma模型设置为`0.0`。
  * `--s` 指定生成的步数。

  `( )`和`[ ]`等提示加权正在工作。

#### 时间步长分布

`--timestep_sampling`和`--sigmoid_scale`、`--discrete_flow_shift`调整时间步长的分布。分布如下图所示。

使用`--timestep_sampling shift`时`--discrete_flow_shift`的效果（当未指定`--sigmoid_scale`时，默认为1.0）：
![Figure_2](https://github.com/user-attachments/assets/d9de42f9-f17d-40da-b88d-d964402569c6)

`--timestep_sampling sigmoid`和`--timestep_sampling uniform`之间的差异（当指定`--timestep_sampling sigmoid`或`uniform`时，`--discrete_flow_shift`被忽略）：
![Figure_3](https://github.com/user-attachments/assets/27029009-1f5d-4dc0-bb24-13d02ac4fdad)

`--timestep_sampling sigmoid`和`--sigmoid_scale`的效果（当指定`--timestep_sampling sigmoid`时，`--discrete_flow_shift`被忽略）：
![Figure_4](https://github.com/user-attachments/assets/08a2267c-e47e-48b7-826e-f9a080787cdc)

#### FLUX.1 LoRA训练的关键特性

1. CLIP-L和T5XXL LoRA支持：
   - FLUX.1 LoRA训练现在支持CLIP-L和T5XXL LoRA训练。
   - 从命令中删除`--network_train_unet_only`。
   - 在`--network_args`中添加`train_t5xxl=True`以训练T5XXL LoRA。同时训练CLIP-L。
   - 可以缓存T5XXL输出以进行CLIP-L LoRA训练。因此，`--cache_text_encoder_outputs`或`--cache_text_encoder_outputs_to_disk`也可用。
   - 可以分别指定CLIP-L和T5XXL的学习率。可以在`--text_encoder_lr`中指定多个数字。例如，`--text_encoder_lr 1e-4 1e-5`。第一个值是CLIP-L的学习率，第二个值是T5XXL的学习率。如果只指定一个，CLIP-L和T5XXL的学习率将相同。如果未指定`--text_encoder_lr`，则CLIP-L和T5XXL都使用默认学习率`--learning_rate`。
   - 训练的LoRA可以与ComfyUI一起使用。
   - 注意：`flux_extract_lora.py`、`convert_flux_lora.py`和`merge_flux_lora.py`尚不支持CLIP-L和T5XXL LoRA。

    | 训练的LoRA|选项|network_args|cache_text_encoder_outputs (*1)|
    |---|---|---|---|
    |FLUX.1|`--network_train_unet_only`|-|o|
    |FLUX.1 + CLIP-L|-|-|o (*2)|
    |FLUX.1 + CLIP-L + T5XXL|-|`train_t5xxl=True`|-|
    |CLIP-L (*3)|`--network_train_text_encoder_only`|-|o (*2)|
    |CLIP-L + T5XXL (*3)|`--network_train_text_encoder_only`|`train_t5xxl=True`|-|

    - *1: `--cache_text_encoder_outputs`或`--cache_text_encoder_outputs_to_disk`也可用。
    - *2: 可以缓存T5XXL输出以进行CLIP-L LoRA训练。
    - *3: 尚未测试。

2. 实验性FP8/FP16混合训练：
   - `--fp8_base_unet`启用使用fp8进行FLUX训练，使用bf16/fp16进行CLIP-L/T5XXL训练。
   - 可以使用fp8训练FLUX，使用bf16/fp16训练CLIP-L/T5XXL。
   - 指定此选项时，自动启用`--fp8_base`选项。

3. 分割Q/K/V投影层（实验性）：
   - 添加了分割注意力中q/k/v/txt的投影层并对每个层应用LoRA的选项。
   - 在network_args中指定`"split_qkv=True"`，如`--network_args "split_qkv=True"`（也可用`train_blocks`）。
   - 可能会增加表现力但也增加训练时间。
   - 保存的模型与sd-scripts中的普通LoRA模型兼容，可在ComfyUI等环境中使用。
   - 使用`convert_flux_lora.py`转换为AI-toolkit（Diffusers）格式将减小大小。
   
4. T5注意力掩码应用：
   - 当指定`--apply_t5_attn_mask`时应用T5注意力掩码。
   - 现在在编码T5和Double和Single Blocks的注意力时应用掩码
   - 影响`flux_minimal_inference.py`中的微调、LoRA训练和推理。

5. 多分辨率训练支持：
   - FLUX.1现在支持多分辨率训练，即使将潜在变量缓存到磁盘。

Q/K/V分割的技术细节：

在Black Forest Labs模型的实现中，q/k/v（和single blocks中的txt）的投影层被连接成一个。如果在那里直接添加LoRA，LoRA模块只有一个，且维度很大。相比之下，在Diffusers的实现中，q/k/v/txt的投影层是分开的。因此，LoRA模块分别应用于q/k/v/txt，维度较小。此选项用于训练类似于后者的LoRA。

通过连接多个LoRA的权重确保保存模型（state dict）的兼容性。但是，由于某些部分有零权重，模型大小会很大。

#### 指定FLUX.1中每层的秩

通过指定以下network_args可以指定FLUX.1中每层的秩。如果指定`0`，则不会对该层应用LoRA。

当未指定network_args时，应用默认值（`network_dim`），与之前相同。

|network_args|目标层|
|---|---|
|img_attn_dim|DoubleStreamBlock中的img_attn|
|txt_attn_dim|DoubleStreamBlock中的txt_attn|
|img_mlp_dim|DoubleStreamBlock中的img_mlp|
|txt_mlp_dim|DoubleStreamBlock中的txt_mlp|
|img_mod_dim|DoubleStreamBlock中的img_mod|
|txt_mod_dim|DoubleStreamBlock中的txt_mod|
|single_dim|SingleStreamBlock中的linear1和linear2|
|single_mod_dim|SingleStreamBlock中的调制|

调试时还可使用`"verbose=True"`。它显示每层的秩。

示例：
```
--network_args "img_attn_dim=4" "img_mlp_dim=8" "txt_attn_dim=2" "txt_mlp_dim=2" 
```