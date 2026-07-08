# ChinaVoices Challenge 2026 Baseline: Qwen3-ASR

本项目是 [ChinaVoices Challenge 2026](https://aslp-lab.github.io/ChinaVoices-Challenge/) 的 baseline 系统，用于中文多方言语音识别与中文多方言种类识别任务。系统基于 ms-swift 对 Qwen3-ASR-1.7B 的 LLM 侧进行 LoRA 微调，训练时冻结音频 encoder 和 audio projector/aligner，并将 LoRA 权重 merge 后作为完整模型发布。发布包包含推理脚本、评测脚本和示例数据；模型权重体积较大，将通过百度网盘单独提供。

## 任务说明

ChinaVoices Challenge 2026 面向中文多方言语音处理，包含两个核心任务：

- 中文多方言种类识别：根据输入语音判断方言类别，评价指标为 Accuracy。
- 中文多方言语音识别：将输入语音转写为对应文本，评价指标为 CER。

本 baseline 的推理输出同时包含方言预测和文本转写，可用于上述两个任务的评测。

## 模型与训练设置

本模型以 `Qwen/Qwen3-ASR-1.7B` 为基座，使用 ms-swift 进行监督微调。需要说明的是，本 baseline 没有全参数训练整个 Qwen3-ASR：训练脚本默认冻结音频 encoder 和 audio projector/aligner，只在 LLM 侧注入 LoRA，用音频-文本样本学习输出方言标签和 ASR 文本。训练数据采用 ms-swift 多模态 JSONL 格式，目标文本统一写成 `language Chinese <dialect><asr_text><transcript>`，使模型在一次生成中同时输出方言标签和转写文本。

主要训练配置如下：

- 训练框架：ms-swift，`swift_version=4.4.0.dev0`。
- 基座模型：`Qwen/Qwen3-ASR-1.7B`。
- 微调方式：PEFT LoRA，只训练 LLM 侧 LoRA 参数；训练完成后将 LoRA merge 到基座模型并作为完整模型发布。
- LoRA 设置：`r=8`，`alpha=32`，`dropout=0.05`，`bias=none`。
- 冻结策略：`freeze_vit=true`、`freeze_aligner=true`，即冻结 audio tower 和 audio projector/aligner，仅训练 LLM 侧 LoRA 参数。
- 精度：`bfloat16`。
- 分布式训练：4 卡训练，DeepSpeed ZeRO-2。
- batch 设置：`per_device_train_batch_size=8`，`gradient_accumulation_steps=16`，全局有效 batch size 约为 `8 x 16 x 4 = 512`。
- 优化器：`adamw_torch_fused`，`beta1=0.9`，`beta2=0.95`，`weight_decay=0.1`，`max_grad_norm=1.0`。
- 学习率：`1e-4`，cosine scheduler，`warmup_ratio=0.05`。
- 序列长度：`max_length=2048`，训练时启用 `gradient_checkpointing` 和 `lazy_tokenize`。

发布的 `Chinavoice_Challenge` 是 merge 后的完整模型目录，推理时不需要再额外加载 LoRA adapter。

## 目录结构

```text
.
├── data/
│   └── dev_ms.jsonl                      # ms-swift 格式示例数据
├── infer/
│   └── infer_batch.py                     # 批量推理脚本
├── eval/
│   ├── eval_jsonl_with_wer_tools.sh       # CER/方言准确率评测入口
│   ├── jsonl_to_text_for_wer.py
│   ├── jsonl_to_text_by_dialect_for_wer.py
│   └── tools/
│       ├── clean_data.py
│       └── wer.py
└── ckpt/
    └── Chinavoice_Challenge/             # 模型权重目录，需单独下载后放置
```

注意：当前开源包默认不包含模型权重。请下载模型后保持目录名为 `Chinavoice_Challenge`，并放到 `ckpt/` 下。

## 模型下载

模型权重将通过百度网盘提供：

```text
百度网盘链接：TODO
```


## 环境准备

本项目在 ms-swift 训练环境中开发和验证。由于不同参赛者的 CUDA、PyTorch、GPU 型号和集群环境可能不同，本文档不提供固定的环境安装命令或完整 `requirements.txt`。

请参赛者自行参考 ms-swift 和 Qwen3-ASR 官方文档配置可运行的训练/推理环境，并确保至少能够正常导入和使用以下组件：

- PyTorch
- ms-swift
- qwen-asr
- peft
- deepspeed，如果需要复现多卡 LoRA 训练
- opencc-python-reimplemented，如果评测时使用 `--apply_t2s 1` 做繁体转简体

推理脚本默认使用 `torch.bfloat16`，建议在支持 bfloat16 的 GPU 上运行；如果显存不足，可以调小 `--batch-size`。

## 输入数据格式

推理脚本读取 ms-swift 风格 JSONL。每行一个样本，至少包含：

- `messages`：对话消息，其中 assistant 内容为参考答案。
- `audios`：音频路径列表，当前脚本使用第一个音频路径。

示例：

```json
{"messages": [{"role": "user", "content": "<audio>"}, {"role": "assistant", "content": "language Chinese anhui<asr_text>那部搞笑电影我传上去了"}], "audios": ["/path/to/audio.wav"]}
```

assistant 文本格式为：

```text
language Chinese <dialect><asr_text><transcript>
```

其中 `<dialect>` 是方言类别，`<transcript>` 是语音转写文本。

## 批量推理

在项目根目录运行：

```bash
python infer/infer_batch.py \
  --model ckpt/Chinavoice_Challenge \
  --data data/dev_ms.jsonl \
  --output outputs/pred.jsonl \
  --device-map auto \
  --batch-size 8 \
  --max-tokens 512
```

参数说明：

- `--model`：完整模型目录，目录下应包含 `config.json` 和 `model.safetensors`。
- `--data`：输入 JSONL 文件。
- `--output`：推理结果保存路径。
- `--device-map`：transformers 的 `device_map`，常用 `auto` 或 `cuda:0`。
- `--batch-size`：推理 batch size；显存不足时请调小。
- `--max-tokens`：每条音频最大生成 token 数。

多卡推理可先设置可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 python infer/infer_batch.py \
  --model ckpt/Chinavoice_Challenge \
  --data data/dev_ms.jsonl \
  --output outputs/pred.jsonl \
  --device-map auto \
  --batch-size 16
```

## 推理输出格式

`outputs/pred.jsonl` 每行包含原始样本信息和预测结果，主要字段如下：

- `utt_id`：样本 ID。
- `audio_path`：音频路径。
- `ref_full`：完整参考标签。
- `ref_text`：参考转写文本。
- `ref_dialect`：参考方言类别。
- `pred_full`：模型原始输出，通常形如 `language Chinese <dialect><asr_text><text>`。
- `pred_text`：预测转写文本。
- `pred_dialect`：预测方言类别。
- `error`：推理异常信息；正常为空字符串。

## 评测

推理完成后，可运行评测脚本计算整体 CER、方言识别准确率，并可按方言分别统计 CER：

```bash
bash eval/eval_jsonl_with_wer_tools.sh \
  --pred_jsonl outputs/pred.jsonl \
  --output_dir outputs/wer_eval \
  --apply_t2s 1 \
  --by_dialect 1
```

参数说明：

- `--pred_jsonl`：推理输出 JSONL。
- `--output_dir`：评测结果输出目录。
- `--apply_t2s`：是否在计算 CER 前进行繁体转简体，默认 `1`。
- `--by_dialect`：是否按 `ref_dialect` 分方言统计 CER，默认 `1`。


## 一键示例

```bash
# 1. 下载模型并放置到 ckpt/Chinavoice_Challenge

# 2. 推理
python infer/infer_batch.py \
  --model ckpt/Chinavoice_Challenge \
  --data data/dev_ms.jsonl \
  --output outputs/pred.jsonl \
  --device-map auto \
  --batch-size 8

# 3. 评测
bash eval/eval_jsonl_with_wer_tools.sh \
  --pred_jsonl outputs/pred.jsonl \
  --output_dir outputs/wer_eval \
  --apply_t2s 1 \
  --by_dialect 1
```

## 注意事项

- 请确保 JSONL 中的音频路径在当前机器上可访问。
- 如果出现 CUDA OOM，请优先调小 `--batch-size`。
- 如果运行环境无法访问互联网，请提前安装依赖并下载模型权重；推理阶段不需要联网。
- `--apply_t2s 1` 依赖 `opencc-python-reimplemented`。
- 本 baseline 面向竞赛复现和快速上手，鼓励参赛者在此基础上改进数据处理、训练策略和解码策略。

## 引用与致谢

- ChinaVoices Challenge 2026: https://aslp-lab.github.io/ChinaVoices-Challenge/
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR

请在使用本 baseline 或提交系统说明时，按照竞赛要求披露所使用的数据、预训练模型、训练策略和外部资源。
