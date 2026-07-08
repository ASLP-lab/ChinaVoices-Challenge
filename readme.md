# ChinaVoices Challenge 2026 Baseline

本项目是 [ChinaVoices Challenge 2026](https://aslp-lab.github.io/ChinaVoices-Challenge/) 的 baseline 系统，用于中文多方言语音识别与中文多方言种类识别任务。系统使用 ms-swift 框架对 Qwen3-ASR-1.7B 进行 LoRA 微调。发布包含推理脚本、评测脚本和示例数据；模型权重体积较大，将通过百度网盘单独提供。

## 任务说明

ChinaVoices Challenge 2026 面向中文多方言语音处理，包含两个核心任务：

- 中文多方言种类识别：根据输入语音判断方言类别，评价指标为 Accuracy。
- 中文多方言语音识别：将输入语音转写为对应文本，评价指标为 WER。

本 baseline 的推理输出同时包含方言预测和文本转写，可用于上述两个任务的评测。

## 模型与训练设置

本模型以 `Qwen/Qwen3-ASR-1.7B` 为基座，使用 ms-swift 进行监督微调，训练的数据来自竞赛官方提供的中文方言语音开源数据公共清单和部分参考数据集。参考集被划分为两部分：一部分用于训练，一部分用于训练阶段评估；两部分包含不同说话人的音频，避免同一说话人同时出现在训练与评估数据中。训练脚本默认冻结音频 encoder 和 audio projector/aligner，只在 LLM 侧注入 LoRA，用音频-文本样本学习输出方言标签和 ASR 文本。训练数据采用 ms-swift 多模态 JSONL 格式，目标文本统一写成 `language Chinese <dialect><asr_text><transcript>`，使模型在一次生成中同时输出方言标签和转写文本。

主要训练配置如下：

- 训练框架：ms-swift，`swift_version=4.4.0.dev0`。
- 基座模型：`Qwen/Qwen3-ASR-1.7B`。
- 微调方式：PEFT LoRA，只训练 LLM 侧 LoRA 参数。
- LoRA 设置：`r=8`，`alpha=32`，`dropout=0.05`，`bias=none`。
- 冻结策略：`freeze_vit=true`、`freeze_aligner=true`，即冻结 audio encoder 和 audio projector/aligner，仅训练 LLM 侧 LoRA 参数。
- 精度：`bfloat16`。
- 分布式训练：4 卡训练，DeepSpeed ZeRO-2。
- batch 设置：`per_device_train_batch_size=16`，`gradient_accumulation_steps=8`。
- 优化器：`adamw_torch_fused`，`beta1=0.9`，`beta2=0.95`，`weight_decay=0.1`，`max_grad_norm=1.0`。
- 学习率：`1e-4`，cosine scheduler，`warmup_ratio=0.05`。

发布的 `Chinavoice_Challenge` 是 merge 后的完整模型目录，推理时不需要再额外加载 LoRA adapter。

## 目录结构

```text
.
├── data/
│   └── reference_set_eval.jsonl           # 从发布参考集中划分出的训练评估数据
├── infer/
│   └── infer_batch.py                     # 批量推理脚本
├── eval/
│   ├── eval_jsonl_with_wer_tools.sh       # WER/方言准确率评测入口
│   ├── jsonl_to_text_for_wer.py
│   ├── jsonl_to_text_by_dialect_for_wer.py
│   └── tools/
│       ├── clean_data.py
│       └── wer.py
└── ckpt/
    └── Chinavoice_Challenge/             # 模型权重目录，需单独下载后放置
```

注意：当前开源包默认不包含模型权重。请下载模型后保持目录名为 `Chinavoice_Challenge`，并放到 `ckpt/` 下。

## 数据说明

`data/reference_set_eval.jsonl` 是从发布的参考集中分离出来的一部分训练评估数据，采用 ms-swift 多模态 JSONL 格式。发布参考集被划分为两部分：一部分用于训练，一部分用于训练阶段评估；两部分包含不同说话人的音频，避免同一说话人同时出现在训练与评估数据中。

## 模型下载

模型权重将通过百度网盘提供：

```text
百度网盘链接：https://pan.baidu.com/s/1URvbemFgL-iwBOKGjWJT7Q?pwd=pkq4
提取码：pkq4
```

## 环境准备

本项目在 ms-swift 训练环境中开发和验证。由于不同参赛者的 CUDA、PyTorch、GPU 型号和集群环境可能不同，本文档不提供固定的环境安装命令或完整 `requirements.txt`。

请参赛者自行参考 ms-swift 和 Qwen3-ASR 官方文档配置可运行的训练/推理环境。

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
  --data data/reference_set_eval.jsonl \
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
  --data data/reference_set_eval.jsonl \
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

推理完成后，可运行评测脚本计算整体 WER、方言识别准确率，并可按方言分别统计 WER：

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
- `--apply_t2s`：是否在计算 WER 前进行繁体转简体，默认 `1`。
- `--by_dialect`：是否按 `ref_dialect` 分方言统计 WER，默认 `1`。

## Baseline 指标

下表为 baseline 在 `data/reference_set_eval.jsonl` 上的结果。中文多方言语音识别的原始基线模型为 `Qwen3-ASR-1.7B`；中文多方言种类识别没有单独的原始基线模型。

### 中文多方言语音识别

| 方言名称          | Qwen3-ASR-1.7B WER (%) | ChinaVoices Baseline WER (%) |
| ----------------- | ---------------------: | ---------------------------: |
| anhui（安徽）     |                  15.84 |                        14.99 |
| cantonese（粤语） |                   8.54 |                         7.64 |
| changsha（长沙）  |                  13.22 |                        11.14 |
| chaoshan（潮汕）  |                  47.28 |                        38.37 |
| dongbei（东北）   |                   5.82 |                         4.02 |
| henan（河南）     |                  10.48 |                        10.06 |
| kejia（客家）     |                  58.98 |                        49.42 |
| minnan（闽南）    |                  26.59 |                        23.39 |
| nanchang（南昌）  |                  39.92 |                        36.44 |
| nanjing（南京）   |                   9.51 |                         8.12 |
| shan1xi（晋语）   |                  23.80 |                        21.12 |
| shan3xi（陕西）   |                   8.05 |                         6.94 |
| shandong（山东）  |                   8.34 |                         6.90 |
| sichuan（四川）   |                   6.79 |                         6.71 |
| wuhan（武汉）     |                   6.41 |                         6.41 |
| wuyu（吴语）      |                  61.03 |                        58.10 |
| Average           |                  21.91 |                        19.36 |

### 中文多方言种类识别

| 方言名称          | ChinaVoices Baseline Accuracy (%) |
| ----------------- | --------------------------------: |
| anhui（安徽）     |                             44.71 |
| cantonese（粤语） |                            100.00 |
| changsha（长沙）  |                             81.30 |
| chaoshan（潮汕）  |                             80.12 |
| dongbei（东北）   |                            100.00 |
| henan（河南）     |                             19.75 |
| kejia（客家）     |                              0.15 |
| minnan（闽南）    |                             46.96 |
| nanchang（南昌）  |                              5.96 |
| nanjing（南京）   |                             15.14 |
| shan1xi（晋语）   |                             23.96 |
| shan3xi（陕西）   |                             67.24 |
| shandong（山东）  |                             52.37 |
| sichuan（四川）   |                             83.98 |
| wuhan（武汉）     |                             17.50 |
| wuyu（吴语）      |                              1.66 |
| Average           |                             46.30 |

## 一键示例

```bash
# 1. 下载模型并放置到 ckpt/Chinavoice_Challenge

# 2. 推理
python infer/infer_batch.py \
  --model ckpt/Chinavoice_Challenge \
  --data data/reference_set_eval.jsonl \
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
- `--apply_t2s 1` 依赖 `opencc-python-reimplemented`。
- 本 baseline 面向竞赛复现和快速上手，鼓励参赛者在此基础上改进数据处理、训练策略和解码策略。
