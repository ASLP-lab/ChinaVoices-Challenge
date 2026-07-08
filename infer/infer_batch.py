import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import normalize_audios

# -------------------------- 默认配置：也可以用命令行参数覆盖 --------------------------
MODEL_NAME = "ckpt/Chinavoice_Challenge"
ADAPTER_PATH = ""
DATA_FILE_PATH = "data/reference_set_eval.jsonl"
SAVE_RESULT_PATH = "outputs/pred.jsonl"
DEVICE_MAP = "auto"       # 用多卡时建议运行前设置 CUDA_VISIBLE_DEVICES=0,1；auto 会自动切分到可见 GPU
BATCH_SIZE = 64             # transformers backend 的实际推理 batch，OOM 就调小
MAX_TOKENS = 512
# -------------------------------------------------------------------------------

ASR_MARKER = "<asr_text>"
DIALECT_RE = re.compile(r"language\s+Chinese\s+([^\s<]+)", re.IGNORECASE)


def split_asr_content(content: str) -> Dict[str, str]:
    """解析 language Chinese anhui<asr_text>... 这种训练/预测目标。"""
    content = (content or "").strip()
    if ASR_MARKER in content:
        prefix, text = content.split(ASR_MARKER, 1)
    else:
        prefix, text = "", content

    dialect = ""
    match = DIALECT_RE.search(prefix)
    if match:
        dialect = match.group(1).strip()

    return {
        "full": content,
        "prefix": prefix.strip(),
        "dialect": dialect,
        "text": text.strip(),
    }


def get_assistant_content(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def load_ms_swift_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 ms-swift 多模态 JSONL：messages + audios。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audios = obj.get("audios") or []
            if not audios:
                raise ValueError(f"第 {line_no} 行没有 audios 字段: {path}")

            audio_path = audios[0]
            ref = split_asr_content(get_assistant_content(obj.get("messages") or []))
            utt_id = obj.get("id") or Path(audio_path).stem or str(line_no)
            samples.append({
                "utt_id": utt_id,
                "audio_path": audio_path,
                "ref_full": ref["full"],
                "ref_text": ref["text"],
                "ref_dialect": ref["dialect"],
            })
    return samples


def infer_raw(asr: Qwen3ASRModel, audio_paths: List[str]) -> List[str]:
    """直接拿 Qwen3-ASR 原始生成串，保留 language Chinese xxx<asr_text> 标签。"""
    wavs = normalize_audios(audio_paths)
    contexts = [""] * len(wavs)
    languages = [None] * len(wavs)
    return asr._infer_asr(contexts, wavs, languages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for Qwen3-ASR ms-swift JSONL data.")
    parser.add_argument("--model", default=MODEL_NAME, help="基座模型或已 merge 后的完整模型路径")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="LoRA adapter checkpoint 路径；如果模型已 merge，可设为空字符串")
    parser.add_argument("--data", default=DATA_FILE_PATH, help="ms-swift 格式 JSONL，例如 data/reference_set_eval.jsonl")
    parser.add_argument("--output", default=SAVE_RESULT_PATH, help="推理结果 JSONL 保存路径")
    parser.add_argument("--device-map", default=DEVICE_MAP, help="transformers device_map，常用 auto 或 cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="推理 batch，显存不够就调小")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="每条音频最多生成 token 数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_path = args.adapter.strip() if args.adapter else ""
    if not (Path(args.model) / "config.json").exists():
        raise FileNotFoundError(f"--model 需要指向完整模型目录，里面应有 config.json；当前是: {args.model}")
    if adapter_path and not (Path(adapter_path) / "adapter_config.json").exists():
        raise FileNotFoundError(f"--adapter 需要指向 LoRA 目录，里面应有 adapter_config.json；当前是: {adapter_path}")

    print(f"正在读取数据: {args.data}")
    samples = load_ms_swift_jsonl(args.data)
    print(f"共 {len(samples)} 条样本")

    print(f"正在加载 Qwen3-ASR 模型: {args.model}")
    print(f"device_map: {args.device_map}")
    asr = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"正在加载 LoRA adapter: {adapter_path}")
        asr.model = PeftModel.from_pretrained(asr.model, adapter_path)
        asr.model.eval()

    done = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            audio_paths = [sample["audio_path"] for sample in batch]

            try:
                pred_full_list = infer_raw(asr, audio_paths)
            except Exception as exc:
                print(f"[错误] batch {start}-{start + len(batch) - 1} 推理失败: {exc}")
                for sample in batch:
                    result = dict(sample)
                    result.update({
                        "pred_full": "",
                        "pred_text": "",
                        "pred_dialect": "",
                        "error": str(exc),
                    })
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                continue

            for sample, pred_full in zip(batch, pred_full_list):
                pred = split_asr_content(pred_full)
                result = dict(sample)
                result.update({
                    "pred_full": pred_full.strip(),
                    "pred_text": pred["text"],
                    "pred_dialect": pred["dialect"],
                    "error": "",
                })
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1

                print(f"[{done}/{len(samples)}] {sample['utt_id']}")
                print(f"  ref:  {sample['ref_full']}")
                print(f"  pred: {pred_full.strip()}")
            f_out.flush()

    print(f"推理完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
