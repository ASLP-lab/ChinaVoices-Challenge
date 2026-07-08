#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

ASR_MARKER = "<asr_text>"


def split_asr_text(value):
    value = (value or "").strip()
    if ASR_MARKER in value:
        return value.split(ASR_MARKER, 1)[1].strip()
    return value


def one_line_text(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def main():
    parser = argparse.ArgumentParser(description="Convert ASR JSONL result to ref/pred text files for tools/wer.py")
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--ref_out", required=True)
    parser.add_argument("--pred_out", required=True)
    args = parser.parse_args()

    ref_out = Path(args.ref_out)
    pred_out = Path(args.pred_out)
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    skipped = 0
    with open(args.pred_jsonl, "r", encoding="utf-8") as fin, \
            ref_out.open("w", encoding="utf-8") as f_ref, \
            pred_out.open("w", encoding="utf-8") as f_pred:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                skipped += 1
                continue

            utt_id = str(obj.get("utt_id") or obj.get("uttid") or obj.get("id") or line_no)
            ref = obj.get("ref_text")
            pred = obj.get("pred_text")
            if ref is None:
                ref = split_asr_text(obj.get("ref_full", ""))
            if pred is None:
                pred = split_asr_text(obj.get("pred_full", ""))

            f_ref.write(f"{utt_id} {one_line_text(ref)}\n")
            f_pred.write(f"{utt_id} {one_line_text(pred)}\n")
            n += 1

    print(f"converted={n} skipped_error={skipped}")
    print(f"ref_out={ref_out}")
    print(f"pred_out={pred_out}")


if __name__ == "__main__":
    main()
