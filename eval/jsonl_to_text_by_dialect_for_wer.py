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


def safe_name(value):
    value = (value or "unknown").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main():
    parser = argparse.ArgumentParser(description="Split ASR JSONL result into ref/pred text files by ref_dialect")
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = {}
    skipped = 0
    missing_ref_dialect = 0
    try:
        with open(args.pred_jsonl, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    skipped += 1
                    continue

                ref_dialect = str(obj.get("ref_dialect") or "").strip()
                if not ref_dialect:
                    missing_ref_dialect += 1
                    continue

                dialect = safe_name(ref_dialect)
                dialect_dir = output_dir / dialect
                dialect_dir.mkdir(parents=True, exist_ok=True)
                if dialect not in handles:
                    handles[dialect] = (
                        (dialect_dir / "ref.txt").open("w", encoding="utf-8"),
                        (dialect_dir / "infer.txt").open("w", encoding="utf-8"),
                    )
                    counts[dialect] = 0

                utt_id = str(obj.get("utt_id") or obj.get("uttid") or obj.get("id") or line_no)
                ref = obj.get("ref_text")
                pred = obj.get("pred_text")
                if ref is None:
                    ref = split_asr_text(obj.get("ref_full", ""))
                if pred is None:
                    pred = split_asr_text(obj.get("pred_full", ""))

                f_ref, f_pred = handles[dialect]
                f_ref.write(f"{utt_id} {one_line_text(ref)}\n")
                f_pred.write(f"{utt_id} {one_line_text(pred)}\n")
                counts[dialect] += 1
    finally:
        for f_ref, f_pred in handles.values():
            f_ref.close()
            f_pred.close()

    dialects_path = output_dir / "dialects.txt"
    with dialects_path.open("w", encoding="utf-8") as f:
        for dialect in sorted(counts):
            f.write(f"{dialect}\t{counts[dialect]}\n")

    print(f"dialects={len(counts)} skipped_error={skipped} missing_ref_dialect={missing_ref_dialect}")
    print(f"dialects_out={dialects_path}")


if __name__ == "__main__":
    main()
