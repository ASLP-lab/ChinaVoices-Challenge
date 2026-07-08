#!/usr/bin/env bash
set -euo pipefail


# cd path/to/qwen3asr/eval

# bash eval/eval_jsonl_with_wer_tools.sh \
#   --pred_jsonl outputs/pred.jsonl \
#   --output_dir outputs/wer_eval \
#   --by_dialect 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pred_jsonl=""
output_dir="${SCRIPT_DIR}/results/jsonl_wer_eval"
apply_t2s=1
by_dialect=1

usage() {
    cat <<USAGE
Usage:
  bash eval_jsonl_with_wer_tools.sh \
    --pred_jsonl PRED_JSONL \
    --output_dir OUTPUT_DIR [options]

Required:
  --pred_jsonl PATH       JSONL with utt_id/ref_text/pred_text or ref_full/pred_full

Options:
  --output_dir DIR        output directory. Default: ${output_dir}
  --apply_t2s 0|1         convert Traditional Chinese to Simplified before WER. Default: ${apply_t2s}
  --by_dialect 0|1       also compute one result.wer for each non-empty ref_dialect. Default: ${by_dialect}

Outputs:
  OUTPUT_DIR/ref.txt
  OUTPUT_DIR/infer.txt
  OUTPUT_DIR/ref_t2s.txt        if --apply_t2s 1
  OUTPUT_DIR/infer_t2s.txt      if --apply_t2s 1
  OUTPUT_DIR/result_clean.txt
  OUTPUT_DIR/result.wer
  OUTPUT_DIR/dialect_accuracy.txt
  OUTPUT_DIR/by_dialect/<ref_dialect>/result.wer  if --by_dialect 1
  OUTPUT_DIR/by_dialect_summary.txt              if --by_dialect 1
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pred_jsonl) pred_jsonl="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --apply_t2s) apply_t2s="$2"; shift 2 ;;
        --by_dialect) by_dialect="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${pred_jsonl}" ]]; then
    echo "ERROR: --pred_jsonl is required." >&2
    usage
    exit 1
fi

if [[ ! -f "${pred_jsonl}" ]]; then
    echo "ERROR: pred_jsonl not found: ${pred_jsonl}" >&2
    exit 1
fi

mkdir -p "${output_dir}"
ref_txt="${output_dir}/ref.txt"
infer_txt="${output_dir}/infer.txt"
ref_t2s_txt="${output_dir}/ref_t2s.txt"
infer_t2s_txt="${output_dir}/infer_t2s.txt"
result_clean_txt="${output_dir}/result_clean.txt"
wer_txt="${output_dir}/result.wer"
dialect_acc_txt="${output_dir}/dialect_accuracy.txt"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

convert_t2s() {
    local input_file="$1"
    local output_file="$2"
    python - "${input_file}" "${output_file}" <<'PY_T2S'
import sys
from opencc import OpenCC

input_file, output_file = sys.argv[1], sys.argv[2]
cc = OpenCC("t2s")
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(cc.convert(line))
PY_T2S
}

compute_dialect_accuracy() {
    local input_jsonl="$1"
    local output_file="$2"
    python - "${input_jsonl}" "${output_file}" <<'PY_DIALECT_ACC'
import json
import re
import sys
from collections import Counter, defaultdict

input_jsonl, output_file = sys.argv[1], sys.argv[2]
asr_marker = "<asr_text>"
dialect_re = re.compile(r"language\s+Chinese\s+([^\s<]+)", re.IGNORECASE)


def extract_dialect(full):
    full = (full or "").strip()
    prefix = full.split(asr_marker, 1)[0] if asr_marker in full else full
    match = dialect_re.search(prefix)
    return match.group(1).strip() if match else ""


def norm_label(value):
    return str(value or "").strip()

rows = []
total_lines = 0
skipped_error = 0
missing_ref = 0
for line_no, line in enumerate(open(input_jsonl, "r", encoding="utf-8"), 1):
    line = line.strip()
    if not line:
        continue
    total_lines += 1
    obj = json.loads(line)
    if obj.get("error"):
        skipped_error += 1
        continue

    ref = norm_label(obj.get("ref_dialect"))
    pred = norm_label(obj.get("pred_dialect"))
    if not ref:
        ref = extract_dialect(obj.get("ref_full", ""))
    if not pred:
        pred = extract_dialect(obj.get("pred_full", ""))
    if not ref:
        missing_ref += 1
        continue
    rows.append((ref, pred or "<empty>"))

ref_counts = Counter(ref for ref, _ in rows)
pred_counts = Counter(pred for _, pred in rows)
correct_by_ref = Counter(ref for ref, pred in rows if ref == pred)
confusion = defaultdict(Counter)
for ref, pred in rows:
    confusion[ref][pred] += 1

correct = sum(1 for ref, pred in rows if ref == pred)
evaluable = len(rows)
acc = correct / evaluable if evaluable else 0.0
labels = sorted(set(ref_counts) | set(pred_counts))

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Dialect recognition accuracy\n")
    f.write("============================\n")
    f.write(f"input_jsonl: {input_jsonl}\n")
    f.write(f"total_lines: {total_lines}\n")
    f.write(f"skipped_error: {skipped_error}\n")
    f.write(f"missing_ref_dialect: {missing_ref}\n")
    f.write(f"evaluable: {evaluable}\n")
    f.write(f"correct: {correct}\n")
    f.write(f"accuracy: {acc:.6f} ({acc * 100:.2f}%)\n")

    f.write("\nAccuracy by ref_dialect\n")
    f.write(f"{'ref_dialect':<20} {'samples':>10} {'correct':>10} {'accuracy':>12}\n")
    f.write("-" * 56 + "\n")
    for ref in sorted(ref_counts):
        n = ref_counts[ref]
        c = correct_by_ref[ref]
        f.write(f"{ref:<20} {n:>10} {c:>10} {c / n * 100:>11.2f}%\n")

    f.write("\nPredicted dialect distribution\n")
    f.write(f"{'pred_dialect':<20} {'count':>10} {'ratio':>12}\n")
    f.write("-" * 44 + "\n")
    for pred, n in pred_counts.most_common():
        ratio = n / evaluable * 100 if evaluable else 0.0
        f.write(f"{pred:<20} {n:>10} {ratio:>11.2f}%\n")

    f.write("\nConfusion matrix (rows=ref, cols=pred)\n")
    f.write("ref\\pred" + "".join(f"\t{label}" for label in labels) + "\n")
    for ref in sorted(ref_counts):
        f.write(ref + "".join(f"\t{confusion[ref][label]}" for label in labels) + "\n")
PY_DIALECT_ACC
}

echo "[1/5] Convert JSONL to text files"
python "${SCRIPT_DIR}/jsonl_to_text_for_wer.py" \
    --pred_jsonl "${pred_jsonl}" \
    --ref_out "${ref_txt}" \
    --pred_out "${infer_txt}"

clean_input="${infer_txt}"
wer_ref="${ref_txt}"
if [[ "${apply_t2s}" == "1" ]]; then
    echo "[2/5] Traditional-to-Simplified conversion"
    convert_t2s "${ref_txt}" "${ref_t2s_txt}"
    convert_t2s "${clean_input}" "${infer_t2s_txt}"
    clean_input="${infer_t2s_txt}"
    wer_ref="${ref_t2s_txt}"
else
    echo "[2/5] Traditional-to-Simplified conversion skipped"
fi

echo "[3/5] Clean prediction"
python "${SCRIPT_DIR}/tools/clean_data.py" \
    "${clean_input}" \
    "${result_clean_txt}" \
    --transcript "${wer_ref}"

echo "[4/5] Compute WER/CER"
python "${SCRIPT_DIR}/tools/wer.py" --char=1 --v=1 \
    "${wer_ref}" \
    "${result_clean_txt}" \
    > "${wer_txt}"

echo "[5/5] Compute dialect recognition accuracy"
compute_dialect_accuracy "${pred_jsonl}" "${dialect_acc_txt}"

echo "Done. Outputs:"
echo "  ${ref_txt}"
echo "  ${infer_txt}"
if [[ "${apply_t2s}" == "1" ]]; then
    echo "  ${ref_t2s_txt}"
    echo "  ${infer_t2s_txt}"
fi
echo "  ${result_clean_txt}"
echo "  ${wer_txt}"
echo "  ${dialect_acc_txt}"

if [[ "${by_dialect}" == "1" ]]; then
    echo "[extra] Compute WER/CER by dialect"
    dialect_root="${output_dir}/by_dialect"
    summary_txt="${output_dir}/by_dialect_summary.txt"
    rm -rf "${dialect_root}"
    mkdir -p "${dialect_root}"

    python "${SCRIPT_DIR}/jsonl_to_text_by_dialect_for_wer.py" \
        --pred_jsonl "${pred_jsonl}" \
        --output_dir "${dialect_root}"

    : > "${summary_txt}"
    printf "%-20s %10s %12s\n" "dialect" "samples" "wer" >> "${summary_txt}"
    printf "%s\n" "----------------------------------------------" >> "${summary_txt}"

    if [[ ! -s "${dialect_root}/dialects.txt" ]]; then
        printf "%s\n" "# No non-empty ref_dialect found; by_dialect_summary is grouped only by ref_dialect." >> "${summary_txt}"
    else
        while IFS=$'\t' read -r dialect samples; do
        [[ -z "${dialect}" ]] && continue
        dialect_dir="${dialect_root}/${dialect}"
        dialect_ref="${dialect_dir}/ref.txt"
        dialect_infer="${dialect_dir}/infer.txt"
        dialect_ref_t2s="${dialect_dir}/ref_t2s.txt"
        dialect_infer_t2s="${dialect_dir}/infer_t2s.txt"
        dialect_clean="${dialect_dir}/result_clean.txt"
        dialect_wer="${dialect_dir}/result.wer"

        dialect_clean_input="${dialect_infer}"
        dialect_wer_ref="${dialect_ref}"
        if [[ "${apply_t2s}" == "1" ]]; then
            convert_t2s "${dialect_ref}" "${dialect_ref_t2s}"
            convert_t2s "${dialect_clean_input}" "${dialect_infer_t2s}"
            dialect_clean_input="${dialect_infer_t2s}"
            dialect_wer_ref="${dialect_ref_t2s}"
        fi

        python "${SCRIPT_DIR}/tools/clean_data.py" \
            "${dialect_clean_input}" \
            "${dialect_clean}" \
            --transcript "${dialect_wer_ref}" >/dev/null

        python "${SCRIPT_DIR}/tools/wer.py" --char=1 --v=1 \
            "${dialect_wer_ref}" \
            "${dialect_clean}" \
            > "${dialect_wer}"

        wer_line=$(grep "Overall ->" "${dialect_wer}" | tail -1 || true)
        wer_value=$(echo "${wer_line}" | awk '{print $3}')
        printf "%-20s %10s %11s%%\n" "${dialect}" "${samples}" "${wer_value:-NA}" >> "${summary_txt}"
        done < "${dialect_root}/dialects.txt"
    fi

    echo "  ${summary_txt}"
fi
