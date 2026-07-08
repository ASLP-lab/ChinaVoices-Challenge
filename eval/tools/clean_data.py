import re
import argparse
import os


MIXED_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]')


def normalize_text(text):
    """去除标点和特殊符号，保留所有 Unicode 字母、数字和空白字符。"""
    if not isinstance(text, str):
        return text
    text = re.sub(r'[^\w\s]|_', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def mixed_tokenize(text):
    """中英文混合切分：英文数字串作为整体，中文按单字切。"""
    return [match.group(0) for match in MIXED_TOKEN_PATTERN.finditer(text)]


def mixed_tokenize_with_spans(text):
    """返回 [(token, start, end)]，用于按 token 裁剪时保留原文格式。"""
    return [(match.group(0), match.start(), match.end()) for match in MIXED_TOKEN_PATTERN.finditer(text)]


def _find_repeated_tail_by_tokens(tokens, min_pattern_tokens=8, min_repeats=2):
    """
    查找 token 级别的连续重复尾巴。
    返回保留到“第一轮重复块结束”时的 token 下标；若未命中则返回 None。
    """
    n = len(tokens)
    best = None  # (repeated_tokens, repeats, pattern_len, start)

    for start in range(n):
        remaining = n - start
        max_pattern_len = remaining // min_repeats
        if max_pattern_len < min_pattern_tokens:
            continue

        for pattern_len in range(min_pattern_tokens, max_pattern_len + 1):
            pattern = tokens[start:start + pattern_len]
            repeats = 1
            pos = start + pattern_len

            while pos + pattern_len <= n and tokens[pos:pos + pattern_len] == pattern:
                repeats += 1
                pos += pattern_len

            if repeats < min_repeats:
                continue

            repeated_tokens = repeats * pattern_len
            candidate = (repeated_tokens, repeats, pattern_len, -start)
            if best is None or candidate > best[0]:
                best = (candidate, start)

    if best is None:
        return None

    _, start = best
    pattern_len = best[0][2]
    return start + pattern_len


def _find_frequent_phrase_by_tokens(tokens, min_pattern_tokens=12, min_occurrences=3, max_pattern_tokens=40):
    """
    查找全句中高频出现的长短语，不要求连续重复。
    适合处理“重复句式中间夹连接词”的幻觉。
    返回保留到首次出现结束时的 token 下标。
    """
    n = len(tokens)
    if n < min_pattern_tokens * min_occurrences:
        return None

    best = None  # (score, occurrences, pattern_len, start)
    upper = min(max_pattern_tokens, n)

    for pattern_len in range(min_pattern_tokens, upper + 1):
        positions = {}
        for start in range(n - pattern_len + 1):
            pattern = tuple(tokens[start:start + pattern_len])
            positions.setdefault(pattern, []).append(start)

        for starts in positions.values():
            occurrences = len(starts)
            if occurrences < min_occurrences:
                continue

            start = starts[0]
            score = pattern_len * occurrences
            candidate = (score, occurrences, pattern_len, -start)
            if best is None or candidate > best[0]:
                best = (candidate, start)

    if best is None:
        return None

    _, start = best
    pattern_len = best[0][2]
    return start + pattern_len


def _find_repeated_tail_by_chars(text, min_chars=6, min_repeats=3):
    """
    回退用的字符级连续重复检测。
    只接受较长重复，避免因为单字或双字重复而误裁。
    返回保留到“第一轮重复块结束”时的字符下标。
    """
    n = len(text)
    best = None  # (repeated_chars, repeats, length, start)

    for start in range(n):
        remaining = n - start
        max_len = remaining // min_repeats
        if max_len < min_chars:
            continue

        for length in range(min_chars, max_len + 1):
            phrase = text[start:start + length]
            repeats = 1
            pos = start + length

            while pos + length <= n and text[pos:pos + length] == phrase:
                repeats += 1
                pos += length

            if repeats < min_repeats:
                continue

            repeated_chars = repeats * length
            candidate = (repeated_chars, repeats, length, -start)
            if best is None or candidate > best[0]:
                best = (candidate, start)

    if best is None:
        return None

    _, start = best
    length = best[0][2]
    return start + length


def clean_repeated_text(text):
    if not isinstance(text, str):
        return text

    token_spans = mixed_tokenize_with_spans(text)
    tokens = [token for token, _, _ in token_spans]

    cut_token_idx = _find_repeated_tail_by_tokens(tokens)
    if cut_token_idx is not None:
        return text[:token_spans[cut_token_idx - 1][2]].strip()

    cut_phrase_idx = _find_frequent_phrase_by_tokens(tokens)
    if cut_phrase_idx is not None:
        return text[:token_spans[cut_phrase_idx - 1][2]].strip()

    cut_char_idx = _find_repeated_tail_by_chars(text)
    if cut_char_idx is not None:
        return text[:cut_char_idx].strip()

    return text.strip()

def load_transcript(transcript_path):
    """加载抄本文件，返回 {key: txt} 字典"""
    transcript = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line, 1)
            if len(parts) == 2:
                transcript[parts[0]] = parts[1]
            elif len(parts) == 1:
                transcript[parts[0]] = ''
    return transcript


def process_text(input_path, output_path, transcript_path=None, transcript_output_path=None, normalize_transcript=False):
    """
    读取text格式文件（每行格式: key txt），处理txt部分并写入输出文件。
    若提供抄本文件，仅当预测文本长度 > 抄本长度 * 2 时才进行重复清洗。
    normalize_transcript=True 时对抄本也做去标点处理并保存。
    """
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 '{input_path}'")
        return

    transcript = {}
    if transcript_path:
        if not os.path.exists(transcript_path):
            print(f"错误: 找不到抄本文件 '{transcript_path}'")
            return
        transcript = load_transcript(transcript_path)
        if normalize_transcript:
            # 对抄本做 normalize_text
            transcript = {k: normalize_text(v) for k, v in transcript.items()}
            # 保存处理后的抄本
            ref_out = transcript_output_path or (
                os.path.splitext(transcript_path)[0] + '_clean' + os.path.splitext(transcript_path)[1]
            )
            with open(ref_out, 'w', encoding='utf-8') as f:
                for k, v in transcript.items():
                    f.write(f"{k} {v}\n")
            print(f"抄本已保存至: {ref_out}")

    count = 0
    cleaned_entries = []  # [(key, ref_txt, original_txt, cleaned_txt)]
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                continue

            # 分割key和txt（第一个空格分隔）
            parts = re.split(r'\s+', line, 1)
            if len(parts) == 2:
                key, txt = parts
                txt = normalize_text(txt)
                ref_txt = transcript.get(key, '') if transcript else None
                ref_len = len(ref_txt) if ref_txt is not None else None
                if ref_len is None or len(txt) > ref_len * 2:
                    original_txt = txt
                    txt = clean_repeated_text(txt)
                    if txt != original_txt:
                        cleaned_entries.append((key, ref_txt, original_txt, txt))
                fout.write(f"{key} {txt}\n")
            elif len(parts) == 1:
                key = parts[0]
                fout.write(f"{key} \n")
            else:
                print(f"错误: 行格式不正确: {line}")
                fout.write(f"{normalize_text(line)}\n")

            count += 1

    print(f"处理完成！共处理 {count} 条数据，其中 {len(cleaned_entries)} 条执行了重复清洗。")
    print(f"结果已保存至: {output_path}")

    if cleaned_entries:
        print("\n--- 被清洗的条目 ---")
        for key, ref_txt, original_txt, cleaned_txt in cleaned_entries:
            print(f"[{key}]")
            if ref_txt is not None:
                print(f"  抄本: {ref_txt}")
            print(f"  裁剪前: {original_txt}")
            print(f"  裁剪后: {cleaned_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗text格式文件中的重复文本内容")
    parser.add_argument("input", help="输入的text文件路径")
    parser.add_argument("output", help="输出的text文件路径")
    parser.add_argument("--transcript", "-t", default=None, help="抄本text文件路径，提供后仅对预测长度>抄本长度2倍的条目进行清洗")
    parser.add_argument("--transcript-output", default=None, help="处理后的抄本输出路径，默认为抄本文件名加 _clean 后缀")
    parser.add_argument("--normalize-transcript", action="store_true", help="对抄本做去标点处理并保存")
    parser.add_argument("--no-normalize-transcript", action="store_true", help=argparse.SUPPRESS)  # 保留兼容性

    args = parser.parse_args()
    process_text(args.input, args.output, args.transcript, args.transcript_output, args.normalize_transcript)
