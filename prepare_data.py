"""prepare_data.py — 构造 2 万条 SFT 数据集 + GRPO 题源 + 评测集

对应《Qwen3-4B数学推理后训练方案(预算版)》数据准备部分：
  - SFT 两种配比（--preset general=数学12% / math=数学45%）
  - 数学来源：GSM8K train / MATH train / OpenMathInstruct-2 子集
  - 通用来源：BELLE；代码：CodeAlpaca-20k；安全：Beavertails(safe 子集)
  - GRPO 题源：GSM8K train（答案可规则验证）
  - 评测集：GSM8K test / MATH-500（红线：test 集绝不能进训练数据）

用法：
  python prepare_data.py --task sft --preset general --out data/sft_general.jsonl
  python prepare_data.py --task sft --preset math    --out data/sft_math.jsonl
  python prepare_data.py --task prompts --out data/gsm8k_train_prompts.jsonl
  python prepare_data.py --task eval --out-dir data/

输出格式（SFT，与 trl SFTTrainer messages 格式对齐）：
  {"messages": [{"role":"system","content":...},
                {"role":"user","content":...},
                {"role":"assistant","content":...}],
   "category": "math|general|code|safety", "source": "gsm8k|..."}
"""

import argparse
import json
import os
import random

from datasets import load_dataset

SYSTEM_PROMPT = "You are Qwen, a helpful and harmless AI assistant."

# 数学题统一提问模板（注意：与 evaluate_math.py / train_grpo.py 保持一致）
MATH_PROMPT_TEMPLATE = (
    "请解答下面的题目，写出完整推理过程，并将最终答案放在 \\boxed{} 中。\n\n{problem}"
)

# 两种配比（条数），见方案「两种配比」表
PRESETS = {
    "general": {"math": 2400, "general": 14000, "code": 2000, "safety": 1600},
    "math": {"math": 9000, "general": 7000, "code": 2400, "safety": 1600},
}

# 数学池内部来源占比：GSM8K 为主（简单可验证），MATH 补难度，OMI-2 补量
MATH_SOURCE_SPLIT = {"gsm8k": 0.45, "math": 0.33, "openmath": 0.22}

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


def gsm8k_final_answer(answer_field: str) -> str:
    """GSM8K answer 字段形如 '... #### 18'，取 #### 后的数字。"""
    return answer_field.split("####")[-1].strip().replace(",", "")


def msg(user: str, assistant: str, category: str, source: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "source": source,
    }


# ---------- 各来源加载器（streaming，按需取量，省带宽） ----------

def load_gsm8k_sft(n: int, seed: int):
    """GSM8K train -> SFT：推理过程 + \\boxed{最终答案}。"""
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    rows = []
    for row in ds:
        ans = gsm8k_final_answer(row["answer"])
        rationale = row["answer"].split("####")[0].strip()
        rows.append(msg(
            MATH_PROMPT_TEMPLATE.format(problem=row["question"]),
            f"{rationale}\n\n答案是 \\boxed{{{ans}}}。",
            "math", "gsm8k",
        ))
        if len(rows) >= n:
            break
    return rows


def load_math_sft(n: int, seed: int):
    """lighteval/MATH train 各 subject -> SFT（solution 本身含 \\boxed）。"""
    rows, per_cfg = [], max(1, n // len(MATH_CONFIGS) + 1)
    for cfg in MATH_CONFIGS:
        ds = load_dataset("lighteval/MATH", cfg, split="train", streaming=True)
        cnt = 0
        for row in ds:
            if "\\boxed" not in row["solution"]:  # 质量铁律：必须有可验证答案
                continue
            rows.append(msg(
                MATH_PROMPT_TEMPLATE.format(problem=row["problem"]),
                row["solution"].strip(),
                "math", "math",
            ))
            cnt += 1
            if cnt >= per_cfg or len(rows) >= n:
                break
        if len(rows) >= n:
            break
    return rows[:n]


def load_openmath_sft(n: int, seed: int):
    """OpenMathInstruct-2 子集：只保留含 \\boxed 最终答案的样本。"""
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    rows = []
    for row in ds:
        problem = row.get("problem") or row.get("question")
        solution = row.get("generated_solution") or row.get("solution") or ""
        if not problem or "\\boxed" not in solution:
            continue
        rows.append(msg(
            MATH_PROMPT_TEMPLATE.format(problem=problem),
            solution.strip(),
            "math", "openmath",
        ))
        if len(rows) >= n:
            break
    return rows


def load_belle_general(n: int, seed: int):
    ds = load_dataset("BELLE-2/Belle_chat_random_2.8M", split="train", streaming=True)
    rows = []
    for row in ds:
        if not row.get("instruction") or not row.get("output"):
            continue
        rows.append(msg(row["instruction"].strip(), row["output"].strip(),
                        "general", "belle"))
        if len(rows) >= n:
            break
    return rows


def load_code(n: int, seed: int):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    rows = []
    for row in ds:
        user = row["instruction"].strip()
        if row.get("input"):
            user += "\n\n" + row["input"].strip()
        if user and row.get("output"):
            rows.append(msg(user, row["output"].strip(), "code", "codealpaca"))
        if len(rows) >= n:
            break
    return rows[:n]


def load_safety(n: int, seed: int):
    """Beavertails 中仅取被标注为安全回复的样本（ refusal / 正常安全作答）。"""
    ds = load_dataset("PKU-Alignment/beavertails", split="train", streaming=True)
    rows = []
    for row in ds:
        if not row.get("is_safe"):
            continue
        rows.append(msg(row["prompt"].strip(), row["response"].strip(),
                        "safety", "beavertails"))
        if len(rows) >= n:
            break
    return rows


def build_sft(preset: str, seed: int) -> list:
    quota = dict(PRESETS[preset])
    math_quota = quota.pop("math")
    g = MATH_SOURCE_SPLIT
    plan = {
        "gsm8k": round(math_quota * g["gsm8k"]),
        "math": round(math_quota * g["math"]),
        "openmath": math_quota - round(math_quota * g["gsm8k"]) - round(math_quota * g["math"]),
    }
    loaders = {
        "gsm8k": (load_gsm8k_sft, plan["gsm8k"]),
        "math": (load_math_sft, plan["math"]),
        "openmath": (load_openmath_sft, plan["openmath"]),
        "general": (load_belle_general, quota["general"]),
        "code": (load_code, quota["code"]),
        "safety": (load_safety, quota["safety"]),
    }
    samples = []
    for name, (fn, n) in loaders.items():
        print(f"[prepare] loading {name}: target {n} ...")
        got = fn(n, seed)
        print(f"[prepare] {name}: got {len(got)}")
        samples.extend(got)
    random.Random(seed).shuffle(samples)
    return samples


def build_prompts() -> list:
    """GRPO 题源：GSM8K train 全量 7473 题，答案可规则验证。"""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [{
        "problem": row["question"],
        "answer": gsm8k_final_answer(row["answer"]),
        "prompt": MATH_PROMPT_TEMPLATE.format(problem=row["question"]),
    } for row in ds]


def build_eval(out_dir: str):
    """评测集（test split，红线：不得进入任何训练数据）。"""
    os.makedirs(out_dir, exist_ok=True)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    with open(os.path.join(out_dir, "gsm8k_test.jsonl"), "w", encoding="utf-8") as f:
        for row in gsm:
            f.write(json.dumps({
                "problem": row["question"],
                "answer": gsm8k_final_answer(row["answer"]),
            }, ensure_ascii=False) + "\n")
    print(f"[prepare] gsm8k_test.jsonl: {len(gsm)} samples")

    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    with open(os.path.join(out_dir, "math500_test.jsonl"), "w", encoding="utf-8") as f:
        for row in m500:
            f.write(json.dumps({
                "problem": row["problem"],
                "answer": row["answer"],
                "subject": row.get("subject", ""),
            }, ensure_ascii=False) + "\n")
    print(f"[prepare] math500_test.jsonl: {len(m500)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sft", "prompts", "eval"], required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="general")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径（sft/prompts）")
    ap.add_argument("--out-dir", default="data", help="eval 任务输出目录")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task == "sft":
        samples = build_sft(args.preset, args.seed)
        out = args.out or f"data/sft_{'general' if args.preset == 'general' else 'math'}.jsonl"
    elif args.task == "prompts":
        samples = build_prompts()
        out = args.out or "data/gsm8k_train_prompts.jsonl"
    else:
        build_eval(args.out_dir)
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {len(samples)} samples -> {out}")
    if args.task == "sft":
        from collections import Counter
        print("[prepare] category distribution:",
              dict(Counter(s["category"] for s in samples)))


if __name__ == "__main__":
    main()
