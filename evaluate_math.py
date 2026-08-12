"""evaluate_math.py — GSM8K / MATH-500 zero-shot CoT pass@1 评测

对应《Qwen3-4B数学推理后训练方案(预算版)》评测协议：
  - zero-shot，提示模型分步推理并将答案放入 \\boxed{}
  - 贪心解码（temperature=0）保证 pass@1 可复现
  - 判定：sympy 数值/表达式等价，与 GRPO 奖励函数同一口径

用法（与方案 bash 示例对齐）：
  # G0 基座：base 无 chat template，裸 prompt 评测
  python evaluate_math.py --mode base --model Qwen/Qwen3-4B \
      --benchmark gsm8k --out results/g0_gsm8k.json
  # G1–G3：SFT/GRPO 以 chat 格式训练，必须加 --chat-template 保持训评一致
  python evaluate_math.py --model merged/g2 --benchmark math500 \
      --chat-template --out results/g2_math500.json
  python evaluate_math.py --model merged/g3 --eval data/gsm8k_test.jsonl \
      --benchmark gsm8k --chat-template --out results/g3_gsm8k.json
"""

import argparse
import json
import os
import re

import sympy

# 与 prepare_data.py / train_grpo.py 保持一致的提问模板
MATH_PROMPT_TEMPLATE = (
    "请解答下面的题目，写出完整推理过程，并将最终答案放在 \\boxed{} 中。\n\n{problem}"
)

# 与 train_grpo.py 同款兜底 chat template（base 模型 tokenizer 无模板时用）
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] + '\\n' + m['content'] + '\\n' }}"
    "{% endfor %}"
)


# ---------- 答案提取与等价判定（与 train_grpo.py 同一套规则） ----------

# \boxed 提取支持一层嵌套大括号（如 \boxed{\frac{1}{2}}）
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_answer(text: str) -> str:
    """优先 \\boxed{}，其次'答案是/为'，最后取末尾数字。"""
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = re.search(r"答案[是为]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"[-\d.,/]+(?=\s*$)", text.strip())
    if m:
        return m.group(0).strip()
    return text.strip().split("\n")[-1][:50]


def normalize_math(s: str) -> str:
    """清洗 LaTeX/货币符号，便于 sympy 解析。"""
    s = s.strip().strip("$").strip()
    s = s.replace(",", "").replace("%", "").replace(" ", "")
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace("\\sqrt", "sqrt").replace("\\pi", "pi")
    return s


def is_equivalent(pred: str, truth: str) -> bool:
    """数值等价判定：字符串/浮点快速路径优先，sympy simplify 带超时防卡死。"""
    p, t = normalize_math(pred), normalize_math(truth)
    if p == t:
        return True
    try:  # 纯数字快速路径，避免走 sympy
        return abs(float(p) - float(t)) < 1e-6
    except (ValueError, OverflowError):
        pass
    try:
        return _sympy_equiv(p, t)
    except Exception:
        return False


def _sympy_equiv(p: str, t: str, timeout: float = 5.0) -> bool:
    """sympy.simplify 对复杂表达式可能耗时很久，SIGALRM 超时保护。"""
    expr = sympy.sympify(p, evaluate=True) - sympy.sympify(t, evaluate=True)
    try:
        import signal

        def _handler(signum, frame):
            raise TimeoutError("sympy.simplify timeout")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return bool(sympy.simplify(expr) == 0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except (AttributeError, ValueError):  # 非 Unix 或非主线程，直接调用
        return bool(sympy.simplify(expr) == 0)


# ---------- 数据加载 ----------

def load_eval_data(args) -> list:
    if args.eval:  # 本地 jsonl（{"problem":..., "answer":...}）
        with open(args.eval, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if args.benchmark == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for r in ds]
    if args.benchmark == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    raise ValueError("需要 --benchmark 或 --eval 之一")


# ---------- prompt 格式化 ----------

def apply_chat_template(prompts: list, model_path: str) -> list:
    """SFT/GRPO 模型以 chat 格式训练，评测必须套同一模板，否则分数失真。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_CHAT_TEMPLATE
        print("[eval] tokenizer 无 chat template，使用兜底模板")
    return [tok.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]


# ---------- 推理后端 ----------

def generate_vllm(prompts: list, args) -> list:
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs]


def generate_hf(prompts: list, args) -> list:
    """无 vLLM 时的兜底后端，速度慢，仅用于小规模调试。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    out = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j, ids in enumerate(gen):
            out.append(tok.decode(ids[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
    return out


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型路径或 HF id")
    ap.add_argument("--mode", default="base", help="兼容方案命令，base/merged 同口径")
    ap.add_argument("--benchmark", choices=["gsm8k", "math500"])
    ap.add_argument("--eval", default=None, help="本地评测 jsonl（优先于 --benchmark 内置集）")
    ap.add_argument("--chat-template", action="store_true",
                    help="用 chat template 包装 prompt（评测 SFT/GRPO 模型必须开启）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=4, help="hf 后端批大小")
    args = ap.parse_args()

    data = load_eval_data(args)
    prompts = [MATH_PROMPT_TEMPLATE.format(problem=r["problem"]) for r in data]
    if args.chat_template:  # G1–G3 训评格式一致性的关键开关
        prompts = apply_chat_template(prompts, args.model)
    print(f"[eval] {len(data)} problems from "
          f"{args.eval or args.benchmark}, backend={args.backend}, "
          f"chat_template={args.chat_template}")

    responses = (generate_vllm(prompts, args) if args.backend == "vllm"
                 else generate_hf(prompts, args))

    details, correct = [], 0
    for row, resp in zip(data, responses):
        pred = extract_answer(resp)
        ok = is_equivalent(pred, row["answer"])
        correct += int(ok)
        details.append({"problem": row["problem"], "truth": row["answer"],
                        "pred": pred, "correct": ok,
                        "response": resp[-500:]})  # 截断保存，控制体积
    acc = correct / max(len(data), 1)
    result = {"model": args.model, "benchmark": args.eval or args.benchmark,
              "accuracy": round(acc, 4), "correct": correct, "total": len(data),
              "details": details}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[eval] {args.eval or args.benchmark}: acc={acc:.4f} "
          f"({correct}/{len(data)}) -> {args.out}")


if __name__ == "__main__":
    main()
