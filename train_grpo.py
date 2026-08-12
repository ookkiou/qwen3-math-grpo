"""train_grpo.py — 基于 trl GRPOTrainer 的数学推理 GRPO 训练（纯规则奖励）

对应《Qwen3-4B数学推理后训练方案(预算版)》阶段 2：
  - 在 G2 SFT checkpoint 上继续 GRPO，500 步，group_size=8
  - 奖励纯规则：答案正确 +1，重复退化 -0.2，否则 0（零 API 成本）
  - 题源：GSM8K train（prepare_data.py --task prompts 生成）

用法（与方案 bash 示例对齐）：
  python train_grpo.py \
      --model Qwen/Qwen3-4B --sft-adapter out/g2_sft_math \
      --prompts data/gsm8k_train_prompts.jsonl \
      --output_dir out/g3_grpo \
      --steps 500 --lr 1e-6 --beta 0.01 \
      --group-size 8 --temperature 0.9 --max-new-tokens 1024

说明：trl GRPOTrainer 在线采样（训练时实时生成 G 条回答），无需 vLLM 预采样；
max-new-tokens 与 evaluate_math.py 保持同口径（1024），显存不够再降并同步评测。
"""

import argparse
import json
import re
from collections import Counter  # 重复检测用，按规范置于文件顶部

import sympy
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import GRPOConfig, GRPOTrainer

# 与 prepare_data.py / evaluate_math.py 保持一致的提问模板
MATH_PROMPT_TEMPLATE = (
    "请解答下面的题目，写出完整推理过程，并将最终答案放在 \\boxed{} 中。\n\n{problem}"
)

# Base 模型可能没有 chat template，给一个最简兜底（Qwen3-Instruct 模板则自动沿用）
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] + '\\n' + m['content'] + '\\n' }}"
    "{% endfor %}"
)


# ---------- 规则奖励组件（与 evaluate_math.py 同一口径） ----------

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
    s = s.strip().strip("$").strip()
    s = s.replace(",", "").replace("%", "").replace(" ", "")
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace("\\sqrt", "sqrt").replace("\\pi", "pi")
    return s


def is_equivalent(pred: str, truth: str) -> bool:
    """数值等价判定：字符串/浮点快速路径优先（reward 高频调用，避免 sympy 拖慢训练）。"""
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
    """sympy.simplify 对复杂表达式可能耗时很久，SIGALRM 超时保护防训练卡死。"""
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


def has_repetition(text: str, ngram: int = 4, threshold: int = 3) -> bool:
    """n-gram 重复检测（reward hacking 迹象），Counter 已在文件顶部导入。"""
    words = text.split()
    if len(words) < ngram * threshold:
        return False
    ngrams = [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
    return max(Counter(ngrams).values()) >= threshold


def correctness_reward(completions, answer, **kwargs):
    """GRPO 奖励函数：答案正确 +1，重复退化 -0.2，否则 0。

    completions 兼容两种形态：str（旧版 trl）或
    list[{"role","content"}]（新版 trl 启用 chat template 时）。
    """
    rewards = []
    for comp, gt in zip(completions, answer):
        if isinstance(comp, list):
            text = comp[0]["content"] if comp else ""
        else:
            text = comp
        pred = extract_answer(text)
        if is_equivalent(pred, gt):
            rewards.append(1.0)
        elif has_repetition(text):
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


# ---------- 数据 ----------

def load_prompt_dataset(path: str) -> Dataset:
    """GRPO 数据集需含 prompt（chat 消息列表）与 answer 两列。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "prompt": [{"role": "user",
                            "content": MATH_PROMPT_TEMPLATE.format(
                                problem=r["problem"])}],
                "answer": str(r["answer"]),
            })
    print(f"[grpo] loaded {len(rows)} prompts from {path}")
    return Dataset.from_list(rows)


# ---------- 模型 ----------

def load_model_and_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_CHAT_TEMPLATE
        print("[grpo] tokenizer 无 chat template，使用兜底模板")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    if args.sft_adapter:
        from peft import PeftModel
        # 继续训练 G2 的 QLoRA adapter（is_trainable=True）
        model = PeftModel.from_pretrained(model, args.sft_adapter,
                                          is_trainable=True)
        print(f"[grpo] loaded SFT adapter: {args.sft_adapter}")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen3-4B-Base 路径或 HF id")
    ap.add_argument("--sft-adapter", default=None, help="G2 SFT LoRA 目录")
    ap.add_argument("--prompts", required=True, help="GRPO 题源 jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.01, help="KL 系数")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="与评测同口径；24G 显存不够时降为 512 并同步评测侧")
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="每卡 prompt 数（实际生成数 = batch*group_size）")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args)
    train_ds = load_prompt_dataset(args.prompts)

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        beta=args.beta,                          # KL 约束，防分布崩塌
        num_generations=args.group_size,         # 每题采样 G 条组内相对排序
        temperature=args.temperature,            # 保证组内多样性
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,             # 24G 显存关键开关
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=["tensorboard"],               # reward/mean、kl 监控
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[grpo] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
