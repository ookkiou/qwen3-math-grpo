# Qwen3-4B 数学推理后训练：SFT 数据配比 + GRPO 规则奖励

在 Qwen3-4B-Base 上通过 **SFT 数据配比 + GRPO 规则奖励** 验证后训练对数学推理的影响，产出「基座 → +SFT-通用 → +SFT-重数学 → +GRPO」四组受控对比数据（GSM8K / MATH-500 / MMLU）。

* **预算**：≤ 100 元（单卡 24G，预计 \~65 元）

* **技术栈**：PyTorch, transformers, PEFT(QLoRA), TRL(GRPOTrainer), vLLM, sympy 规则奖励

  <br />

## 实验设计（G0–G3）

每组只改一个变量，隔离「数学 SFT 配比」和「GRPO」各自的贡献：

| 组              | 配置                         | 验证假设            |
| -------------- | -------------------------- | --------------- |
| **G0 基座**      | Qwen3-4B-Base，无后训练         | 基线              |
| **G1 SFT-通用**  | 2 万条 SFT，数学仅 12%           | 通用 SFT 不提升推理    |
| **G2 SFT-重数学** | 2 万条 SFT，数学 45%            | 数学 SFT 配比是主因    |
| **G3 G2+GRPO** | G2 checkpoint + GRPO 500 步 | RL 在 SFT 基础上再提升 |

对比读法：G1 vs G0 证明"会聊天 ≠ 会解题"；G2 vs G1 证明数学 CoT 配比是跃升主因；G3 vs G2 证明 GRPO 增益。

## 仓库结构

```
├── prepare_data.py                # 数据准备：SFT 两种配比 / GRPO 题源 / 评测集
├── train_grpo.py                  # GRPO 训练（trl GRPOTrainer + 纯规则奖励）
├── evaluate_math.py               # GSM8K / MATH-500 zero-shot CoT 评测
├── data/                          # 生成的数据集（不入库）
├── out/                           # 训练输出 checkpoint（不入库）
└── results/                       # 评测结果（不入库）
```

> SFT 训练复用外部项目的 `train_sft.py`（QLoRA r=64，覆盖 q/k/v/o + gate/up/down），本项目只换数据。

## 环境

```bash
pip install "transformers>=4.42" "peft>=0.11" "trl>=0.9" "bitsandbytes>=0.43" \
    accelerate datasets vllm sympy lm_eval
```

## 使用流程

### 1. 数据准备（prepare\_data.py）

```bash
# SFT 两种配比
python prepare_data.py --task sft --preset general --out data/sft_general.jsonl   # G1：数学 12%
python prepare_data.py --task sft --preset math    --out data/sft_math.jsonl      # G2：数学 45%

# GRPO 题源（GSM8K train 全量，答案可规则验证）
python prepare_data.py --task prompts --out data/gsm8k_train_prompts.jsonl

# 评测集（GSM8K test / MATH-500）
python prepare_data.py --task eval --out-dir data/
```

数据来源（全部开源）：GSM8K、lighteval/MATH、OpenMathInstruct-2（数学）；BELLE（通用）；CodeAlpaca-20k（代码）；Beavertails safe 子集（安全）。

> ⚠️ 红线：GSM8K / MATH 的 **test 集绝不能出现在 SFT 或 GRPO 数据里**，训练与 GRPO 题源只用 train 集。

### 2. SFT 训练 G1 / G2

```bash
python train_sft.py \
    --model Qwen/Qwen3-4B \
    --data data/sft_general.jsonl \
    --output_dir out/g1_sft_general \
    --epochs 2 --lr 2e-4 --r 64 --alpha 128 --max-len 2048

python train_sft.py \
    --model Qwen/Qwen3-4B \
    --data data/sft_math.jsonl \
    --output_dir out/g2_sft_math \
    --epochs 2 --lr 2e-4 --r 64 --alpha 128 --max-len 2048
```

### 3. GRPO 训练 G3（train\_grpo.py）

在 G2 checkpoint 上继续训练，奖励纯规则（答案正确 +1，重复退化 -0.2，否则 0），零 API 成本。trl GRPOTrainer **在线采样**（每题实时生成 G=8 条回答），无需 vLLM 预采样：

```bash
python train_grpo.py \
    --model Qwen/Qwen3-4B \
    --sft-adapter out/g2_sft_math \
    --prompts data/gsm8k_train_prompts.jsonl \
    --output_dir out/g3_grpo \
    --steps 500 --lr 1e-6 --beta 0.01 \
    --group-size 8 --temperature 0.9 --max-new-tokens 1024
```

> max-new-tokens 与评测同口径（1024）；24G 显存不够时降为 512，但必须同步评测侧，避免训评截断不一致。4bit QLoRA 下在线生成较慢，500 步预留 4–6h。

监控指标：`reward/mean` 应稳步上升；`kl_divergence` 保持在合理范围（beta 控制，防分布崩塌）。

### 4. 评测（evaluate\_math.py）

zero-shot CoT、贪心解码（pass\@1 可复现），判定与 GRPO 奖励函数同一口径（sympy 数值等价）：

```bash
# G0 基座：base 无 chat template，裸 prompt 评测
python evaluate_math.py --model Qwen/Qwen3-4B \
    --benchmark gsm8k --out results/g0_gsm8k.json
python evaluate_math.py --model Qwen/Qwen3-4B \
    --benchmark math500 --out results/g0_math500.json

# G1–G3：先合并 adapter，再评测；SFT/GRPO 以 chat 格式训练，必须加 --chat-template 保持训评一致
python evaluate_math.py --model merged/g2 \
    --benchmark gsm8k --chat-template --out results/g2_gsm8k.json
python evaluate_math.py --model merged/g3 \
    --eval data/gsm8k_test.jsonl --benchmark gsm8k --chat-template --out results/g3_gsm8k.json
```

### 5. MMLU 评测（lm-eval-harness）

四档统一用 lm_eval 的 5-shot log-likelihood 判定（选择题判分与生成格式无关，口径统一即可）：

```bash
# G0 基座（其余三档换 pretrained=merged/g1、merged/g2、merged/g3）
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-4B,dtype=bfloat16 \
    --tasks mmlu --num_fewshot 5 --batch_size 8 \
    --output_path results/g0_mmlu
```

> 关注 G2/G3 相对 G1 的 MMLU 变化，验证"重数学配比是否伤通用能力"（预期持平或略降）。

## 预期结果（实测替换）

| 模型          | GSM8K | MATH-500 | MMLU |
| ----------- | ----- | -------- | ---- |
| G0 基座       | \~45  | \~28     | \~62 |
| G1 +SFT 通用  | \~54  | \~33     | \~63 |
| G2 +SFT 重数学 | \~66  | \~43     | \~60 |
| G3 +GRPO    | \~70  | \~47     | \~60 |

## 关键设计点

* **训评同口径**：三个脚本共用同一套提问模板、答案提取（`\boxed{}` 支持嵌套 → `答案是/为` → 末尾数字）与 sympy 等价判定（浮点快速路径 + 超时保护），避免训练奖励与评测判定分裂；SFT/GRPO 模型评测时通过 `--chat-template` 保证输入格式与训练一致

* **纯规则奖励**：不引入过程奖励模型（PRM），避免 RM 偏差放大；附带 n-gram 重复检测防 reward hacking

* **QLoRA + GRPO**：无 critic 模型，4bit QLoRA 单卡 24G 可跑；reference model 由 PEFT 禁用 adapter 得到，零额外显存

* **数据质量铁律**：数学样本必须带可验证的最终答案，CoT 过程需真实推导

## License

仅用于学习与研究。
