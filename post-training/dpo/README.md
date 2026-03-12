# Direct Preference Optimization (DPO)

This notebook implements DPO from scratch on a small Qwen2.5 model and walks through the full training loop end‑to‑end.

What’s inside
1. **Model setup**
   - Loads `Qwen/Qwen2.5-0.5B-Instruct` with `AutoModelForCausalLM`.
   - Builds a **frozen reference model** as a deep copy of the policy model.
2. **Lightweight LoRA (manual)**
   - Implements a minimal LoRA wrapper (`LoRALayer`, `LinearWithLoRA`).
   - Replaces the last `n` transformer blocks’ linear layers with LoRA‑augmented versions.
3. **Custom DPO training loop**
   - Explicit computation of policy vs reference log‑probs.
   - Implements the DPO objective using `logsigmoid` over log‑ratio differences.
   - Logs metrics like `loss`, `logp_chosen`, `logp_rejected`, and `reward_margin`.
4. **Custom collation and masking**
   - Pads sequences using `pad_sequence`.
   - Builds `attention_mask` (non‑pad) and **response‑only masks** to isolate completion tokens.
   - Supports chosen/rejected pairs in each batch item.
5. **DataLoader + optimizer wiring**
   - Uses `AdamW` with a small learning rate.
   - Manually runs a training loop with periodic validation logging.

Files
- Notebook: `post-training/dpo/dpo.ipynb`
