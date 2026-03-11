# Large Language Models (LLM)

This repository contains a set of experiments to familiarise myself with LLM. I cover the tokenization, pre-training and post-training stages, and the repo is under active development so notebooks and utilities evolve frequently.

Some used references are left here. 

## Repository Structure

- **Tokenization** (`experiments/tokenizer/`)
  - `bpe.ipynb`: implements a Byte Pair Encoding tokenizer from scratch and tests it on the bundled `input.txt` corpus.

- **Pre-Training** (`experiments/pre-training/`)
  - `gpt_from_scratch/attention.ipynb`: dissects scaled dot-product attention math and code before integrating it into a full decoder block.
  - `gpt_from_scratch/data_processing.ipynb`: prepares the Edu-FineWeb10B shards (`edu_fineweb10B/*.npy`) and the batching pipeline used across later notebooks.
  - `gpt_from_scratch/gpt.ipynb`: assembles the base GPT architecture, optimizer, and evaluation routine.
  - `gpt_from_scratch/gpt-2.ipynb`: scales the previous model up to a GPT-2 class configuration to study depth/width effects.
  - `huggingface/Self_Supervised_Learning.ipynb`: reproduces Hugging Face’s self-supervised training loop using 🤗 Transformers utilities.
  - `scaling_laws/scaling_laws.ipynb`: analyzes compute/parameter/token scaling behavior using cached `data/` binaries and the reference plots in `resources/`.

- **Fine-Tuning** (`experiments/fine-tuning/`)
  - `from_scratch_classification_fine_tuning.ipynb`: adapts a from-scratch pretrained encoder/decoder for downstream text classification.

- **Post-Training & Alignment** (`experiments/post-training/`)
  - `instruction-fine-tuning/instruction_fine_tuning.ipynb`: instruction-tunes a GPT-2 (124M) checkpoint on the `nvidia/Nemotron-Instruction-Following-Chat-v1` corpus, adds chat-specific tokens, and logs quick MT-Bench/HellaSwag evals.
  - `huggingface/sft.ipynb`: runs supervised fine-tuning (SFT) on instruction-style data to align base models with prompts.
  - `huggingface/DPO.ipynb`: explores Direct Preference Optimization as an alignment method without explicit reward models.
  - `huggingface/RLVR.ipynb`: experiments with reinforcement learning variants that use preference models (reward/value) on top of SFT.
  - `reinforcement_learning_verifiable_rewards/rlvr.ipynb`: toy RLVR GRPO implementation on MATH-500 using only training data (no held-out evaluation).


# References

A set of papers and other sources I consult to develop my experiments.

* [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

* [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
* [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

* [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main)

* [Build GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=11s) lecture by Andrej Kaparthy

* [Attention is All you need](https://arxiv.org/abs/1706.03762) paper

* [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) paper

* [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)

* [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

* ['Reinforcement Learning From Human Feedback'](https://rlhfbook.com/) Book

* [HuggingFace Learn](https://huggingface.co/learn)

* Hugging Face - TRL https://github.com/huggingface/trl/tree/main

* Introduction to Reinforcement Learning. https://arxiv.org/pdf/2408.07712

* [Olmo 3](https://arxiv.org/pdf/2512.13961)
