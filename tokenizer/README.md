# Tokenizer Experiments

This folder focuses on tokenization fundamentals and vocabulary construction. The notebook is self-contained and builds a tokenizer from scratch to make the preprocessing pipeline explicit and inspectable.

## What Is Implemented
- A Byte Pair Encoding (BPE) tokenizer trained from raw text.
- Basic preprocessing, token counting, and merge rule generation.
- Encoding and decoding routines to verify reversibility.

## Notebooks
- `bpe.ipynb`: trains BPE on the bundled `input.txt` corpus, applies merges, and evaluates the resulting vocabulary with sample encodes/decodes.
