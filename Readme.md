# LLM From Scratch

This project is different from the attention mechanism repo on my GitHub. That one focused on isolated model architecture components — MultiHeadAttention, FeedForward, PositionalEncoding, EncoderLayer — with no data or training around them. This one is the full thing: building a complete LLM in raw PyTorch, from raw text to a trained model that generates text. Components first, full system second.

## Roadmap

- [x] Data loading and tokenization
- [x] Vocabulary and token IDs (special tokens, BPE)
- [x] Data sampling with sliding window
- [x] Token + positional embeddings
- [ ] Causal multi-head attention
- [ ] Full GPT architecture
- [ ] Pretraining loop
- [ ] Fine-tuning
- [ ] Alignment (DPO/RLHF, beyond the book)

## Files

- `corpus.py` — downloads "The Verdict" (~20k chars) once, exposes `raw_text` for every other file
- `tokenizer.py` — regex word-level split, ~4,690 tokens
- `vocabulary.py` — hand-built tokenizer with `<|unk|>` and `<|endoftext|>`, encode/decode both ways
- `bpe_tokenizer.py` — same job with GPT-2's BPE via tiktoken, plus the input → next-token demo
- `dataset.py` — sliding window over token IDs, PyTorch Dataset + DataLoader producing input/target pairs
- `embeddings.py` — token embeddings + learned positional embeddings, output shape `(batch, context, 256)`

## Why two tokenizers

Built the word-level one by hand first to hit its limits directly: vocabulary bloat (every word form is its own token) and unknown words needing `<|unk|>`. BPE solves both by splitting rare words into known pieces, so nothing is ever unknown. The hand-built one stays in the repo as the reason the second one exists.

Trade-off of the regex version: whitespace gets stripped, so the original text can't be rebuilt exactly. Fine for training.

## Why raw PyTorch

No Hugging Face abstractions. The point is understanding, not delivery speed. The model will be weak — that's fine, the model is the byproduct.

## Notes

- Variable names are verbose on purpose (`embedding_dim`, not `d_model`)
- Windows + PowerShell, venv via `uv`, CPU is enough until pretraining