# LLM From Scratch

This project is different from the attention mechanism repo on my GitHub. That one focused on model architecture components. This one is the full thing: building a complete LLM in raw PyTorch, from raw text to a trained model that generates text.



## Roadmap

- [x] Data loading and tokenization
- [ ] Vocabulary and token IDs (special tokens, BPE)
- [ ] Data sampling with sliding window
- [ ] Token + positional embeddings
- [ ] Causal multi-head attention
- [ ] Full GPT architecture
- [ ] Pretraining loop
- [ ] Fine-tuning
- [ ] Alignment (DPO/RLHF, beyond the book)

## Current stage: tokenization

Downloads "The Verdict" (~20k chars) and splits it into ~4,690 tokens.

```
python tokenize.py
```

Why regex instead of `str.split()`: split on whitespace alone leaves punctuation stuck to words, so "surprise," and "surprise" become two different tokens and the vocabulary doubles for nothing. The regex cuts punctuation into its own tokens. The capture group keeps the punctuation instead of throwing it away, since it carries meaning the model should see.

Trade-off: whitespace gets stripped, so the original text can't be rebuilt exactly. Fine for training. Also word-level tokenization means every word form is its own token — BPE fixes that in the next stage.

## Why raw PyTorch

No Hugging Face abstractions. The point is understanding, not delivery speed. The model will be weak — that's fine, the model is the byproduct.

## Notes

- Variable names are verbose on purpose (`embedding_dim`, not `d_model`)
- Windows + PowerShell, venv via `uv`, CPU is enough until pretraining