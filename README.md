# Arabic Vocabulary Expansion for Qwen3

University research project on **initializing new Arabic token embeddings** for `Qwen3-0.6B-Base`.

The project studies vocabulary expansion with a custom Arabic tokenizer that adds roughly **47k tokens**. The main question is how to place those new tokens into the model's existing embedding space before continued pretraining.

## What I worked on

- classified newly added Arabic tokens with a local `Qwen2.5-32B-Instruct` model and built label-based training presets;
- adapted **Token Distillation** to Qwen3 and a custom Arabic tokenizer;
- added staged training and support for distillation against multiple target layers;
- compared subtoken-mean, FOCUS, Token Distillation, and **FOCUS + Token Distillation**;
- built a deterministic held-out evaluation split from `FineWeb2-HQ / arb_Arab`;
- ran ablations over target layers, token classes, snippet counts, corpus size, loss variants, and learning rate.

## Result

On the project-specific held-out split, the best **FOCUS + Token Distillation** configuration improved the FOCUS-only baseline:

| | FOCUS | FOCUS + TD |
|---|---:|---:|
| LM loss | 7.4350 | **7.1702** |
| Perplexity | 1694.29 | **1300.14** |

That is about a **23.3% relative reduction in perplexity**. These numbers are specific to this evaluation setup and are not intended as a general Qwen3 benchmark.

## Repository layout

- `run_staged_from_labels.py` — main staged distillation pipeline;
- `classify_arabic_tokens.py` — LLM-based token classification;
- `make_eval_dataset_fineweb2.py` — deterministic evaluation split construction;
- `demo_embs_init.ipynb` — embedding-initialization experiments;
- `token_distillation/` — project-modified copy of the official Token Distillation implementation.

The custom Arabic tokenizer and FOCUS-initialized embedding artifacts used in the experiments are not included in this repository.

## Setup

The project uses Python 3.11+ and `uv`:

```bash
uv sync
```

Most experiment settings are exposed as CLI arguments in `run_staged_from_labels.py`.

## Upstream work and license

The Token Distillation code in `token_distillation/` is derived from the official implementation by **Konstantin Dobler, Desmond Elliott, and Gerard de Melo**:

- paper: [Token Distillation: Attention-aware Input Embeddings for New Tokens](https://arxiv.org/abs/2505.20133) (ICLR 2026; earlier arXiv versions used the name *AweDist*);
- official code: [konstantinjdobler/token-distillation](https://github.com/konstantinjdobler/token-distillation).

The upstream implementation is distributed under the **MIT License**. Its copyright and license notice are preserved in `token_distillation/LICENSE`. The surrounding experiment code in this repository was written/adapted for the Arabic vocabulary-expansion project.
