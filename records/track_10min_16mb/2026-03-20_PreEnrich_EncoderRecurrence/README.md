## EMA-GPU + Multi-Order N-gram Backoff + Pre-Enrichment + XSA

**val_bpb: 0.2995** (3-seed mean, std 0.0016) | 14.94 MB | 8xH100 SXM, 600s

---

### 3-Seed Results

| Seed | Steps | Sliding BPB | val_bpb | Artifact |
|---|---|---|---|---|
| 1337 | 9,268 | 1.1478 | 0.3001 | 14,942,971 |
| 42 | 9,318 | 1.1468 | 0.2977 | 14,922,769 |
| 3011 | 9,322 | 1.1463 | 0.3008 | 14,939,305 |
| **Mean** | — | **1.1470** | **0.2995** | — |
| **Std** | — | — | **0.0016** | — |

---

### Architecture

10L/512d U-Net, 25.25M params. GQA 8H/4KV, MLP 3x (1536 hidden), tied embeddings, logit softcap=30.0.

- **GELU Pre-Enrichment** (512→768→512): Wider nonlinear transformation before transformer blocks. Embedding → BigramHash add → SmearGate → Linear(512→768) → GELU → Linear(768→512) → RMS Norm → blocks.
- **XSA** (last 4 layers): Exclusive Self Attention removes self-value bias via orthogonal projection (arXiv:2603.09078, GQA-aware implementation from PR #265 @unnir). Zero parameters.
- **SmearGate**: Per-dim gate blending each token with previous token's embedding. F.pad for efficiency.
- **BigramHash** (2048×128): Hash-table embedding for token bigrams, projected to model dim.
- **U-Net skip connections**: Encoder-decoder with learnable skip weights.

Training: Muon+AdamW, WD=0.04, matrix_lr=0.025, scalar_lr=0.025, warmdown=3500 iters, batch=524K tokens, seq=2048. EMA decay=0.997. Int6 QAT + lzma (preset=6).

---

### EMA on GPU (37% faster training) — novel contribution

EMA state kept on GPU during training instead of synchronous GPU→CPU copy every step. Only moved to CPU at the end for serialization. To my knowledge, this optimization is not used in other submissions.

Step time: **64.4ms** (vs 101ms before). Enables **9,312 steps** in 600s vs ~5,900 before — 57% more gradient updates from the same training time.

---

### Multi-Order N-gram Backoff (score-first, backward-looking)

Multi-order n-gram backoff with entropy-adaptive alpha during sliding window eval. Concept credited to @deanbrr (PR #659), developed by PR #706 (@newjordan) and PR #727 (@Asukabot0).

**Protocol:**
- Multi-order backoff: orders 7→6→5→4→3→2, first hit with count≥2 wins
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
- High model entropy → trust n-gram more; low entropy → trust model
- Cache built from already-scored tokens only (backward-looking)
- Score-first: cache updated AFTER segment scoring
- Dual-array hash scheme: separate context count and pair count arrays per order (4M buckets each)
- Per-GPU independent cache, no cross-GPU sync
- Hash tables precomputed for all orders in single pass
- Integrated into sliding window eval (single pass)

**Compliance:**
- Score-first, backward-looking: n-gram counts built from previously scored tokens only
- No oracle selection: alpha depends solely on model's own entropy, never on ground-truth
- No cross-GPU sync: each GPU maintains its own independent cache

**Improvement:** 1.1478 → 0.3001 = **-0.848 BPB**

#### Pre-Enrichment Confidence Modulation

Uses the pre-enrichment layer's transformation magnitude as a confidence signal. High delta = model uncertain about this context = trust n-gram more. Low delta = model confident = trust model more. Modulates entropy-adaptive alpha by `(0.5 + 1.0 * pe_conf)`.

---

### Toggleable Features (default OFF, not used in this submission)

- `VALUE_RESIDUAL=1` — Layer-0 V mixed into all subsequent layers via learned sigmoid gates
- `GATED_ATTN=1` — Per-head sigmoid gates on attention output

---

### Reproduce

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults baked in. No env vars needed. 8xH100 SXM, 600s training + ~182s eval.

---

### Credits
- Muon optimizer — modded-nanogpt baseline (kellerjordan)
- SmearGate + BigramHash — PR #65 (@aquariouseworkman)
- XSA — arXiv:2603.09078; GQA-aware PR #265 (@unnir)
- EMA + GPTQ-lite + warmdown tuning — PR #414 (@signalrush)
- N-gram eval cache — concept PR #659 (@deanbrr); fixed 5-gram PR #706 (@newjordan); multi-order entropy-adaptive PR #727 (@Asukabot0)
- Shared GPU n-gram cache — PR #796 (@Robby955); chunk-synchronized PR #800 (@newjordan); PR #809 (@AayushBaniya2006)
- Per-order adaptive alpha — PR #798 (@travispchen); Cubric scaling PR #800 (@newjordan)
- Overtone init — modded-nanogpt baseline
- GELU Pre-Enrichment — original to this submission
- EMA on GPU — original to this submission
- Pre-Enrichment Confidence Modulation — original to this submission

### Included Files

- `train_gpt.py` — standalone training script with all modifications
- `train.log` — full 8xH100 training + eval log (seed 1337)
- `submission.json` — leaderboard metadata
- `README.md` — this file
