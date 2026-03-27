# Valid-Distribution N-gram Mixing (val_bpb = 0.3922)

## Approach

This submission replaces the standard `pair_count / ctx_count` n-gram scoring
with properly normalized 1024-token distributions. For each scored position and
each n-gram order, we look up counts for ALL 1024 vocabulary tokens and normalize
to sum to 1.0, rather than computing a pseudo-probability for only the correct token.

### Why this matters

Standard n-gram scoring in hash tables computes `p = pair_count / ctx_count`.
Due to hash collisions, both counts are inflated by unrelated entries. The pair
bucket is specifically queried for the target token being scored, creating an
information leak through collision structure.

A 256M-bucket experiment (near collision-free) showed the n-gram contribution
drops to near-zero (1.1123 vs 1.1109 float base). This submission quantifies
the collision premium: **0.12 BPP** of the 0.87 BPP total n-gram gain comes
from inflated probabilities.

## Architecture

- **Neural base**: 10L/512d GPT with pre-enrichment, SmearGate, BigramHash, U-Net skips
- **Quantization**: int6 per-row + GPTQ-lite + lzma (same artifact as PR #810)
- **N-gram**: Orders 2-11, 4M buckets, first-match-wins backoff
- **Scoring**: Full 1024-token distribution lookup + normalization per order
- **Mixing**: Bayesian local estimate `p_local = (raw + beta * p_neural) / (ctx + beta)` with beta=2.0
- **Blend**: Fixed 0.5: `final = 0.5 * p_local + 0.5 * p_neural`
- **No phrase cache** (dropped — same collision problem)

## Results

| Metric | Value |
|---|---|
| Neural sliding BPB | 1.1478 |
| Valid n-gram val_bpb | **0.3922** |
| Collision-exploiting val_bpb (PR #810) | 0.2722 |
| Collision premium | 0.12 BPP |

### A/B Experiment

| Config | val_bpb |
|---|---|
| Fixed 0.5 blend | **0.3922** |
| Count-confidence (gain=12) | 0.4942 |
| Count-confidence (gain=50) | 0.7041 |

Less gating = better. The n-gram signal is real but weak after normalization.

## Hardware

8xH100 SXM, EVAL_ONLY on existing artifact from PR #810.
Eval time: ~193s.

## Credits

- Pre-enrichment MLP: original contribution
- N-gram cache concept: PR #659 (@deanbrr), PR #706 (@newjordan)
- Multi-order backoff: PR #727 (@newjordan)
- Bayesian mixing inspired by PR #944 (@aamodbhatt) Dirichlet approach
- Context Tree Weighting theory: Willems, Shtarkov, Tjalkens (1995)
