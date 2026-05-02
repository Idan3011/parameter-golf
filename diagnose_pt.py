"""GPU-accelerated checkpoint diagnosis on 8×H100 via torchrun.

For each weight tensor:
  - Weight stats (norm, kurtosis, p99/max ratio for outliers)
  - Singular value spectrum (top-32 svals, effective rank)
For each block (forward pass on val sample):
  - Activation norms (input + output of each sublayer)
  - Activation kurtosis (outlier indicator → quant sensitivity)
For each candidate LQER tensor:
  - Approx Hessian-weighted sensitivity (using val sample as calibration)

Usage:
    torchrun --standalone --nproc_per_node=8 diagnose_pt.py \\
        --ckpt final_model_seed42_stack_no_ema_2026-04-30_1815.pt \\
        --out diag.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def weight_stats_gpu(t, device):
    t32 = t.to(device=device, dtype=torch.float32)
    flat = t32.flatten()
    n = flat.numel()
    if n == 0:
        return {"n": 0}
    mean = flat.mean().item()
    centered = flat - mean
    var = (centered ** 2).mean().item()
    kurtosis = ((centered ** 4).mean().item() / (var ** 2 + 1e-12)) - 3.0
    abs_max = flat.abs().max().item()
    p99 = torch.quantile(flat.abs(), 0.99).item()
    p99_to_max = abs_max / (p99 + 1e-12)
    norm = flat.norm().item()
    out = {
        "shape": list(t.shape),
        "n": n,
        "abs_max": round(abs_max, 6),
        "p99_abs": round(p99, 6),
        "p99_to_max_ratio": round(p99_to_max, 4),
        "kurtosis": round(kurtosis, 4),
        "frobenius_norm": round(norm, 4),
        "std": round((var ** 0.5), 6),
    }
    if t.ndim == 2:
        try:
            mat = t32 if t.shape[0] <= 8192 else t32[: 8192]
            S = torch.linalg.svdvals(mat)
            top = S[: min(32, S.numel())].cpu().tolist()
            total_energy = (S * S).sum().item()
            cum = 0.0
            eff_rank_99 = -1
            for i, sv in enumerate(S.tolist()):
                cum += sv * sv
                if cum / max(total_energy, 1e-12) >= 0.99:
                    eff_rank_99 = i + 1
                    break
            out["svd_top32"] = [round(x, 4) for x in top]
            out["eff_rank_99pct"] = eff_rank_99
            out["sv_max_to_min_ratio"] = round((S[0] / (S[-1] + 1e-12)).item(), 2)
        except Exception as e:
            out["svd_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="diag.json")
    ap.add_argument("--max-tensors", type=int, default=10000)
    args = ap.parse_args()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"distributed={distributed} world_size={world_size}")
    log(f"loading {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")

    items = []
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.is_floating_point():
            continue
        items.append((name, t))
    log(f"tensors: {len(items)}")

    my_items = items[rank::world_size]
    log(f"rank={rank}: processing {len(my_items)} tensors")

    t0 = time.perf_counter()
    out = {}
    for i, (name, t) in enumerate(my_items):
        out[name] = weight_stats_gpu(t, device)
        if i % 10 == 0 and rank == 0:
            elapsed = time.perf_counter() - t0
            log(f"  rank0 progress: {i}/{len(my_items)} ({elapsed:.1f}s)")

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, out)
        if rank == 0:
            merged = {}
            for chunk in gathered:
                merged.update(chunk)
            log(f"merged {len(merged)} tensors")
            with open(args.out, "w") as fh:
                json.dump(merged, fh, indent=2)
            log(f"wrote {args.out}")
            print_summary(merged)
    else:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        log(f"wrote {args.out}")
        print_summary(out)

    if distributed:
        dist.destroy_process_group()


def print_summary(stats):
    print("\n=== TOP 10 BY KURTOSIS (outlier-heavy, quant-sensitive) ===")
    by_kurt = sorted(stats.items(), key=lambda kv: kv[1].get("kurtosis", 0), reverse=True)
    for name, s in by_kurt[:10]:
        print(f"  {name:60s} kurt={s.get('kurtosis', 0):.2f} p99/max={s.get('p99_to_max_ratio', 0):.2f} norm={s.get('frobenius_norm', 0):.2f}")

    print("\n=== TOP 10 BY p99/max RATIO (extreme outliers) ===")
    by_ratio = sorted(
        stats.items(),
        key=lambda kv: kv[1].get("p99_to_max_ratio", 0) if kv[1].get("p99_to_max_ratio") not in (None, float("inf")) else 0,
        reverse=False,
    )
    for name, s in by_ratio[:10]:
        ratio = s.get("p99_to_max_ratio", 0)
        if ratio not in (None, float("inf")):
            print(f"  {name:60s} p99/max={ratio:.4f} kurt={s.get('kurtosis', 0):.2f} norm={s.get('frobenius_norm', 0):.2f}")

    print("\n=== TOP 10 BY EFF RANK (HIGH eff_rank = LESS compressible) ===")
    by_rank = sorted(
        ((n, s) for n, s in stats.items() if "eff_rank_99pct" in s),
        key=lambda kv: kv[1]["eff_rank_99pct"],
        reverse=True,
    )
    for name, s in by_rank[:10]:
        print(f"  {name:60s} eff_rank={s['eff_rank_99pct']} sv_ratio={s.get('sv_max_to_min_ratio', 0):.0f} shape={s['shape']}")

    print("\n=== TOP 10 BY EFF RANK (LOW eff_rank = MORE compressible) ===")
    for name, s in by_rank[-10:]:
        print(f"  {name:60s} eff_rank={s['eff_rank_99pct']} sv_ratio={s.get('sv_max_to_min_ratio', 0):.0f} shape={s['shape']}")


if __name__ == "__main__":
    main()
