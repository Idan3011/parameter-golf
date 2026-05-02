"""Sliding-window eval on a checkpoint, distributed (8x H100 via torchrun).

Each scored token sees up to (seq_len - stride) tokens of preceding context,
vs ~seq_len/2 average for disjoint-chunk eval. No TTT, no EMA at eval time.

Auto-detects .ptz (quantized, deserializes via train_gpt.deserialize) vs
.pt (float, loads state_dict directly). For the ship metric, use .ptz.

Launch:
    torchrun --standalone --nproc_per_node=8 eval_sliding.py \\
        --ckpt final_model.int6.ptz --stride 64 --batch 8

Pattern lifted from the user's submission-branch eval_val_sliding.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_gpt as tg

    # --- DDP init (if torchrun launched us) ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    device = torch.device("cuda", local_rank)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"distributed={distributed} world_size={world_size} local_rank={local_rank}")

    # --- Hyperparameters + ValidationData (read from env / defaults) ---
    h = tg.Hyperparameters()
    h.local_rank = local_rank
    h.rank = rank
    h.world_size = world_size
    h.distributed = distributed
    h.is_main_process = (rank == 0)
    if tg.BOS_ID is None:
        tg.BOS_ID = 1

    log(f"loading val data + tokenizer (caseops_enabled={h.caseops_enabled})")
    val_data = tg.ValidationData(h, device)
    val_tokens = val_data.val_tokens
    val_bytes = val_data.val_bytes
    log(f"val_tokens: {val_tokens.numel()}")
    if val_bytes is None:
        raise RuntimeError("val_bytes sidecar required for caseops BPB computation")
    log(f"val_bytes : {val_bytes.numel()}")

    # --- Load checkpoint ---
    log(f"loading checkpoint {args.ckpt}")
    if args.ckpt.endswith(".ptz"):
        h.quantized_model_path = args.ckpt
        model = tg.deserialize(h, device)
        log("  loaded quantized model (.ptz, deserialized)")
    else:
        model = tg.GPT(h).to(device).bfloat16()
        tg.restore_fp32_params(model)
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        log("  loaded float model (.pt, state_dict)")
    model.eval()
    if h.num_loops > 0:
        model.looping_active = True

    # --- Build sliding windows, shard across ranks ---
    seq_len = args.seq_len
    stride = args.stride
    batch_size = args.batch
    n = val_tokens.numel()
    windows = []
    pos = 0
    while pos + seq_len < n:
        windows.append((pos, 0 if pos == 0 else seq_len - stride))
        pos += stride
    my_windows = windows[rank::world_size]
    log(f"sliding windows: total={len(windows)} per_rank={len(my_windows)} "
        f"(seq_len={seq_len}, stride={stride})")

    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_bytes = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=device)

    t0 = time.perf_counter()
    log_every = max(1, len(my_windows) // 10)
    with torch.inference_mode():
        for batch_start in range(0, len(my_windows), batch_size):
            batch = my_windows[batch_start:batch_start + batch_size]
            xs, ys = [], []
            for w_start, _ in batch:
                tokens = val_tokens[w_start : w_start + seq_len + 1]
                xs.append(tokens[:-1])
                ys.append(tokens[1:])
            x = torch.stack(xs).to(device=device, dtype=torch.int64)
            y = torch.stack(ys).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x).detach()
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            ).reshape(y.shape[0], y.shape[1])
            for i, (w_start, score_start) in enumerate(batch):
                scored = per_token[i, score_start:]
                total_loss += scored.to(torch.float64).sum()
                total_tokens += float(scored.numel())
                bs = val_bytes[w_start + 1 + score_start : w_start + 1 + seq_len].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                total_bytes += bs.to(torch.float64).sum()
            if (batch_start // batch_size) % log_every == 0 and rank == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  rank0 progress: {batch_start + len(batch)}/{len(my_windows)} windows "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )

    # --- All-reduce across ranks ---
    if distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    elapsed = time.perf_counter() - t0
    bpb = total_loss.item() / (total_bytes.item() * math.log(2.0))
    val_loss = total_loss.item() / max(total_tokens.item(), 1.0)
    log(
        f"sliding_window val_loss:{val_loss:.6f} val_bpb:{bpb:.6f} "
        f"tokens_scored:{int(total_tokens.item())} bytes:{int(total_bytes.item())} "
        f"elapsed:{elapsed:.1f}s"
    )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
