"""Smart requant sweep for pod. Loads a float.pt, tries configs in order.
For each: GPTQ → check size → if fits: eval → if gap OK: TTT.
Skips expensive steps on configs that fail early checks.

Usage:
  FLOAT_PATH=final_model.float_12l_4mlp.pt MLP_MULT=4.0 GAP_THRESHOLD=0.025 \
  torchrun --standalone --nproc_per_node=8 archive/scripts/misc/smart_requant_sweep.py
"""
from __future__ import annotations
import io, math, os, sys, time
import brotli, sentencepiece as spm, torch, torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import train_gpt as tg

FLOAT_PATH = os.environ.get("FLOAT_PATH", "final_model.float_12l_4mlp.pt")
MAX_BYTES = 16_000_000 - 15_000
GAP_THRESHOLD = float(os.environ.get("GAP_THRESHOLD", "0.025"))

CONFIGS = [
    ("k=18 cr=31", 18.0, 31),
    ("k=20 cr=31", 20.0, 31),
]

def log0(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)

def pack_model(export_sd, gptq_scales, cr):
    quantized_t, scales_t, dtypes_t, passthrough_t, qmeta_t = {}, {}, {}, {}, {}
    passthrough_orig_dtypes_t = {}
    for name, tensor in export_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough_t[name] = t; continue
        if any(p in name for p in tg.CONTROL_TENSOR_NAME_PATTERNS):
            passthrough_t[name] = t.float().contiguous(); continue
        if t.numel() <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough_t[name] = tg.keep_float_tensor(name, t, passthrough_orig_dtypes_t); continue
        if "tok_emb.weight" in name:
            q, s = tg.quantize_float_tensor_int6(t, bits=8)
        elif name in gptq_scales:
            sf = gptq_scales[name].float()
            q = torch.clamp(torch.round(t.float() / sf[:, None]), -cr, cr).to(torch.int8).contiguous()
            s = sf.to(dtype=tg.INT8_PER_ROW_SCALE_DTYPE).contiguous()
        else:
            q, s = tg.quantize_float_tensor_int6(t, bits=6)
        quantized_t[name] = q; scales_t[name] = s; dtypes_t[name] = str(t.dtype).removeprefix("torch.")
        if s.ndim > 0: qmeta_t[name] = {"scheme": "per_row", "axis": 0}
    obj = {"__quant_format__": f"gptq_sdclip_cr{cr}_per_row_v1",
           "quantized": quantized_t, "scales": scales_t, "dtypes": dtypes_t, "passthrough": passthrough_t}
    if qmeta_t: obj["qmeta"] = qmeta_t
    if passthrough_orig_dtypes_t: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes_t
    buf = io.BytesIO(); torch.save(obj, buf); raw = buf.getvalue()
    blob = brotli.compress(tg._byte_shuffle(raw), quality=11)
    return obj, blob

def main():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    master = rank == 0

    if master:
        import glob as _g
        if not os.path.exists("data/tokenizers/fineweb_9000_bpe.model") or not _g.glob("data/datasets/fineweb10B_sp9000/fineweb_val_*.bin"):
            from huggingface_hub import snapshot_download
            import shutil
            cache = snapshot_download("Idan3011/parameter-golf-sp9000", repo_type="dataset",
                                      allow_patterns=["*val*.bin", "*.model", "*.vocab"])
            os.makedirs("data/tokenizers", exist_ok=True)
            os.makedirs("data/datasets/fineweb10B_sp9000", exist_ok=True)
            for root, _, files in os.walk(cache):
                for f in files:
                    if f.endswith(".bin") and "val" not in f: continue
                    dst = os.path.join("data/tokenizers" if f.endswith((".model", ".vocab")) else "data/datasets/fineweb10B_sp9000", f)
                    if not os.path.exists(dst): shutil.copy2(os.path.join(root, f), dst); log0(f"  copied {f}")
    if distributed:
        dist.barrier()
    args = tg.Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = tg.build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = tg.load_validation_tokens(args.val_files, args.train_seq_len)
    grad_accum = 8 // world_size

    model = tg.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        num_loops=args.num_loops, loop_start=args.loop_start, loop_end=args.loop_end,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear): m.float()
    tg.restore_low_dim_params_to_fp32(model)

    log0(f"loading {FLOAT_PATH}...")
    sd = torch.load(FLOAT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)

    log0("baseline eval (float)...")
    compiled = torch.compile(model)
    compiled.eval()
    _, baseline_bpb = tg.eval_val(args, compiled, rank, world_size, device, grad_accum,
                                   val_tokens, *luts)
    log0(f"baseline: {baseline_bpb:.6f}")

    best_tag, best_ttt = None, 999.0

    for tag, k, cr in CONFIGS:
        log0(f"\n{'='*60}")
        log0(f"CONFIG: {tag}")

        model.load_state_dict(sd, strict=True)
        for m in model.modules():
            if isinstance(m, tg.CastedLinear): m.float()
        tg.restore_low_dim_params_to_fp32(model)

        tg.GPTQ_SD_K = k
        tg.GPTQ_CR = cr
        gptq_scales = tg.apply_gptq_sdclip_inplace(model, device, args, log_fn=log0)
        quant_obj, blob = pack_model(model.state_dict(), gptq_scales, cr)
        artifact_bytes = len(blob)
        log0(f"  artifact: {artifact_bytes/1e6:.3f} MB")

        fits = artifact_bytes <= MAX_BYTES
        if not fits:
            log0(f"  OVER CAP by {(artifact_bytes - MAX_BYTES)/1000:.0f} KB")

        if master and fits:
            with open("final_model.int6.ptz", "wb") as f: f.write(blob)
        if distributed: dist.barrier()

        deq = tg.dequantize_state_dict_int8(quant_obj)
        model.load_state_dict(deq, strict=True)
        compiled = torch.compile(model)
        compiled.eval()
        _, q_bpb = tg.eval_val(args, compiled, rank, world_size, device, grad_accum,
                                val_tokens, *luts)
        gap = q_bpb - baseline_bpb
        log0(f"  post-quant: {q_bpb:.6f}  gap: {gap:+.6f}  {'FITS' if fits else 'OVER'}")

        if not fits:
            log0(f"  SKIP sliding+TTT — artifact over cap")
            continue
        if gap > GAP_THRESHOLD:
            log0(f"  SKIP sliding+TTT — gap {gap:+.4f} exceeds threshold {GAP_THRESHOLD}")
            continue

        _, sw_bpb = tg.eval_val_sliding(args, compiled, rank, world_size, device,
                                         val_tokens, *luts)
        log0(f"  sliding: {sw_bpb:.6f}  (gain over post-quant: {sw_bpb - q_bpb:+.6f})")

        log0(f"  GAP OK — running TTT...")
        ttt_model = tg.GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            num_loops=args.num_loops, loop_start=args.loop_start, loop_end=args.loop_end,
        ).to(device).bfloat16()
        for m in ttt_model.modules():
            if isinstance(m, tg.CastedLinear): m.float()
        tg.restore_low_dim_params_to_fp32(ttt_model)
        ttt_model.load_state_dict(tg.dequantize_state_dict_int8(quant_obj), strict=False)
        ttt_model.eval_hash_emb = nn.Embedding(args.eval_hash_buckets, args.model_dim).to(device)
        nn.init.zeros_(ttt_model.eval_hash_emb.weight)
        ttt_model = torch.compile(ttt_model)
        ttt_bpb = tg.eval_val_ttt(args, ttt_model, rank, world_size, device,
                                    val_tokens, *luts, log_fn=log0)
        log0(f"  TTT: {ttt_bpb:.6f}")
        del ttt_model; torch.cuda.empty_cache()

        if ttt_bpb < best_ttt:
            best_ttt = ttt_bpb
            best_tag = tag
            log0(f"  NEW BEST!")

    log0(f"\n{'='*60}")
    if best_tag:
        log0(f"BEST: {best_tag} → TTT val_bpb: {best_ttt:.6f}")
    else:
        log0("NO CONFIG PASSED ALL CHECKS")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
