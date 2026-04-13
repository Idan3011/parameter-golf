"""Pod test: two configs on 5090.
Config 1: GPTQ k=12.85 + opt-scale + emb5
Config 4: opt-scale k=12.85 + emb5 (NO GPTQ)
Downloads float from HF. No flash-attn needed."""
from __future__ import annotations
import io, math, os, sys, time
import brotli, sentencepiece as spm, torch, torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import train_gpt as tg

FLOAT_HF = "final_model.float_12l_9000.pt"
FLOAT_REPO = "Idan3011/parameter-golf-sp9000"
MAX_BYTES = 16_000_000 - 15_000

def log0(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)

def opt_scale_quant(t_orig, k, cr, lo=0.85, hi=1.15, n=300):
    t32 = t_orig.float()
    sf_def = (k * t32.std(dim=1).clamp_min(1e-12) / cr)
    best_sf = sf_def.clone()
    best_mse = torch.full((t32.shape[0],), float('inf'))
    for mult in torch.linspace(lo, hi, n):
        sf_try = sf_def * mult
        q = torch.clamp(torch.round(t32 / sf_try[:, None]), -cr, cr)
        mse = ((t32 - q * sf_try[:, None])**2).mean(dim=1)
        better = mse < best_mse
        best_sf[better] = sf_try[better]
        best_mse[better] = mse[better]
    q = torch.clamp(torch.round(t32 / best_sf[:, None]), -cr, cr).to(torch.int8)
    return q, best_sf

def pack_artifact(export_sd, scales, cr, emb_bits):
    quantized, scale_t, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    for name, tensor in export_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough[name] = t; continue
        if any(p in name for p in tg.CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float().contiguous(); continue
        if t.numel() <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = tg.keep_float_tensor(name, t, passthrough_orig_dtypes)
            continue
        if "tok_emb" in name:
            q, s = tg.quantize_float_tensor_int6(t, bits=emb_bits)
        elif name in scales:
            sf = scales[name].float()
            q = torch.clamp(torch.round(t.float() / sf[:, None]), -cr, cr).to(torch.int8).contiguous()
            s = sf.to(dtype=tg.INT8_PER_ROW_SCALE_DTYPE).contiguous()
        else:
            q, s = tg.quantize_float_tensor_int6(t, bits=6)
        quantized[name] = q; scale_t[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
    obj = {"__quant_format__": f"optscale_cr{cr}", "quantized": quantized, "scales": scale_t,
           "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
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
            cache = snapshot_download(FLOAT_REPO, repo_type="dataset",
                                      allow_patterns=["*val*.bin", "*.model", "*.vocab"])
            os.makedirs("data/tokenizers", exist_ok=True)
            os.makedirs("data/datasets/fineweb10B_sp9000", exist_ok=True)
            for root, _, files in os.walk(cache):
                for f in files:
                    if f.endswith(".bin") and "val" not in f: continue
                    dst = os.path.join("data/tokenizers" if f.endswith((".model", ".vocab")) else "data/datasets/fineweb10B_sp9000", f)
                    if not os.path.exists(dst): shutil.copy2(os.path.join(root, f), dst); log0(f"  copied {f}")
        if not os.path.exists(FLOAT_HF):
            from huggingface_hub import hf_hub_download
            hf_hub_download(FLOAT_REPO, FLOAT_HF, repo_type="dataset", local_dir=".")
            log0(f"downloaded {FLOAT_HF}")
    if distributed: dist.barrier()

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

    log0(f"loading {FLOAT_HF}...")
    sd_orig = torch.load(FLOAT_HF, map_location="cpu", weights_only=False)

    configs = [
        ("CONFIG A: GPTQ k=12.85 + emb5 (SOTA recipe)", 12.85, 31, True, False, 5),
        ("CONFIG B: GPTQ k=12.85 + opt-scale + emb5", 12.85, 31, True, True, 5),
        ("CONFIG C: opt-scale k=12.85 + emb5 (NO GPTQ)", 12.85, 31, False, True, 5),
        ("CONFIG D: GPTQ k=15 + emb8 (baseline)", 15.0, 31, True, False, 8),
    ]

    for tag, k, cr, use_gptq, use_opt, emb_bits in configs:
        log0(f"\n{'='*60}")
        log0(f"  {tag}")
        log0(f"{'='*60}")

        model.load_state_dict(sd_orig, strict=True)
        for m in model.modules():
            if isinstance(m, tg.CastedLinear): m.float()
        tg.restore_low_dim_params_to_fp32(model)

        if use_gptq:
            tg.GPTQ_SD_K = k; tg.GPTQ_CR = cr
            gptq_scales = tg.apply_gptq_sdclip_inplace(model, device, args, log_fn=log0)
        else:
            gptq_scales = {}
            for name, module in model.named_modules():
                if isinstance(module, tg.CastedLinear) and module.weight.ndim == 2:
                    pname = name + ".weight"
                    sf = (k * module.weight.data.float().std(dim=1).clamp_min(1e-12) / cr).to(device)
                    gptq_scales[pname] = sf.cpu()

        if use_opt:
            log0("  running opt-scale search...")
            final_scales = {}
            for name, module in model.named_modules():
                if isinstance(module, tg.CastedLinear) and module.weight.ndim == 2:
                    pname = name + ".weight"
                    t_orig = sd_orig[pname].float().to(device)
                    sf_base = gptq_scales[pname].float().to(device)
                    best_sf = sf_base.clone()
                    best_mse = torch.full((t_orig.shape[0],), float('inf'), device=device)
                    for mult in torch.linspace(0.85, 1.15, 300, device=device):
                        sf_try = sf_base * mult
                        q = torch.clamp(torch.round(t_orig / sf_try[:, None]), -cr, cr)
                        mse = ((t_orig - q * sf_try[:, None])**2).mean(dim=1)
                        better = mse < best_mse
                        best_sf[better] = sf_try[better]
                        best_mse[better] = mse[better]
                    final_scales[pname] = best_sf.cpu()
                    q_best = torch.clamp(torch.round(t_orig / best_sf[:, None]), -cr, cr)
                    recon = q_best * best_sf[:, None]
                    with torch.no_grad():
                        module.weight.data.copy_(recon.to(module.weight.dtype))
            scales_for_pack = final_scales
        else:
            scales_for_pack = gptq_scales

        quant_obj, blob = pack_artifact(model.state_dict(), scales_for_pack, cr, emb_bits)
        art = len(blob)
        log0(f"  artifact: {art/1e6:.3f}M  margin: {(MAX_BYTES-art)/1000:.0f}KB  {'OK' if art<=MAX_BYTES else 'OVER'}")

        if art > MAX_BYTES:
            log0("  SKIP — over cap"); continue

        if master:
            with open("final_model.int6.ptz", "wb") as f: f.write(blob)
        if distributed: dist.barrier()

        deq = tg.dequantize_state_dict_int8(quant_obj)
        model.load_state_dict(deq, strict=True)
        for m in model.modules():
            if isinstance(m, tg.CastedLinear): m.float()
        tg.restore_low_dim_params_to_fp32(model)
        compiled = torch.compile(model)
        compiled.eval()

        torch.cuda.synchronize(); t0 = time.perf_counter()
        _, q_bpb = tg.eval_val(args, compiled, rank, world_size, device, grad_accum, val_tokens, *luts)
        log0(f"  post-quant: {q_bpb:.6f}  eval_time: {time.perf_counter()-t0:.0f}s")

        if q_bpb >= 1.1260:
            log0(f"  SKIP — post-quant {q_bpb:.4f} not better than baseline 1.1260")
            continue

        torch.cuda.synchronize(); t0 = time.perf_counter()
        _, sw_bpb = tg.eval_val_sliding(args, compiled, rank, world_size, device,
                                         val_tokens, *luts, batch_size=32)
        log0(f"  sliding: {sw_bpb:.6f}  eval_time: {time.perf_counter()-t0:.0f}s")

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
        ttt_model.load_state_dict(deq, strict=False)
        ttt_model.eval_hash_emb = nn.Embedding(args.eval_hash_buckets, args.model_dim).to(device)
        nn.init.zeros_(ttt_model.eval_hash_emb.weight)
        ttt_model = torch.compile(ttt_model)
        log0("  ttt: starting")
        ttt_bpb = tg.eval_val_ttt(args, ttt_model, rank, world_size, device,
                                    val_tokens, *luts, log_fn=log0)
        log0(f"  ttt: {ttt_bpb:.6f}")
        del ttt_model; torch.cuda.empty_cache()

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
