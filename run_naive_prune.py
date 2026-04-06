"""Load GPTQ'd .pt, quantize with naive sparsity, compress, check size."""
import os, sys, io, time, torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
try:
    import brotli
    COMP = "brotli"
except ImportError:
    COMP = "lzma"
import lzma

LIMIT = 16_000_000
CODE = 72_000
CTRL = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "smear", "scale")

path = sys.argv[1] if len(sys.argv) > 1 else "final_model_11l_8gpu.pt"
print(f"Loading {path}...")
sd = torch.load(path, map_location="cpu", weights_only=True)
print(f"  {sum(t.numel() for t in sd.values()):,} params")

def byte_shuffle(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    pad = (4 - len(arr) % 4) % 4
    if pad: arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return bytes(arr.reshape(-1, 4).T.ravel())

def quant_and_prune(sd, clip, sparsity):
    quantized, scales, passthrough = {}, {}, {}
    for name, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536 or "tok_emb" in name:
            if any(c in name for c in CTRL): passthrough[name] = t.float()
            elif t.dtype in {torch.float32, torch.bfloat16}: passthrough[name] = t.to(torch.float16)
            else: passthrough[name] = t
            continue
        t32 = t.float()
        if t32.ndim == 2:
            s = (t32.abs().amax(dim=1).clamp_min(1e-12) / clip).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip, clip).to(torch.int8)
        else:
            amax = t32.abs().max().item()
            s = torch.tensor(amax / clip if amax > 0 else 1.0, dtype=torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()), -clip, clip).to(torch.int8)
        if sparsity > 0 and q.ndim == 2:
            flat = q.view(-1).abs().float()
            threshold = torch.quantile(flat, sparsity)
            q[q.abs() <= threshold.to(torch.int8)] = 0
        quantized[name] = q.contiguous()
        scales[name] = s.contiguous()
    obj = {"quantized": quantized, "scales": scales, "passthrough": passthrough}
    buf = io.BytesIO(); torch.save(obj, buf); raw = buf.getvalue()
    if COMP == "brotli":
        blob = brotli.compress(byte_shuffle(raw), quality=11)
    else:
        blob = lzma.compress(raw, preset=9)
    return blob

print(f"\nNaive prune after GPTQ (int5-all + brotli + byte-shuffle):")
for sp in [0.0, 0.15, 0.20, 0.25, 0.28, 0.30]:
    t0 = time.time()
    blob = quant_and_prune(sd, 15, sp)
    total = len(blob) + CODE
    delta = LIMIT - total
    fits = "FITS" if delta > 0 else "OVER"
    print(f"  prune={sp:.2f}  {total/1e6:.2f}MB  {fits} ({delta:+,})  [{time.time()-t0:.0f}s]")
    if fits == "FITS" and sp > 0:
        with open(f"final_model_sparse{int(sp*100)}.int6.ptz", "wb") as f:
            f.write(blob)
        print(f"    Saved final_model_sparse{int(sp*100)}.int6.ptz")
        if torch.cuda.is_available():
            from train_gpt import (GPT, Hyperparameters, dequantize_state_dict_int8,
                                   build_sentencepiece_luts, eval_val, load_validation_tokens)
            from train_gpt import _byte_unshuffle, _decompress
            import sentencepiece as spm
            args = Hyperparameters()
            device = torch.device("cuda")
            base_model = GPT(
                vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
                num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
                tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
                logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            ).to(device).bfloat16()
            qs = torch.load(io.BytesIO(_decompress(blob)), map_location="cpu")
            base_model.load_state_dict(dequantize_state_dict_int8(qs), strict=False)
            sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
            vt = load_validation_tokens(args.val_files, args.train_seq_len)
            bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)
            vl, vb = eval_val(args, base_model, 0, 1, device, 1, vt, bl, hl, il)
            print(f"    Post-quant val_bpb:{vb:.4f}")
            del base_model; torch.cuda.empty_cache()
