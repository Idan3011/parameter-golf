"""Run GPTQ + compress + eval on an existing .pt file. No training."""
import os, sys, io, time, torch
sys.path.insert(0, os.path.dirname(__file__))

pt_path = sys.argv[1] if len(sys.argv) > 1 else "final_model.pt"
print(f"Loading {pt_path}...")

from train_gpt import (
    Hyperparameters, GPT, apply_gptq_inplace, quantize_state_dict_int6,
    build_sentencepiece_luts, eval_val, load_validation_tokens, _byte_shuffle,
)
import numpy as np
import sentencepiece as spm
import math
try:
    import brotli
    COMP = "brotli"
except ImportError:
    COMP = "lzma"
import lzma

args = Hyperparameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
).to(device).bfloat16()
for m in base_model.modules():
    if isinstance(m, torch.nn.Linear):
        m.float()

sd = torch.load(pt_path, map_location=device, weights_only=True)
base_model.load_state_dict(sd, strict=False)
print(f"  {sum(t.numel() for t in sd.values()):,} params loaded")

if bool(int(os.environ.get("USE_GPTQ", "1"))):
    apply_gptq_inplace(base_model, device, args, log_fn=print)

quant_obj, quant_stats = quantize_state_dict_int6(base_model.state_dict())
buf = io.BytesIO()
torch.save(quant_obj, buf)
quant_raw = buf.getvalue()
if COMP == "brotli":
    quant_blob = brotli.compress(_byte_shuffle(quant_raw), quality=11)
else:
    quant_blob = lzma.compress(quant_raw, preset=9)
code_bytes = 72000
total = len(quant_blob) + code_bytes
print(f"Artifact: {len(quant_blob):,} bytes ({len(quant_blob)/1e6:.2f}MB)")
print(f"Total: {total:,} bytes ({total/1e6:.2f}MB)")
print(f"Fits: {'YES' if total < 16_000_000 else 'NO'} (delta: {16_000_000 - total:+,})")

with open("final_model.int6.ptz", "wb") as f:
    f.write(quant_blob)

if torch.cuda.is_available():
    print("Evaluating post-quant BPB...")
    deq_sd = {}
    for name, q in quant_obj["quantized"].items():
        s = quant_obj["scales"][name]
        if s.ndim > 0:
            deq_sd[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(torch.bfloat16)
        else:
            deq_sd[name] = (q.float() * float(s.item())).to(torch.bfloat16)
    for name, t in quant_obj["passthrough"].items():
        orig_dtype = quant_obj.get("passthrough_orig_dtypes", {}).get(name)
        if isinstance(orig_dtype, str):
            deq_sd[name] = t.to(dtype=getattr(torch, orig_dtype))
        else:
            deq_sd[name] = t
    base_model.load_state_dict(deq_sd, strict=False)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_loss, val_bpb = eval_val(args, base_model, 0, 1, device, 1, val_tokens, bl, hl, il)
    print(f"Post-quant val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
