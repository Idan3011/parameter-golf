"""TTT eval only — no sliding, no post-quant eval. Fastest possible."""
import os, sys, io, time, torch, math, lzma
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(__file__))
try:
    import brotli
except ImportError:
    pass
from train_gpt import (Hyperparameters, GPT, CastedLinear, eval_val_ttt,
                        build_sentencepiece_luts, load_validation_tokens, _decompress)
import sentencepiece as spm

distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
else:
    rank, world_size = 0, 1
    device = torch.device("cuda")

args = Hyperparameters()

base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
).to(device).bfloat16()
for m in base_model.modules():
    if isinstance(m, CastedLinear): m.float()

ptz = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
if rank == 0: print(f"Loading {ptz}...")
with open(ptz, "rb") as f:
    blob = f.read()
qs = torch.load(io.BytesIO(_decompress(blob)), map_location="cpu")
deq = {}
for name, q in qs["quantized"].items():
    s = qs["scales"][name]
    if s.ndim > 0:
        deq[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(torch.bfloat16)
    else:
        deq[name] = (q.float() * float(s.item())).to(torch.bfloat16)
for name, t in qs["passthrough"].items():
    orig = qs.get("passthrough_orig_dtypes", {}).get(name)
    deq[name] = t.to(dtype=getattr(torch, orig)) if isinstance(orig, str) else t
base_model.load_state_dict(deq, strict=False)
if rank == 0: print(f"Model loaded ({world_size} GPUs). Starting TTT...")

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
vt = load_validation_tokens(args.val_files, args.train_seq_len)
bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)

log_fn = print if rank == 0 else None
t0 = time.perf_counter()
ttt_bpb = eval_val_ttt(args, base_model, rank, world_size, device, vt, bl, hl, il, log_fn=log_fn)
elapsed = time.perf_counter() - t0
if rank == 0:
    print(f"final_ttt val_bpb:{ttt_bpb:.4f} eval_time:{elapsed*1000:.0f}ms")
    print(f"final_ttt_exact val_bpb:{ttt_bpb:.8f}")
if distributed: dist.destroy_process_group()
