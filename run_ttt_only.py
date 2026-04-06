"""TTT eval only — no sliding, no post-quant eval. Fastest possible."""
import os, sys, io, time, torch, math, lzma
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(__file__))
try:
    import brotli
except ImportError:
    pass
from train_gpt import (Hyperparameters, GPT, CastedLinear, eval_val_ttt, eval_val,
                        build_sentencepiece_luts, load_validation_tokens, _decompress,
                        dequantize_state_dict_int8, restore_low_dim_params_to_fp32)
import sentencepiece as spm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

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
if rank == 0:
    print(f"Config: vocab={args.vocab_size} layers={args.num_layers} dim={args.model_dim} "
          f"mlp={args.mlp_mult} tied={args.tie_embeddings} seq_len={args.train_seq_len}")

base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
).to(device).bfloat16()
for m in base_model.modules():
    if isinstance(m, CastedLinear): m.float()
restore_low_dim_params_to_fp32(base_model)

ptz = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
if rank == 0: print(f"Loading {ptz} ({os.path.getsize(ptz)} bytes)...")
with open(ptz, "rb") as f:
    blob = f.read()
qs = torch.load(io.BytesIO(_decompress(blob)), map_location="cpu")

if rank == 0:
    print(f"  quantized keys: {len(qs['quantized'])}")
    print(f"  passthrough keys: {len(qs['passthrough'])}")
    model_keys = set(k for k, _ in base_model.named_parameters())
    loaded_keys = set(qs['quantized'].keys()) | set(qs['passthrough'].keys())
    missing = model_keys - loaded_keys
    extra = loaded_keys - model_keys
    if missing: print(f"  MISSING from ptz: {missing}")
    if extra: print(f"  EXTRA in ptz (not in model): {extra}")

deq = dequantize_state_dict_int8(qs)
result = base_model.load_state_dict(deq, strict=False)
if rank == 0:
    if result.missing_keys: print(f"  !!! MISSING keys: {result.missing_keys}")
    if result.unexpected_keys: print(f"  !!! UNEXPECTED keys: {result.unexpected_keys}")
restore_low_dim_params_to_fp32(base_model)
if rank == 0:
    for n in ["tok_emb.weight", "blocks.0.mlp.fc.weight", "blocks.9.mlp.proj.weight", "skip_weights"]:
        p = dict(base_model.named_parameters())[n]
        print(f"  {n}: dtype={p.dtype} mean={p.float().mean():.6f} std={p.float().std():.6f} min={p.float().min():.4f} max={p.float().max():.4f}")
    print(f"Model loaded ({world_size} GPUs).")

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
vt = load_validation_tokens(args.val_files, args.train_seq_len)
bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)

if rank == 0: print(f"Val tokens: {vt.numel()}")

if rank == 0: print("Quick sanity: running eval_val (non-sliding)...")
sanity_loss, sanity_bpb = eval_val(args, base_model, rank, world_size, device, 1, vt, bl, hl, il)
if rank == 0: print(f"eval_val sanity: loss={sanity_loss:.4f} bpb={sanity_bpb:.4f}")

ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
if ttt_epochs > 0:
    deq2 = dequantize_state_dict_int8(qs)
    base_model.load_state_dict(deq2, strict=False)
    restore_low_dim_params_to_fp32(base_model)
    if rank == 0: print(f"Model reloaded for TTT ({ttt_epochs} epochs)...")
    log_fn = print if rank == 0 else None
    t0 = time.perf_counter()
    ttt_bpb = eval_val_ttt(args, base_model, rank, world_size, device, vt, bl, hl, il, log_fn=log_fn)
    elapsed = time.perf_counter() - t0
    if rank == 0:
        print(f"final_ttt val_bpb:{ttt_bpb:.4f} eval_time:{elapsed*1000:.0f}ms")
        print(f"final_ttt_exact val_bpb:{ttt_bpb:.8f}")
else:
    if rank == 0: print("TTT_EPOCHS=0, skipping TTT.")

if distributed: dist.destroy_process_group()
