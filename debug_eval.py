"""Just load .pt and eval — no quantization. Tests if loading works."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (Hyperparameters, GPT, CastedLinear, build_sentencepiece_luts,
                        eval_val, load_validation_tokens)
import sentencepiece as spm

args = Hyperparameters()
device = torch.device("cuda")
print(f"vocab={args.vocab_size} mlp={args.mlp_mult} layers={args.num_layers}")

m = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init
).to(device).bfloat16()
for mod in m.modules():
    if isinstance(mod, CastedLinear):
        mod.float()

pt_path = sys.argv[1] if len(sys.argv) > 1 else "final_model_sp4096_10l_35mlp.pt"
sd = torch.load(pt_path, map_location=device, weights_only=True)
m.load_state_dict(sd, strict=False)
print(f"Loaded {sum(t.numel() for t in sd.values()):,} params")

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
vt = load_validation_tokens(args.val_files, args.train_seq_len)
bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)
vl, vb = eval_val(args, m, 0, 1, device, 1, vt, bl, hl, il)
print(f"Pre-quant val_bpb: {vb:.4f}")
