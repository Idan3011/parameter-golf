import os, torch, sys
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import Hyperparameters, GPT, quantize_state_dict_int6
args = Hyperparameters()
print(f"vocab={args.vocab_size} mlp={args.mlp_mult} layers={args.num_layers}")
m = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init)
pt_path = sys.argv[1] if len(sys.argv) > 1 else "final_model_sp4096_10l_35mlp.pt"
sd = torch.load(pt_path, map_location="cpu", weights_only=True)
model_keys = set(m.state_dict().keys())
pt_keys = set(sd.keys())
print(f"Model keys: {len(model_keys)}")
print(f"PT keys: {len(pt_keys)}")
missing = model_keys - pt_keys
extra = pt_keys - model_keys
if missing: print(f"Missing from PT: {missing}")
if extra: print(f"Extra in PT: {extra}")
obj, _ = quantize_state_dict_int6(sd)
quant_keys = set(obj["quantized"].keys())
pass_keys = set(obj["passthrough"].keys())
print(f"Quantized: {len(quant_keys)} keys")
print(f"Passthrough: {len(pass_keys)} keys")
all_deq = set()
all_deq.update(quant_keys)
all_deq.update(pass_keys)
missing_after = model_keys - all_deq
if missing_after: print(f"MISSING after quant/deq: {missing_after}")
else: print("All model keys covered by quant+passthrough")
