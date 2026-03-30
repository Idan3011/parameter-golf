import torch, io, lzma, sys, os
os.environ['USE_BITNET'] = '1'
os.environ['QUANT_CLIP'] = '31'
from train_gpt import *
args = Hyperparameters()
device = torch.device('cuda')
model = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init).to(device)
sd = torch.load('final_model_bitnet.pt', map_location='cpu')
model.load_state_dict(sd, strict=True)
bn = set()
with torch.no_grad():
    for bi, block in enumerate(model.blocks):
        for mn, m in block.named_modules():
            if isinstance(m, CastedLinear):
                w = m.weight.data
                gamma = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
                m.weight.data = (torch.clamp(torch.round(w / gamma), -1, 1) * gamma).to(w.dtype)
                bn.add(f'blocks.{bi}.{mn}.weight')
print(f'Projected {len(bn)} layers')
obj, stats = quantize_state_dict_int6(model.state_dict(), bitnet_names=bn)
q = obj['quantized']['blocks.0.attn.c_q.weight']
print(f'q unique: {q.unique().tolist()}')
recon = dequantize_state_dict_int8(obj)
eval_model = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init).to(device)
eval_model.load_state_dict(recon, strict=True)
eval_model.eval()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
x = val_tokens[:2048].unsqueeze(0).to(device, dtype=torch.int64)
y = val_tokens[1:2049].unsqueeze(0).to(device, dtype=torch.int64)
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = F.cross_entropy(eval_model.forward_logits(x).float().reshape(-1, args.vocab_size), y.reshape(-1))
print(f'Loss: {loss.item():.4f} BPB: {loss.item() / 0.6931:.4f}')
