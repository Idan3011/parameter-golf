import torch, os
os.environ['USE_BITNET'] = '1'
os.environ['QUANT_CLIP'] = '31'
from train_gpt import *
args = Hyperparameters()
device = torch.device('cuda')

model1 = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init).to(device)
sd = torch.load('final_model.pt', map_location='cpu')

model1_keys = set(model1.state_dict().keys())
sd_keys = set(sd.keys())
print('In model but not checkpoint:', model1_keys - sd_keys)
print('In checkpoint but not model:', sd_keys - model1_keys)

for k in sorted(sd_keys):
    ms = model1.state_dict()[k].shape if k in model1.state_dict() else 'MISSING'
    ss = sd[k].shape
    if ms != ss:
        print(f'SHAPE MISMATCH: {k} model={ms} checkpoint={ss}')

print('XSA_LAST_N env:', os.environ.get('XSA_LAST_N', 'not set'))
print('LEAKY_RELU env:', os.environ.get('LEAKY_RELU', 'not set'))
print('NUM_LAYERS:', args.num_layers)
print('MLP_MULT:', args.mlp_mult)

model1.load_state_dict(sd, strict=True)
print('load_state_dict: OK')

for i, block in enumerate(model1.blocks):
    print(f'block {i}: use_xsa={block.attn.use_xsa}')
    break
