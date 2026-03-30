import torch, os
os.environ['USE_BITNET'] = '1'
os.environ['QUANT_CLIP'] = '31'
from train_gpt import *
args = Hyperparameters()
device = torch.device('cuda')
model = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init).to(device)
sd = torch.load('final_model.pt', map_location='cpu')
model.load_state_dict(sd, strict=True)
model.eval()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
x = val_tokens[:2048].unsqueeze(0).to(device, dtype=torch.int64)
y = val_tokens[1:2049].unsqueeze(0).to(device, dtype=torch.int64)
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model.forward_logits(x)
    loss = F.cross_entropy(logits.float().reshape(-1, args.vocab_size), y.reshape(-1))
print('NO QUANT Loss: %.4f BPB: %.4f' % (loss.item(), loss.item() / 0.6931))
