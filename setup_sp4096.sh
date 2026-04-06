#!/bin/bash
# Parameter Golf — sp4096 Custom Tokenizer Setup
# Downloads pre-tokenized FineWeb data from HuggingFace
# Run once before training. ~16GB download, takes ~5 min.

set -e
pip install brotli -q

echo "Downloading sp4096 tokenizer + data from HuggingFace..."
python3 -c "
from huggingface_hub import hf_hub_download, list_repo_tree
import os, shutil
REPO = 'idan3011/parameter-golf-sp4096'
os.makedirs('data/tokenizers', exist_ok=True)
os.makedirs('data/datasets/fineweb10B_sp4096', exist_ok=True)
files = list(list_repo_tree(REPO, repo_type='dataset', recursive=True))
for f in files:
    if not hasattr(f, 'size'): continue
    p = f.path
    if p.endswith('.model') or p.endswith('.vocab'):
        dst = 'data/' + p
    elif p.endswith('.bin'):
        dst = 'data/' + p
    else: continue
    if os.path.exists(dst): print(f'  skip {dst}'); continue
    print(f'  downloading {p}...', flush=True)
    src = hf_hub_download(REPO, p.split('/')[-1], subfolder='/'.join(p.split('/')[:-1]), repo_type='dataset')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
print('Done!')
"
echo ""
echo "Data ready. Run training with:"
echo ""
echo "  DATA_PATH=./data/datasets/fineweb10B_sp4096 \\"
echo "  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \\"
echo "  VOCAB_SIZE=4096 TIE_EMBEDDINGS=0 \\"
echo "  USE_BIGRAM=0 USE_SMEAR=0 USE_PRE_ENRICH=0 \\"
echo "  USE_GPTQ=1 WARMDOWN_FRAC=0.35 \\"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt.py"
