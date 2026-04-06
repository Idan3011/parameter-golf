#!/bin/bash
# Parameter Golf — sp4096 Custom Tokenizer Setup
# Usage: bash setup_sp4096.sh [train_shards]
# Default: all shards. For screening: bash setup_sp4096.sh 1

set -e
SHARDS=${1:-0}
pip install brotli -q

echo "Downloading sp4096 tokenizer + data from HuggingFace..."
python3 -c "
import sys
from huggingface_hub import hf_hub_download, list_repo_tree
import os, shutil
REPO = 'idan3011/parameter-golf-sp4096'
SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.makedirs('data/tokenizers', exist_ok=True)
os.makedirs('data/datasets/fineweb10B_sp4096', exist_ok=True)
files = list(list_repo_tree(REPO, repo_type='dataset', recursive=True))
count = 0
for f in files:
    if not hasattr(f, 'size'): continue
    p = f.path
    is_train = 'train' in p and p.endswith('.bin')
    is_val = 'val' in p and p.endswith('.bin')
    is_tok = p.endswith('.model') or p.endswith('.vocab')
    if not (is_train or is_val or is_tok): continue
    if is_train and SHARDS > 0 and count >= SHARDS: continue
    dst = 'data/' + p
    if os.path.exists(dst): print(f'  skip {dst}'); continue
    print(f'  downloading {p}...', flush=True)
    src = hf_hub_download(REPO, p.split('/')[-1], subfolder='/'.join(p.split('/')[:-1]), repo_type='dataset')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if is_train: count += 1
print('Done!')
" $SHARDS
echo "Data ready."
