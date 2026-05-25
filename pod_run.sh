#!/usr/bin/env bash
# Pod runner: installs deps, clones CASEOPS dataset from HF (parallel LFS),
# then fires torchrun training with the stack env vars.
#
# Usage on pod (HF_TOKEN MUST be exported first, never baked into git):
#   export HF_TOKEN=hf_...
#   bash pod_run.sh
set -euo pipefail

REPO_ROOT="/workspace/parameter-golf"
DATASET="$REPO_ROOT/data/datasets/fineweb10B_sp8192_caseops"
TOK="$DATASET/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
SHARDS="$DATASET/shards"
: "${HF_TOKEN:?HF_TOKEN must be exported before running this script}"
NPROC=8

echo "===== [1/4] install brotli + git-lfs ====="
pip install brotli -q
if ! command -v git-lfs >/dev/null 2>&1; then
    apt-get install -y git-lfs 2>/dev/null \
        || (curl -sSL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
              | tar xz -C /tmp/ \
            && /tmp/git-lfs-3.5.1/install.sh)
fi
git lfs install
git config --global lfs.concurrenttransfers 16
mkdir -p "$REPO_ROOT/data/datasets"

echo "===== [2/4] git clone CASEOPS dataset (skip if present) ====="
if [ -f "$TOK" ] && [ -n "$(ls "$SHARDS"/fineweb_train_*.bin 2>/dev/null)" ]; then
    echo "    CASEOPS dataset already present, skipping clone"
else
    cd "$REPO_ROOT/data/datasets"
    rm -rf fineweb10B_sp8192_caseops
    git clone "https://hf:${HF_TOKEN}@huggingface.co/datasets/Idan3011/parameter-golf-sp8192-caseops" \
        fineweb10B_sp8192_caseops
fi

echo "===== [3/4] verify dataset files ====="
[ -f "$TOK" ] || { echo "ERROR: caseops tokenizer missing at $TOK"; exit 1; }
[ -n "$(ls "$SHARDS"/fineweb_train_*.bin 2>/dev/null)" ] \
    || { echo "ERROR: no caseops train shards at $SHARDS"; exit 1; }
echo "    OK: caseops tokenizer + $(ls "$SHARDS"/fineweb_train_*.bin | wc -l) train shards"

echo "===== [4/4] fire training ====="
cd "$REPO_ROOT"
mkdir -p logs
LOG="logs/pod_run_$(date +%s).log"
echo "    Logging to: $LOG"

CASEOPS_ENABLED=1 \
DATA_PATH="$SHARDS" \
TOKENIZER_PATH="$TOK" \
DS_AUX_ENABLED=1 DS_AUX_LAYER=6 DS_AUX_RANK=256 \
DEQ_INNER_LAYER=2 DEQ_INNER_T=3 \
SQUARED_DIST_ATTN=1 \
ENABLE_PCM=1 PCM_K=96 \
ENABLE_SPHERE=1 \
torchrun --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee "$LOG"
