#!/usr/bin/env bash
set -eu

CKPT=${CKPT:-final_model.int6.ptz}
STRIDE=${STRIDE:-64}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH=${BATCH:-8}
TAG=${TAG:-sliding}
STAMP=$(date +%Y-%m-%d_%H%M)
RUN_TAG="${TAG}_stride${STRIDE}_${STAMP}"
: "${HF_TOKEN:?HF_TOKEN env var required (export it before running this script)}"
HF_REPO=${HF_REPO:-Idan3011/parameter-golf-sp8192-caseops}
RECORDS_DIR=$(cd "$(dirname "$0")" && pwd)
HF_REPO_DIR=/workspace/parameter-golf-sp8192-caseops
LOG=$RECORDS_DIR/archive/logs/eval_sliding_${RUN_TAG}.log

echo "[$(date +%H:%M:%S)] === sliding eval setup ==="

apt-get update -qq
apt-get install -y -qq lrzip git-lfs
git lfs install

if [ ! -d "$HF_REPO_DIR" ]; then
    git clone https://Idan3011:${HF_TOKEN}@huggingface.co/datasets/${HF_REPO} ${HF_REPO_DIR}
fi
cd ${HF_REPO_DIR} && git lfs pull
cd ${RECORDS_DIR}

mkdir -p data/datasets data/tokenizers archive/logs

ln -sfn ${HF_REPO_DIR}/shards data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
ln -sfn ${RECORDS_DIR}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

pip install -q brotli
pip install -q -r ${RECORDS_DIR}/requirements.txt
pip install -q --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

if [ ! -f "${RECORDS_DIR}/${CKPT}" ]; then
    echo "ERROR: checkpoint ${RECORDS_DIR}/${CKPT} not found"
    ls -la ${RECORDS_DIR}/final_model_*.pt 2>/dev/null || true
    exit 1
fi

echo "[$(date +%H:%M:%S)] === sliding eval run ==="
echo "  CKPT=${CKPT}"
echo "  STRIDE=${STRIDE}"
echo "  SEQ_LEN=${SEQ_LEN}"
echo "  BATCH=${BATCH}"
echo "  RUN_TAG=${RUN_TAG}"

env \
    DATA_DIR=./data \
    VOCAB_SIZE=8192 \
    DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    CASEOPS_ENABLED=${CASEOPS_ENABLED:-1} \
    NUM_LAYERS=${NUM_LAYERS:-11} \
    MLP_MULT=${MLP_MULT:-4.0} \
    MUON_BACKEND_STEPS=${MUON_BACKEND_STEPS:-5} \
    EMBED_BITS=${EMBED_BITS:-7} \
    GATE_WINDOW=${GATE_WINDOW:-12} \
    SPARSE_ATTN_GATE_ENABLED=${SPARSE_ATTN_GATE_ENABLED:-1} \
    SPARSE_ATTN_GATE_SCALE=${SPARSE_ATTN_GATE_SCALE:-0.5} \
    SMEAR_GATE_ENABLED=${SMEAR_GATE_ENABLED:-1} \
    LQER_ENABLED=${LQER_ENABLED:-1} \
    HEADWISE_GATE_ENABLED=${HEADWISE_GATE_ENABLED:-0} \
    GATED_ATTN_QUANT_GATE=${GATED_ATTN_QUANT_GATE:-1} \
    LOOP_START=${LOOP_START:-3} \
    LOOP_END=${LOOP_END:-5} \
    NUM_LOOPS=${NUM_LOOPS:-2} \
    QK_GAIN_INIT=${QK_GAIN_INIT:-5.0} \
    torchrun --standalone --nproc_per_node=${NPROC:-8} \
        ${RECORDS_DIR}/eval_sliding.py \
        --ckpt ${RECORDS_DIR}/${CKPT} \
        --stride ${STRIDE} \
        --seq_len ${SEQ_LEN} \
        --batch ${BATCH} \
    2>&1 | tee ${LOG}

echo "[$(date +%H:%M:%S)] === sliding eval done ==="
grep "sliding_window" ${LOG} | tail -3
