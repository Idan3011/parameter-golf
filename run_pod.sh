#!/usr/bin/env bash
set -eu

SEED=${SEED:-42}
TAG=${TAG:-run}
STAMP=$(date +%Y-%m-%d_%H%M)
RUN_TAG="seed${SEED}_${TAG}_${STAMP}"
: "${HF_TOKEN:?HF_TOKEN env var required (export it before running this script)}"
HF_REPO=${HF_REPO:-Idan3011/parameter-golf-sp8192-caseops}
RECORDS_DIR=$(cd "$(dirname "$0")" && pwd)
HF_REPO_DIR=/workspace/parameter-golf-sp8192-caseops
LOG=$RECORDS_DIR/archive/logs/ship_${RUN_TAG}.log

echo "[$(date +%H:%M:%S)] === setup ==="

if [ "${CLEAR_COMPILE_CACHE:-1}" = "1" ]; then
    rm -rf ~/.cache/torch/inductor ~/.triton/cache
fi

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

echo "shards: $(ls data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ | wc -l) files"
echo "val:    $(ls data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ | grep -c val) files"

echo "[$(date +%H:%M:%S)] === seed=${SEED} run ==="

# Background watcher: copies final_model.{pt,float.pt,int6.ptz} to tagged
# names as soon as each appears + is stable. Survives early kills (no need
# to wait for the full TTT eval to finish to get artifacts).
(
    while true; do
        for ext in pt float.pt int6.ptz; do
            src="${RECORDS_DIR}/final_model.${ext}"
            dst="${RECORDS_DIR}/final_model_${RUN_TAG}.${ext}"
            if [ -f "${src}" ] && [ ! -f "${dst}" ]; then
                size1=$(stat -c%s "${src}" 2>/dev/null || echo 0)
                sleep 2
                size2=$(stat -c%s "${src}" 2>/dev/null || echo 0)
                if [ "${size1}" = "${size2}" ] && [ "${size1}" -gt 0 ]; then
                    cp "${src}" "${dst}" 2>/dev/null \
                        && echo "[$(date +%H:%M:%S)] watcher copied: ${dst} ($(du -h ${dst} | cut -f1))"
                fi
            fi
        done
        sleep 8
    done
) &
WATCHER_PID=$!
trap "kill ${WATCHER_PID} 2>/dev/null || true" EXIT INT TERM

echo "=== caller env overrides ==="
for v in SEED TAG CAUTIOUS_MUON MUON_BACKEND_STEPS MUON_MOMENTUM_WARMUP_STEPS MUON_WD \
         LQER_HESSIAN_SCORE LQER_RANK LQER_TOP_K LQER_FACTOR_BITS \
         EVAL_FUSED_CE TTT_ENABLED TTT_LORA_RANK TTT_CHUNK_SIZE TTT_WEIGHT_DECAY TTT_BETA2 \
         PHASED_TTT_PREFIX_DOCS PHASED_TTT_NUM_PHASES \
         MIN_LR MATRIX_LR EMBED_BITS GRAD_CLIP_NORM WARMDOWN_FRAC WARMUP_STEPS \
         BETA2 MLP_CLIP_SIGMAS ATTN_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
         GATE_WINDOW SPARSE_ATTN_GATE_SCALE SMEAR_GATE_ENABLED GATED_ATTN_QUANT_GATE \
         ENABLE_LOOPING_AT LOOP_START LOOP_END NUM_LOOPS \
         FUSED_CE_ENABLED COMPRESSOR CASEOPS_ENABLED CLEAR_COMPILE_CACHE; do
    val="${!v:-<unset>}"
    echo "  ${v}=${val}"
done | tee -a "${LOG}"
echo "==========================="

env \
    DATA_DIR=./data \
    VOCAB_SIZE=8192 \
    DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    CASEOPS_ENABLED=${CASEOPS_ENABLED:-1} \
    ITERATIONS=${ITERATIONS:-20000} \
    MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600} \
    TTT_ENABLED=${TTT_ENABLED:-1} \
    PHASED_TTT_PREFIX_DOCS=${PHASED_TTT_PREFIX_DOCS:-2500} \
    PHASED_TTT_NUM_PHASES=${PHASED_TTT_NUM_PHASES:-3} \
    EMBED_BITS=${EMBED_BITS:-7} \
    MATRIX_LR=${MATRIX_LR:-0.026} \
    MIN_LR=${MIN_LR:-0.1} \
    MLP_CLIP_SIGMAS=${MLP_CLIP_SIGMAS:-11.5} \
    ATTN_CLIP_SIGMAS=${ATTN_CLIP_SIGMAS:-13.0} \
    EMBED_CLIP_SIGMAS=${EMBED_CLIP_SIGMAS:-14.0} \
    GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.3} \
    TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-48} \
    WARMUP_STEPS=${WARMUP_STEPS:-20} \
    MUON_BACKEND_STEPS=${MUON_BACKEND_STEPS:-5} \
    GLOBAL_TTT_MOMENTUM=${GLOBAL_TTT_MOMENTUM:-0.9} \
    WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.85} \
    BETA2=${BETA2:-0.99} \
    TTT_BETA2=${TTT_BETA2:-0.99} \
    TTT_WEIGHT_DECAY=${TTT_WEIGHT_DECAY:-0.5} \
    TTT_LORA_RANK=${TTT_LORA_RANK:-80} \
    SPARSE_ATTN_GATE_SCALE=${SPARSE_ATTN_GATE_SCALE:-0.5} \
    GPTQ_RESERVE_SECONDS=${GPTQ_RESERVE_SECONDS:-0.5} \
    GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-16} \
    VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0} \
    GATED_ATTN_QUANT_GATE=${GATED_ATTN_QUANT_GATE:-1} \
    SPARSE_ATTN_GATE_ENABLED=${SPARSE_ATTN_GATE_ENABLED:-1} \
    GATE_WINDOW=${GATE_WINDOW:-12} \
    SMEAR_GATE_ENABLED=${SMEAR_GATE_ENABLED:-1} \
    LQER_ENABLED=${LQER_ENABLED:-1} \
    LQER_ASYM_ENABLED=${LQER_ASYM_ENABLED:-1} \
    LQER_RANK=${LQER_RANK:-4} \
    LQER_FACTOR_BITS=${LQER_FACTOR_BITS:-4} \
    LQER_ASYM_GROUP=${LQER_ASYM_GROUP:-64} \
    LQER_TOP_K=${LQER_TOP_K:-3} \
    FUSED_CE_ENABLED=${FUSED_CE_ENABLED:-1} \
    COMPRESSOR=${COMPRESSOR:-pergroup} \
    NCCL_NET=${NCCL_NET:-Socket} \
    LQER_HESSIAN_SCORE=${LQER_HESSIAN_SCORE:-1} \
    EVAL_FUSED_CE=${EVAL_FUSED_CE:-1} \
    ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35} \
    LOOP_START=${LOOP_START:-3} \
    LOOP_END=${LOOP_END:-5} \
    NUM_LOOPS=${NUM_LOOPS:-2} \
    SANDWICH_NORM=${SANDWICH_NORM:-0} \
    SANDWICH_SCALE=${SANDWICH_SCALE:-1.0} \
    HEADWISE_GATE_ENABLED=${HEADWISE_GATE_ENABLED:-0} \
    ROPE_YARN=${ROPE_YARN:-0} \
    EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-2048} \
    SEED=${SEED} \
    torchrun --standalone --nproc_per_node=8 ${RECORDS_DIR}/train_gpt.py 2>&1 | tee ${LOG}

echo "[$(date +%H:%M:%S)] === seed=${SEED} done ==="
set +e
grep -E "final_int8|val_bpb|submission size|lqer:" ${LOG} | tail -20
# Final-pass safety net: ensure all artifacts are tagged (in case watcher missed any)
sleep 5
echo "[$(date +%H:%M:%S)] === artifact tag: ${RUN_TAG} ==="
for ext in pt float.pt int6.ptz; do
    src="${RECORDS_DIR}/final_model.${ext}"
    dst="${RECORDS_DIR}/final_model_${RUN_TAG}.${ext}"
    if [ -f "${src}" ] && [ ! -f "${dst}" ]; then
        cp "${src}" "${dst}"
        echo "  ${dst} ($(du -h ${dst} | cut -f1)) [final pass]"
    elif [ -f "${dst}" ]; then
        echo "  ${dst} ($(du -h ${dst} | cut -f1)) [already copied by watcher]"
    fi
done
set -e
