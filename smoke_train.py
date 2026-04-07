"""Smoke training: same code path as train_gpt.py but tiny model that fits ~6GB VRAM.

Run with:
    MAX_WALLCLOCK_SECONDS=50 MAX_TRAIN_SHARDS=1 torchrun --standalone --nproc_per_node=1 smoke_train.py

Validates the full production pipeline (download, train, GPTQ, quantize, eval, sliding, TTT)
end-to-end on a tiny model + 1 train shard. Targets ~50s training + ~30s post-training pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt as tg

# Tiny model: ~700K params, fits 6GB VRAM with room for activations + compile.
tg.Hyperparameters.num_layers = 4
tg.Hyperparameters.model_dim = 128
tg.Hyperparameters.num_heads = 4
tg.Hyperparameters.num_kv_heads = 2
tg.Hyperparameters.mlp_mult = 2.0
tg.Hyperparameters.train_batch_tokens = 16_384
tg.Hyperparameters.train_seq_len = 256
tg.Hyperparameters.val_batch_size = 16_384
tg.Hyperparameters.warmup_steps = 5
tg.Hyperparameters.iterations = 20000
tg.Hyperparameters.warmdown_iters = 3500
tg.Hyperparameters.val_loss_every = 200
tg.Hyperparameters.train_log_every = 50

# Recurrence with smaller layer indices so loop_start/loop_end are valid for 4 layers.
tg.Hyperparameters.num_loops = 1
tg.Hyperparameters.loop_start = 1
tg.Hyperparameters.loop_end = 2

# TTT smoke: keep it small.
tg.Hyperparameters.ttt_chunk_tokens = 8_192
tg.Hyperparameters.ttt_epochs = 2
tg.Hyperparameters.ttt_score_batch = 8
tg.Hyperparameters.ttt_train_batch = 4

if __name__ == "__main__":
    tg.main()
