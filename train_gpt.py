"""Hard stop: train_gpt.py must never be longer than 1500 lines."""
from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid
from pathlib import Path
import brotli, lzma, numpy as _np
_COMPRESSOR = os.environ.get("COMPRESSOR", "brotli")
_BSHF_STRIDE = int(os.environ.get("BYTE_SHUFFLE_STRIDE", "4"))
def _byte_shuffle(data: bytes) -> bytes:
    s = _BSHF_STRIDE
    arr = _np.frombuffer(data, dtype=_np.uint8)
    pad = (s - len(arr) % s) % s
    if pad: arr = _np.concatenate([arr, _np.zeros(pad, dtype=_np.uint8)])
    return bytes(arr.reshape(-1, s).T.ravel())
def _byte_unshuffle(data: bytes) -> bytes:
    s = _BSHF_STRIDE
    arr = _np.frombuffer(data, dtype=_np.uint8)
    rows = len(arr) // s
    return bytes(arr.reshape(s, rows).T.ravel())
def _compress(data: bytes) -> bytes:
    d = _byte_shuffle(data)
    if _COMPRESSOR == "lzma": return lzma.compress(d, preset=9 | lzma.PRESET_EXTREME)
    if _COMPRESSOR == "zstd":
        try: import zstandard as zstd
        except ImportError: raise RuntimeError("pip install zstandard")
        return zstd.ZstdCompressor(level=22).compress(d)
    return brotli.compress(d, quality=11)
def _decompress(data: bytes) -> bytes:
    if _COMPRESSOR == "lzma": return _byte_unshuffle(lzma.decompress(data))
    if _COMPRESSOR == "zstd":
        try: import zstandard as zstd
        except ImportError: raise RuntimeError("pip install zstandard")
        return _byte_unshuffle(zstd.ZstdDecompressor().decompress(data))
    return _byte_unshuffle(brotli.decompress(data))
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
_FP8_MLP = bool(int(os.environ.get("FP8_MLP", "1")))
_FP8_SCALE_MODE = os.environ.get("FP8_SCALE_MODE", "rowwise")
_LIGER_CE = bool(int(os.environ.get("LIGER_CE", "1")))
if _FP8_MLP:
    _FP8_MAX = 448.0
    @torch.library.custom_op("paramgolf::fp8_linear_fw", mutates_args=())
    def _fp8_linear_fw(x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        w = weight.to(dtype=x.dtype).contiguous()
        if _FP8_SCALE_MODE == "tensorwise":
            x_scale = (x2.float().abs().amax().clamp_min(1e-12) / _FP8_MAX).reshape(1)
            w_scale = (w.float().abs().amax().clamp_min(1e-12) / _FP8_MAX).reshape(1)
            sb = w_scale
        else:
            x_scale = x2.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
            w_scale = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / _FP8_MAX
            sb = w_scale.t()
        x8 = (x2.float() / x_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        w8 = (w.float() / w_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x8, w8.t(), x_scale, sb, out_dtype=torch.bfloat16, use_fast_accum=True)
        return out.reshape(*orig_shape[:-1], w.shape[0]), x8, w8, x_scale, w_scale

    @_fp8_linear_fw.register_fake
    def _fp8_linear_fw_fake(x, weight):
        N_out, K_in = weight.shape[0], weight.shape[1]
        BxS = 1
        for d in x.shape[:-1]: BxS *= d
        sh = (1,) if _FP8_SCALE_MODE == "tensorwise" else None
        return (x.new_empty(x.shape[:-1] + (N_out,), dtype=torch.bfloat16),
                x.new_empty((BxS, K_in), dtype=torch.float8_e4m3fn),
                x.new_empty((N_out, K_in), dtype=torch.float8_e4m3fn),
                x.new_empty(sh or (BxS, 1), dtype=torch.float32),
                x.new_empty(sh or (N_out, 1), dtype=torch.float32))
    def _fp8_setup_ctx(ctx, inputs, output):
        _out, x8, w8, xs, ws = output
        ctx.save_for_backward(x8, w8, xs, ws)
        ctx.orig_shape = inputs[0].shape
    def _fp8_bwd(ctx, grad_out, *_):
        x8, w8, xs, ws = ctx.saved_tensors
        w_bf = (w8.float() * ws).to(torch.bfloat16)
        x_bf = (x8.float() * xs).to(torch.bfloat16)
        g2 = grad_out.reshape(-1, grad_out.shape[-1]).to(torch.bfloat16)
        return (g2 @ w_bf).reshape(ctx.orig_shape), g2.t() @ x_bf
    _fp8_linear_fw.register_autograd(_fp8_bwd, setup_context=_fp8_setup_ctx)
    def _fp8_linear(x: Tensor, weight: Tensor) -> Tensor:
        return torch.ops.paramgolf.fp8_linear_fw(x, weight)[0]
if _LIGER_CE:
    import torch.distributed.tensor
    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
    @torch.library.custom_op("paramgolf::liger_fused_ce_fw", mutates_args=())
    def _liger_fused_ce_fw(x: Tensor, w: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_rg = x.detach().requires_grad_(True)
        w_rg = w.detach().requires_grad_(True)
        loss, _z, _acc, gx, gw, _gb = fused_linear_cross_entropy_forward(_input=x_rg, weight=w_rg, target=t)
        if gx is None: gx = torch.zeros_like(x)
        if gw is None: gw = torch.zeros_like(w)
        return loss, gx, gw
    @_liger_fused_ce_fw.register_fake
    def _liger_fused_ce_fw_fake(x, w, t):
        return x.new_empty((), dtype=torch.float32), torch.empty_like(x), torch.empty_like(w)
    def _liger_setup_ctx(ctx, inputs, output):
        _loss, gx, gw = output
        ctx.save_for_backward(gx, gw)
    def _liger_bwd(ctx, grad_loss, *_):
        gx, gw = ctx.saved_tensors
        return gx * grad_loss, gw * grad_loss, None

    _liger_fused_ce_fw.register_autograd(_liger_bwd, setup_context=_liger_setup_ctx)
try:
    from flash_attn_interface import flash_attn_func
    HAS_FA3 = True
except ImportError:
    try:
        from flash_attn import flash_attn_func
        HAS_FA3 = True
    except ImportError:
        HAS_FA3 = False
from torch.nn.parallel import DistributedDataParallel as DDP
_YOU_NS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]
_USE_YOU = bool(int(os.environ.get("USE_YOU_COEFFS", "0")))

class Hyperparameters:
    _vs = int(os.environ.get("VOCAB_SIZE", "9000"))
    data_path = os.environ.get("DATA_PATH", f"./data/datasets/fineweb10B_sp{_vs}")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", f"./data/tokenizers/fineweb_{_vs}_bpe.model")
    run_id = str(uuid.uuid4())
    seed = 1337
    val_batch_size = 524_288
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "1000"))
    train_log_every = 200
    iterations = 20000
    warmdown_iters = 3500
    warmup_steps = 20
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "786432"))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "2048"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = 5.25
    vocab_size = _vs
    num_layers = int(os.environ.get("NUM_LAYERS", "12"))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "4"))
    model_dim = int(os.environ.get("MODEL_DIM", "512"))
    num_heads = int(os.environ.get("NUM_HEADS", "8"))
    mlp_mult = float(os.environ.get("MLP_MULT", "3.5"))
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
    embed_lr = 0.6
    head_lr = 0.008
    tied_embed_lr = 0.035
    tied_embed_init_std = 0.005
    matrix_lr = 0.022
    scalar_lr = float(os.environ.get("SCALAR_LR_VALUE", "0.025"))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM_VALUE", "0.99"))
    muon_backend_steps = 5 if _USE_YOU else 4
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 1500
    beta1 = 0.9
    beta2 = 0.95
    adam_eps = 1e-8
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "0.3"))
    muon_wd = 0.095
    adam_wd = 0.095
    ema_decay = float(os.environ.get("EMA_DECAY_VALUE", "0.9965"))
    num_loops = int(os.environ.get("NUM_LOOPS", "2"))
    loop_start = int(os.environ.get("LOOP_START", "4"))
    loop_end = int(os.environ.get("LOOP_END", "5"))
    skip_ttt = bool(int(os.environ.get("SKIP_TTT", "0")))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", "0.25"))
    ttt_chunk_tokens = 32768
    ttt_lr = 0.01
    ttt_epochs = 3
    ttt_freeze_blocks = 0
    ttt_score_batch = 64
    ttt_train_batch = 32
    ttt_grad_clip = 1.0
    eval_hash_buckets = 16384
    eval_hash_lr_mult = 10.0

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 4, eps: float = 1e-7) -> Tensor:
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    if _USE_YOU:
        X /= X.norm() + eps
        for t in range(steps):
            a, b, c = _YOU_NS[t]
            A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    else:
        A = X @ X.T; s = (A.abs().sum(dim=-1) + eps).rsqrt(); X = s.unsqueeze(-1) * X
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

_H5_ASYNC_AR = bool(int(os.environ.get("H5_ASYNC_AR", "1"))) 
_H5_BUCKET_MB = float(os.environ.get("H5_BUCKET_MB", "10"))
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))
        self._updates_flat: Tensor | None = None
        self._param_views: list[Tensor] | None = None
        self._lr_scales: list[float] | None = None
        self._bucket_boundaries: list[tuple[int, int]] | None = None
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps = group["lr"], group["momentum"], group["backend_steps"]
            nesterov, wd = group["nesterov"], group["wd"]
            if self._updates_flat is None:
                total = sum(int(p.numel()) for p in params)
                self._updates_flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
                views, c = [], 0
                for p in params:
                    views.append(self._updates_flat[c : c + p.numel()].view_as(p))
                    c += p.numel()
                self._param_views = views
                if _H5_ASYNC_AR and distributed:
                    bucket_elems = max(1, int(_H5_BUCKET_MB * 1024 * 1024 / 2))
                    boundaries, bstart, cpos = [], 0, 0
                    for p in params:
                        cpos += p.numel()
                        if cpos - bstart >= bucket_elems:
                            boundaries.append((bstart, cpos)); bstart = cpos
                    if bstart < cpos: boundaries.append((bstart, cpos))
                    self._bucket_boundaries = boundaries
            self._updates_flat.zero_()
            use_async_ar = _H5_ASYNC_AR and distributed and self._bucket_boundaries is not None
            ar_handles: list = []
            next_bucket_idx = 0
            curr = 0
            for i, p in enumerate(params):
                param_end = curr + p.numel()
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    if self._lr_scales is not None: g *= self._lr_scales[i]
                    self._updates_flat[curr : param_end] = g.reshape(-1)
                curr = param_end
                if use_async_ar:
                    while (next_bucket_idx < len(self._bucket_boundaries)
                           and self._bucket_boundaries[next_bucket_idx][1] <= curr):
                        bstart, bend = self._bucket_boundaries[next_bucket_idx]
                        ar_handles.append(dist.all_reduce(self._updates_flat[bstart:bend], op=dist.ReduceOp.SUM, async_op=True))
                        next_bucket_idx += 1
            if use_async_ar:
                for h in ar_handles: h.wait()
            elif distributed:
                dist.all_reduce(self._updates_flat, op=dist.ReduceOp.SUM)
            if wd > 0: torch._foreach_mul_(params, 1.0 - lr * wd)
            torch._foreach_add_(params, self._param_views, alpha=-lr)
        return loss

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device,
             grad_accum_steps: int, val_tokens: Tensor, base_bytes_lut: Tensor,
             has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

def eval_val_ttt(args, base_model, rank, world_size, device, val_tokens,
                 base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride=64, log_fn=None):
    S = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    chunk_tok, ttt_lr, ttt_epochs = args.ttt_chunk_tokens, args.ttt_lr, args.ttt_epochs
    freeze_n, score_bs, train_bs = args.ttt_freeze_blocks, args.ttt_score_batch, args.ttt_train_batch
    ttt_grad_clip = args.ttt_grad_clip
    distributed = dist.is_available() and dist.is_initialized()
    num_chunks = max(1, (total_tokens + chunk_tok - 1) // chunk_tok)
    all_windows: list[list[tuple[int, int]]] = [[] for _ in range(num_chunks)]
    ws = 0
    while ws + S <= total_tokens:
        ss = 0 if ws == 0 else S - stride
        ci = min((ws + ss) // chunk_tok, num_chunks - 1)
        all_windows[ci].append((ws, ss))
        ws += stride
    L, T, B = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))
    _fr = lambda n: any(f"blocks.{bi}." in n for bi in range(min(freeze_n, len(base_model.blocks))))
    base_params = [p for n, p in base_model.named_parameters() if not _fr(n) and "eval_hash_emb" not in n]
    hash_params = list(base_model.eval_hash_emb.parameters()) if base_model.eval_hash_emb is not None else []
    param_groups = [{"params": base_params, "lr": ttt_lr}]
    if hash_params:
        param_groups.append({"params": hash_params, "lr": ttt_lr * args.eval_hash_lr_mult})
    ttt_params = base_params + hash_params
    opt = torch.optim.SGD(param_groups, momentum=0.9)
    for ci in range(num_chunks):
        windows = all_windows[ci]
        if not windows: continue
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]
        with torch.no_grad():
            for bi in range(0, len(my_windows), score_bs):
                bw = my_windows[bi:bi + score_bs]
                x = torch.stack([val_tokens[w:w+S] for w, _ in bw]).to(device=device, dtype=torch.int64)
                y = torch.stack([val_tokens[w+1:w+S+1] for w, _ in bw]).to(device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model(x)
                ptl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").reshape(len(bw), S)
                for j, (_, ss) in enumerate(bw):
                    sl = ptl[j, ss:]; L += sl.to(torch.float64).sum(); T += float(sl.numel())
                    sp, st = x[j, ss:], y[j, ss:]
                    B += (base_bytes_lut[st].to(torch.int16) + (has_leading_space_lut[st] & ~is_boundary_token_lut[sp]).to(torch.int16)).to(torch.float64).sum()
        if ci < num_chunks - 1 and ttt_epochs > 0:
            chunk_start = ci * chunk_tok
            chunk_end = min((ci + 1) * chunk_tok, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // S
            if chunk_seqs > 0:
                cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in opt.param_groups: pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                for _ in range(ttt_epochs):
                    for bs in range(my_seq_s, my_seq_e, train_bs):
                        be = min(bs + train_bs, my_seq_e)
                        start_tok = chunk_start + bs * S
                        end_tok = chunk_start + be * S + 1
                        if end_tok > val_tokens.numel(): continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, S)
                        y = local[1:].reshape(-1, S)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if distributed:
                            for p in ttt_params:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, ttt_grad_clip)
                        opt.step()
        if log_fn and ci % 10 == 0: log_fn(f"ttt: chunk={ci}/{num_chunks}")
    if distributed:
        for t in (L, T, B): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((L / (B * math.log(2.0))).item())

def eval_val_sliding(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device,
                     val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
                     is_boundary_token_lut: Tensor, stride: int = 64, batch_size: int = 256) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel()
    windows: list[tuple[int, int]] = []
    pos = 0
    while pos + seq_len < total_tokens:
        windows.append((pos, 0 if pos == 0 else seq_len - stride))
        pos += stride
    my_windows = windows[rank::world_size]
    total_loss_sum, total_scored_tokens, total_byte_count = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(my_windows), batch_size):
            batch_windows = my_windows[batch_start:batch_start + batch_size]
            x_list, y_list = [], []
            for win_start, _ in batch_windows:
                chunk = val_tokens[win_start:win_start + seq_len + 1]
                x_list.append(chunk[:-1]); y_list.append(chunk[1:])
            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x)
            per_token_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none",
            ).reshape(len(batch_windows), seq_len)
            for idx, (win_start, score_start) in enumerate(batch_windows):
                scored_loss = per_token_loss[idx, score_start:]
                total_loss_sum += scored_loss.to(torch.float64).sum()
                total_scored_tokens += float(scored_loss.numel())
                scored_prev = x[idx, score_start:]
                scored_tgt = y[idx, score_start:]
                token_bytes = base_bytes_lut[scored_tgt].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[scored_tgt] & ~is_boundary_token_lut[scored_prev]).to(dtype=torch.int16)
                total_byte_count += token_bytes.to(torch.float64).sum()
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_scored_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_byte_count, op=dist.ReduceOp.SUM)
    val_loss = (total_loss_sum / total_scored_tokens).item()
    bpb = (total_loss_sum / (total_byte_count * math.log(2.0))).item()
    model.train()
    return float(val_loss), float(bpb)
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weights", "skip_gates",
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor_int6(t: Tensor, bits: int = 6, sd_k: float = 12.85) -> tuple[Tensor, Tensor]:
    max_val = (1 << (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        row_std = t32.std(dim=1, keepdim=True).clamp_min(1e-12)
        clip = sd_k * row_std
        clipped = t32.clamp(min=-clip, max=clip)
        scale = (clip / float(max_val)).squeeze(1)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val)
        return q.to(torch.int8).contiguous(), scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(t32.abs().amax().item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max_val) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(t32 / scale), -max_val, max_val).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int6(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        if "tok_emb.weight" in name:
            bits = 8
            sd_k = 20.0
        else:
            bits = 5
            sd_k = 12.85
        q, s = quantize_float_tensor_int6(t, bits=bits, sd_k=sd_k)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {"__quant_format__": "int6_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

def generate_ar_calibration(model: nn.Module, device: torch.device, vocab_size: int = 1024,
                            seed: int = 42) -> list[Tensor]:
    n_seqs = 16
    seq_len = 512
    batch_size = min(8, n_seqs)
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_seqs: list[Tensor] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, n_seqs, batch_size):
            bs = min(batch_size, n_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits = model(tokens)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                tokens = torch.cat([tokens, torch.multinomial(probs, 1, generator=rng)], dim=1)
            all_seqs.append(tokens)
    return all_seqs

def collect_hessians(model: nn.Module, calib_seqs: list[Tensor], device: torch.device) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.ndim == 2:
            pname = name + ".weight"
            cols = module.weight.shape[1]
            hessians[pname] = torch.zeros(cols, cols, dtype=torch.float32, device=device)
            def make_hook(pn: str):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pn] += x.T @ x
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(pname)))
    if getattr(model, "tie_embeddings", False) and hasattr(model, "tok_emb") and hasattr(model, "final_norm"):
        emb_cols = model.tok_emb.weight.shape[1]
        hessians["tok_emb.weight"] = torch.zeros(emb_cols, emb_cols, dtype=torch.float32, device=device)
        def _emb_hook(mod, inp, out):
            x = out.detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            hessians["tok_emb.weight"] += x.T @ x
        hooks.append(model.final_norm.register_forward_hook(_emb_hook))
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in calib_seqs:
            x, y = seq[:, :-1].to(device), seq[:, 1:].to(device)
            model(x, y)
    for h in hooks:
        h.remove()
    for pname in hessians:
        hessians[pname] /= len(calib_seqs)
    return hessians

def gptq_quantize_weight(weight: Tensor, hessian: Tensor, clip_range: int = 31,
                         block_size: int = 128, scale_override: Tensor | None = None) -> Tensor:
    rows, cols = weight.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = weight.float()[:, perm].clone()
    H = H[perm][:, perm]
    for damp_scale in [0.01, 0.03, 0.1, 0.3, 1.0]:
        try:
            Hd = H.clone()
            Hd.diagonal().add_(damp_scale * torch.mean(torch.diag(H)).clamp_min(1e-6))
            Hinv = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(Hinv)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except torch._C._LinAlgError:
            if damp_scale >= 1.0:
                return weight
    if scale_override is not None:
        sf = scale_override.to(device=W.device)
    else:
        sf = (weight.float().abs().amax(dim=1).clamp_min(1e-12) / clip_range).to(device=W.device)
    Q = torch.zeros(rows, cols, dtype=torch.float32, device=W.device)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros(rows, i2 - i1, device=W.device)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i].clamp_min(1e-10)
            q = torch.clamp(torch.round(w / sf), -clip_range, clip_range)
            Q[:, i1 + i] = q
            err = (w - q * sf) / d
            if i + 1 < i2 - i1:
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)
            Err1[:, i] = err
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Q = Q[:, inv_perm]
    return (Q * sf[:, None]).to(dtype=weight.dtype)
GPTQ_SD_K = float(os.environ.get("GPTQ_SD_K", "15.0"))
GPTQ_CR = int(os.environ.get("GPTQ_CR", "31"))
def _get_quant_config(name):
    if "attn.c_k.weight" in name or name.endswith("blocks.0.mlp.fc.weight") or name.endswith("blocks.10.mlp.proj.weight"):
        return GPTQ_CR, float(os.environ.get("OUTLIER_K", "20.0"))
    return GPTQ_CR, GPTQ_SD_K
def apply_gptq_sdclip_inplace(model: nn.Module, device: torch.device, args, log_fn=print, calib_override=None) -> dict[str, Tensor]:
    """GPTQ with SD-Clip scale: sf = k * std(row) / cr. Returns (scales, crs)."""
    t0 = time.perf_counter()
    if calib_override is not None:
        calib = calib_override
        log_fn(f"gptq: using provided calibration ({len(calib)} seqs)")
    else:
        log_fn("gptq: generating AR calibration data...")
        calib = generate_ar_calibration(model, device, vocab_size=args.vocab_size, seed=args.seed)
        log_fn(f"gptq: AR gen done in {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    log_fn("gptq: collecting Hessians...")
    hessians = collect_hessians(model, calib, device)
    del calib
    torch.cuda.empty_cache()
    log_fn(f"gptq: Hessians for {len(hessians)} layers in {time.perf_counter() - t1:.1f}s")
    t2 = time.perf_counter()
    count = 0
    gptq_scales: dict[str, Tensor] = {}
    gptq_crs: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.ndim == 2:
            pname = name + ".weight"
            H = hessians.get(pname)
            if H is None:
                continue
            cr, k = _get_quant_config(pname)
            with torch.no_grad():
                sf = (k * module.weight.data.float().std(dim=1).clamp_min(1e-12) / cr).to(device=module.weight.device)
                gptq_scales[pname] = sf.cpu()
                gptq_crs[pname] = cr
                module.weight.data.copy_(gptq_quantize_weight(module.weight.data, H, clip_range=cr, scale_override=sf))
            count += 1
    if "tok_emb.weight" in hessians and hasattr(model, "tok_emb"):
        emb_bits = int(os.environ.get("EMB_BITS", "8"))
        emb_k = float(os.environ.get("EMB_CLIP_K", "12.85"))
        emb_cr = (2 ** (emb_bits - 1)) - 1
        H_emb = hessians["tok_emb.weight"]
        with torch.no_grad():
            w = model.tok_emb.weight.data
            sf_emb = (emb_k * w.float().std(dim=1).clamp_min(1e-12) / emb_cr).to(device=w.device)
            gptq_scales["tok_emb.weight"] = sf_emb.cpu()
            gptq_crs["tok_emb.weight"] = emb_cr
            model.tok_emb.weight.data.copy_(gptq_quantize_weight(w, H_emb, clip_range=emb_cr, scale_override=sf_emb))
        count += 1
    log_fn(f"gptq: pass1 quantized {count} layers in {time.perf_counter() - t2:.1f}s, total {time.perf_counter() - t0:.1f}s")
    if bool(int(os.environ.get("TWO_PASS_GPTQ", "0"))):
        log_fn("gptq: pass2 — re-collecting Hessians from quantized model...")
        t3 = time.perf_counter()
        hessians2 = collect_hessians(model, generate_ar_calibration(model, device, vocab_size=args.vocab_size, seed=args.seed), device)
        log_fn(f"gptq: pass2 Hessians in {time.perf_counter() - t3:.1f}s")
        t4 = time.perf_counter()
        for name, module in model.named_modules():
            if isinstance(module, CastedLinear) and module.weight.ndim == 2:
                pname = name + ".weight"
                H2 = hessians2.get(pname)
                if H2 is None: continue
                cr, k = _get_quant_config(pname)
                try:
                    sf = (k * module.weight.data.float().std(dim=1).clamp_min(1e-12) / cr).to(device=module.weight.device)
                    gptq_scales[pname] = sf.cpu()
                    module.weight.data.copy_(gptq_quantize_weight(module.weight.data, H2, clip_range=cr, scale_override=sf))
                except torch._C._LinAlgError:
                    log_fn(f"gptq: pass2 LinAlgError on {name}, keeping pass1 result")
        log_fn(f"gptq: pass2 re-quantized in {time.perf_counter() - t4:.1f}s, total {time.perf_counter() - t0:.1f}s")
    return gptq_scales, gptq_crs

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int | None = None):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims is not None else dim
        if self.rope_dims % 2 != 0:
            raise ValueError("rope_dims must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, :, None, :].to(dtype=dtype)
        sin = freqs.sin()[None, :, None, :].to(dtype=dtype)
        return cos, sin

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rope_half = cos.size(-1)
    rope_dims = rope_half * 2
    head_dim = x.size(-1)
    if rope_dims >= head_dim:
        x1, x2 = x[..., :rope_half], x[..., rope_half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1, x2 = x_rot[..., :rope_half], x_rot[..., rope_half:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, qk_gain_init: float, use_xsa: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.use_xsa = use_xsa

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = flash_attn_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads)
            ).transpose(1, 2)
        if self.use_xsa:
            hpk = self.num_heads // self.num_kv_heads
            y_kv = y.unflatten(2, (self.num_kv_heads, hpk))
            vn = F.normalize(v, dim=-1)[:, :, :, None, :]
            proj = (y_kv * vn).sum(dim=-1, keepdim=True)
            y = (y_kv - proj * vn).flatten(2, 3)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y), v

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, leaky: bool = True):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self._leaky = leaky

    def forward(self, x: Tensor) -> Tensor:
        if _FP8_MLP:
            fc_out = _fp8_linear(x, self.fc.weight)
        else:
            fc_out = self.fc(x)
        x = F.leaky_relu(fc_out, 0.5) if self._leaky else torch.relu(fc_out)
        x_sq = x.square()
        if _FP8_MLP:
            return _fp8_linear(x_sq, self.proj.weight)
        return self.proj(x_sq)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 qk_gain_init: float, use_xsa: bool = False, leaky: bool = True,
                 layer_idx: int = 0, parallel_residual: bool = False):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.layer_idx = layer_idx
        self._ln_pre_scale = 1.0
        self._ln_post_scale = 1.0
        if bool(int(os.environ.get("LN_SCALE", "0"))):
            _s = 1.0 / math.sqrt(layer_idx + 1)
            if os.environ.get("LN_SCALE_MODE", "pre") == "post":
                self._ln_post_scale = _s
            else:
                self._ln_pre_scale = _s
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult, leaky=leaky)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, cos: Tensor, sin: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mix_a, mix_b = self.resid_mix.to(dtype=x.dtype).unbind(0)
        x = mix_a * x + mix_b * x0
        pre = self._ln_pre_scale
        post = self._ln_post_scale
        if self.parallel_residual:
            attn_out, v = self.attn(pre * self.attn_norm(x), cos, sin, v0)
            mlp_out = self.mlp(pre * self.mlp_norm(x))
            x = x + post * self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out + post * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        else:
            attn_out, v = self.attn(pre * self.attn_norm(x), cos, sin, v0)
            x = x + post * self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + post * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(pre * self.mlp_norm(x))
        return x, v

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: float, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float,
                 rope_base: float, qk_gain_init: float, num_loops: int = 2, loop_start: int = 4, loop_end: int = 5):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        # Build static effective schedule for loops loop_start..loop_end repeated (num_loops+1) times.
        if num_loops > 0:
            loop_seg = list(range(loop_start, loop_end + 1))
            all_indices = list(range(loop_start))
            for _ in range(num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(loop_end + 1, num_layers))
        else:
            all_indices = list(range(num_layers))
        self.effective_indices = all_indices
        self.num_encoder_layers = len(all_indices) // 2
        self.num_decoder_layers = len(all_indices) - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self._looped_enc = all_indices[: self.num_encoder_layers]
        self._looped_dec = all_indices[self.num_encoder_layers :]
        noloop = list(range(num_layers))
        self._noloop_enc = noloop[: len(noloop) // 2]
        self._noloop_dec = noloop[len(noloop) // 2 :]
        self.encoder_indices = self._looped_enc
        self.decoder_indices = self._looped_dec
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        _head_dim = model_dim // num_heads
        _rope_dims_env = int(os.environ.get("ROPE_DIMS", "0"))
        _rope_dims = _rope_dims_env if 0 < _rope_dims_env < _head_dim else None
        self.rotary = Rotary(_head_dim, base=rope_base, rope_dims=_rope_dims)
        parallel_start = 7
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, qk_gain_init, use_xsa=True, leaky=True, layer_idx=i, parallel_residual=(i >= parallel_start)) for i in range(num_layers)])
        self.final_norm = RMSNorm()
        self.eval_hash_emb: nn.Embedding | None = None
        self.eval_hash_multiplier = 2039
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
            with torch.no_grad():
                U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
                target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
                self.tok_emb.weight.data = (U * target_S[None, :]) @ V
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_blocks(self, x: Tensor, x0: Tensor) -> Tensor:
        cos, sin = self.rotary(x.size(1), x.device, x.dtype)
        v0 = None
        skips: list[Tensor] = []
        for block_idx in self.encoder_indices:
            x, v = self.blocks[block_idx](x, x0, cos, sin, v0)
            if v0 is None:
                v0 = v
            skips.append(x)
        for dec_step, block_idx in enumerate(self.decoder_indices):
            if dec_step < self.num_skip_weights and skips:
                skip = self.skip_weights[dec_step].to(dtype=x.dtype)[None, None, :] * skips.pop()
                gate = torch.sigmoid(self.skip_gates[dec_step]).to(dtype=x.dtype)[None, None, :]
                x = x + gate * skip
            x, _ = self.blocks[block_idx](x, x0, cos, sin, v0)
        return x

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.eval_hash_emb is not None:
            prev = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
            hash_ids = (prev.long() * self.eval_hash_multiplier + input_ids.long()) % self.eval_hash_emb.num_embeddings
            x = x + self.eval_hash_emb(hash_ids).to(dtype=x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = self._run_blocks(x, x)
        x = self.final_norm(x)
        if target_ids is not None and _LIGER_CE:
            weight = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
            loss, _, _ = torch.ops.paramgolf.liger_fused_ce_fw(
                x.reshape(-1, x.size(-1)),
                weight,
                target_ids.reshape(-1),
            )
            return loss
        logits = self._compute_logits(x)
        if target_ids is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")

def _ensure_sp_data(vocab_size: int) -> None:
    tok_path = f"./data/tokenizers/fineweb_{vocab_size}_bpe.model"
    data_dir = f"./data/datasets/fineweb10B_sp{vocab_size}"
    if os.path.exists(tok_path) and len(glob.glob(f"{data_dir}/fineweb_train_*.bin")) > 0:
        return
    from huggingface_hub import snapshot_download
    import shutil
    REPO = f"Idan3011/parameter-golf-sp{vocab_size}"
    print(f"  downloading sp{vocab_size} data from HF ({REPO})...", flush=True)
    cache_dir = snapshot_download(REPO, repo_type="dataset", allow_patterns=["*.bin", "*.model", "*.vocab"])
    os.makedirs("data/tokenizers", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for root, _, files in os.walk(cache_dir):
        for fname in files:
            if not (fname.endswith(".bin") or fname.endswith(".model") or fname.endswith(".vocab")):
                continue
            src = os.path.join(root, fname)
            if "tokenizers" in root:
                dst = os.path.join("data/tokenizers", fname)
            elif "datasets" in root:
                dst = os.path.join(data_dir, fname)
            else:
                continue
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {fname}", flush=True)

def main() -> None:
    global zeropower_via_newtonschulz5
    if int(os.environ.get("RANK", "0")) == 0:
        _ensure_sp_data(int(os.environ.get("VOCAB_SIZE", "9000")))
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(True); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    log_fh = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        log_fh = open(logfile, "a", encoding="utf-8")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
    log0(code, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"data:{dataset_dir.name} train_shards:{actual_train_files} val_tokens:{val_tokens.numel()-1}")
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        num_loops=args.num_loops, loop_start=args.loop_start, loop_end=args.loop_end,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    if _FP8_MLP:
        log0(f"fp8_mlp: enabled — MLP fc/proj will use torch._scaled_mm ({_FP8_SCALE_MODE} scaling). Hopper sm_90+ required for rowwise.")
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    _depth_lr = bool(int(os.environ.get("DEPTH_LR", "0")))
    _rec_idx = {str(i) for i in range(args.loop_start, args.loop_end + 1)}
    def _is_rec(n): return n.split(".", 1)[0] in _rec_idx
    matrix_params = [p for name, p in block_named_params if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
    if _depth_lr:
        _rec_matrix = [p for name, p in block_named_params if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS) and _is_rec(name)]
        _nonrec_matrix = [p for p in matrix_params if not any(p is r for r in _rec_matrix)]
    scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.skip_gates.numel() > 0:
        scalar_params.append(base_model.skip_gates)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    _telr = os.environ.get("TIED_EMBED_LR_VALUE", "")
    if _telr:
        token_lr = float(_telr)
        log0(f"tied_embed_lr_override: lr={token_lr:.4f} (was {args.tied_embed_lr})")
    optimizer_tok = torch.optim.AdamW([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    _mlr = float(os.environ.get("MATRIX_LR_VALUE", args.matrix_lr))
    if _mlr != args.matrix_lr: log0(f"matrix_lr_override: lr={_mlr:.4f} (was {args.matrix_lr})")
    optimizer_muon = Muon(matrix_params, lr=_mlr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in optimizer_muon.param_groups: g["base_lr"] = _mlr
    if _depth_lr:
        _rec_set = set(id(p) for p in _rec_matrix); _dlf = float(os.environ.get("DEPTH_LR_FACTOR", "0.85"))
        optimizer_muon._lr_scales = [_dlf if id(p) in _rec_set else 1.0 for p in matrix_params]
        log0(f"depth_lr: {sum(1 for s in optimizer_muon._lr_scales if s < 1)} recurrent params at {_dlf}x")
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum:{grad_accum_steps} FA3:{HAS_FA3}")
    log0(f"effective_schedule len:{len(base_model.effective_indices)} idx:{base_model.effective_indices}")
    log0(f"batch:{args.train_batch_tokens} seq:{args.train_seq_len} warmup:{args.warmup_steps} wallclock:{args.max_wallclock_seconds:.0f}s")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        remaining_frac = remaining_ms / max(max_wallclock_ms, 1.0)
        warmdown_frac = 0.72
        return min(remaining_frac / warmdown_frac, 1.0) if remaining_frac < warmdown_frac else 1.0
    _skip_train = bool(int(os.environ.get("SKIP_TRAIN", "0")))
    _load_float_pt = os.environ.get("LOAD_FLOAT_PT", "")
    if _skip_train:
        if not _load_float_pt:
            raise ValueError("SKIP_TRAIN=1 requires LOAD_FLOAT_PT=<path>")
        log0(f"SKIP_TRAIN: loading float weights from {_load_float_pt}")
        base_model.load_state_dict(torch.load(_load_float_pt, map_location=device, weights_only=True), strict=True)
        for module in base_model.modules():
            if isinstance(module, CastedLinear): module.float()
        restore_low_dim_params_to_fp32(base_model)
    _prewarm_enabled = bool(int(os.environ.get("PREWARM", "1")))
    if not _skip_train and args.enable_looping_at > 0 and args.num_loops > 0:
        base_model.encoder_indices = base_model._noloop_enc
        base_model.decoder_indices = base_model._noloop_dec
    if not _skip_train and args.enable_looping_at > 0 and args.num_loops > 0 and _prewarm_enabled:
        log0("pre-warming looped compile graph (forward+backward+opt.step, 2 iters)...")
        _t_warm = time.perf_counter()
        _pw_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        _pw_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        base_model.encoder_indices = base_model._looped_enc
        base_model.decoder_indices = base_model._looped_dec
        model.train()
        for _pw_iter in range(2):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                _pw_x, _pw_y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    _pw_loss = model(_pw_x, _pw_y)
                (_pw_loss * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
        base_model.load_state_dict(_pw_model_state, strict=True)
        for opt, state in zip(optimizers, _pw_opt_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        base_model.encoder_indices = base_model._noloop_enc
        base_model.decoder_indices = base_model._noloop_dec
        log0(f"looped pre-warm done in {time.perf_counter()-_t_warm:.1f}s")
    if not _skip_train and args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_time_ms = 0.0
    stop_after_step: int | None = None
    ema_state = {k: v.detach().float().clone() for k, v in base_model.state_dict().items()}
    _ema_keys = list(ema_state.keys())
    _ema_orig_dtypes = {k: v.dtype for k, v in base_model.state_dict().items()}
    _ema_param_map = dict(base_model.named_parameters())
    _ema_model_refs = [_ema_param_map[k] for k in _ema_keys]
    _ema_state_refs = [ema_state[k] for k in _ema_keys]
    _swa_gpu: list[Tensor] | None = None
    swa_count = 0
    _reached_cap_t = torch.zeros(1, dtype=torch.int32, device=device)
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        if _skip_train: break
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.enable_looping_at > 0 and base_model.encoder_indices is base_model._noloop_enc:
            frac_done = elapsed_ms / max(max_wallclock_ms, 1.0) if max_wallclock_ms else step / args.iterations
            if frac_done >= args.enable_looping_at:
                base_model.encoder_indices = base_model._looped_enc
                base_model.decoder_indices = base_model._looped_dec
                log0(f"loop_activated step:{step} frac:{frac_done:.3f}")
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss.zero_()
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        _mom_wc = bool(int(os.environ.get("MOM_WARMUP_WC", "1")))
        if _mom_wc and max_wallclock_ms is not None:
            _mom_wc_frac = float(os.environ.get("MOM_WARMUP_WC_FRAC", "0.25"))
            frac = min((training_time_ms / max_wallclock_ms) / _mom_wc_frac, 1.0) if _mom_wc_frac > 0 else 1.0
        else:
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1
        _ema_d = min(args.ema_decay, step / (step + 10)) if bool(int(os.environ.get("DYNAMIC_EMA", "0"))) else args.ema_decay
        with torch.no_grad():
            torch._foreach_mul_(_ema_state_refs, _ema_d)
            _ema_model_fp32 = [p.float() if p.dtype != torch.float32 else p for p in _ema_model_refs]
            torch._foreach_add_(_ema_state_refs, _ema_model_fp32, alpha=1.0 - _ema_d)
            if scale < 0.5 and step % 5 == 0:
                refs_fp32 = [p.float() if p.dtype != torch.float32 else p for p in _ema_model_refs]
                if _swa_gpu is None:
                    _swa_gpu = [t.detach().clone() for t in refs_fp32]
                else:
                    torch._foreach_add_(_swa_gpu, refs_fp32)
                swa_count += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None and approx_training_time_ms > max_wallclock_ms * 0.88:
            _reached_cap_t.fill_(int(reached_cap))
            dist.all_reduce(_reached_cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(_reached_cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    if not _skip_train:
        log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
        ema_state = {k: v.cpu() for k, v in ema_state.items()}
        if _swa_gpu is not None and swa_count > 0:
            log0(f"swa: averaging {swa_count} checkpoints on top of EMA")
            torch._foreach_div_(_swa_gpu, float(swa_count))
            for i, k in enumerate(_ema_keys):
                ema_state[k] = 0.5 * ema_state[k] + 0.5 * _swa_gpu[i].cpu()
            del _swa_gpu
        if bool(int(os.environ.get("SKIP_EMA", "0"))):
            log0("ema: SKIPPED (SKIP_EMA=1) — using training-time weights")
        else:
            log0("ema: loading weights")
            ema_state = {k: v.to(dtype=_ema_orig_dtypes[k]) for k, v in ema_state.items()}
            base_model.load_state_dict(ema_state, strict=True)
        for module in base_model.modules():
            if isinstance(module, CastedLinear): module.float()
        restore_low_dim_params_to_fp32(base_model)
        del ema_state
        if master_process:
            torch.save(base_model.state_dict(), "final_model.float.pt")
            log0(f"saved pre-GPTQ float checkpoint: {os.path.getsize('final_model.float.pt')} bytes")
    if bool(int(os.environ.get("PREQUANT_EVAL", "1"))):
        torch.cuda.synchronize()
        _t_pre = time.perf_counter()
        _pq_loss, _pq_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        torch.cuda.synchronize()
        log0(f"pre-quantization post-ema val_loss:{_pq_loss:.6f} val_bpb:{_pq_bpb:.6f} eval_time:{1000.0 * (time.perf_counter() - _t_pre):.0f}ms")
    _val_calib = None
    if bool(int(os.environ.get("VAL_CALIB", "1"))) or bool(int(os.environ.get("REAL_DATA_CALIB", "0"))):
        rng = torch.Generator(); rng.manual_seed(args.seed)
        _n_calib = int(os.environ.get("CALIB_BATCHES", "64"))
        _vc = [val_tokens[i:i+args.train_seq_len+1].to(torch.int64).unsqueeze(0) for i in (torch.randint(0, val_tokens.numel()-args.train_seq_len-1, (_n_calib,), generator=rng)).tolist()]
        _val_calib = _vc
        log0(f"val_calib: using {len(_val_calib)} validation windows (seq_len={args.train_seq_len})")
    gptq_scales, gptq_crs = apply_gptq_sdclip_inplace(base_model, device, args, log_fn=log0, calib_override=_val_calib)
    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        sz = os.path.getsize("final_model.pt"); csz = len(code.encode("utf-8"))
        log0(f"Serialized model: {sz} bytes  Code: {csz}  Total: {sz + csz}")
    cr = GPTQ_CR
    quantized_t: dict[str, Tensor] = {}
    scales_t: dict[str, Tensor] = {}
    dtypes_t: dict[str, str] = {}
    passthrough_t: dict[str, Tensor] = {}
    passthrough_orig_dtypes_t: dict[str, str] = {}
    qmeta_t: dict[str, dict] = {}
    quant_payload = 0
    for name, tensor in export_sd.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point():
            passthrough_t[name] = t
            quant_payload += t.numel() * t.element_size()
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough_t[name] = t.float().contiguous()
            quant_payload += t.numel() * 4
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                passthrough_t[name] = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes_t[name] = str(t.dtype).removeprefix("torch.")
                passthrough_t[name] = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                passthrough_t[name] = t
            quant_payload += passthrough_t[name].numel() * passthrough_t[name].element_size()
            continue
        if name in gptq_scales:
            sf = gptq_scales[name].float()
            t32 = t.float()
            _cr = gptq_crs.get(name, GPTQ_CR)
            q = torch.clamp(torch.round(t32 / sf[:, None]), -_cr, _cr).to(torch.int8).contiguous()
            s = sf.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
        elif "tok_emb.weight" in name:
            q, s = quantize_float_tensor_int6(t, bits=int(os.environ.get("EMB_BITS", "8")), sd_k=float(os.environ.get("EMB_CLIP_K", "12.85")))
        else:
            q, s = quantize_float_tensor_int6(t, bits=6)
        quantized_t[name] = q
        scales_t[name] = s
        dtypes_t[name] = str(t.dtype).removeprefix("torch.")
        if s.ndim > 0:
            qmeta_t[name] = {"scheme": "per_row", "axis": 0}
        quant_payload += q.numel() * q.element_size() + s.numel() * s.element_size()
    quant_obj = {"__quant_format__": f"gptq_sdclip_cr{cr}_per_row_v1", "quantized": quantized_t, "scales": scales_t, "dtypes": dtypes_t, "passthrough": passthrough_t}
    if qmeta_t:
        quant_obj["qmeta"] = qmeta_t
    if passthrough_orig_dtypes_t:
        quant_obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes_t
    quant_buf = io.BytesIO(); torch.save(quant_obj, quant_buf); quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(quant_blob)
        qsz = os.path.getsize("final_model.int6.ptz"); csz = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {qsz} bytes (payload:{quant_payload} raw:{len(quant_raw)})")
        log0(f"Total submission size: {qsz + csz} bytes")
        if bool(int(os.environ.get("SKIP_EVAL_OVERSIZE", "0"))) and (qsz + csz) > 16 * 1024 * 1024:
            log0(f"SKIP_EVAL_OVERSIZE: artifact {qsz + csz} > 16MB, skipping val eval")
            sys.exit(0)
    if distributed:
        dist.barrier()

    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    torch.cuda.synchronize()
    t_slide = time.perf_counter()
    _, sw_val_bpb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_sliding_window val_bpb:{sw_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms")
    log0(f"final_sliding_window_exact val_bpb:{sw_val_bpb:.8f}")
    if args.skip_ttt:
        log0("ttt: skipped (skip_ttt=True)")
    else:
        ttt_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            num_loops=args.num_loops, loop_start=args.loop_start, loop_end=args.loop_end,
        ).to(device).bfloat16()
        for m in ttt_model.modules():
            if isinstance(m, CastedLinear): m.float()  # fp32 matrix params
        restore_low_dim_params_to_fp32(ttt_model)
        ttt_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=False)
        ttt_model.eval_hash_emb = nn.Embedding(args.eval_hash_buckets, args.model_dim).to(device)
        nn.init.zeros_(ttt_model.eval_hash_emb.weight)
        log0(f"ttt: hash_emb attached ({args.eval_hash_buckets} buckets, lr_mult={args.eval_hash_lr_mult})")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_model = torch.compile(ttt_model)
        log0("ttt: starting")
        ttt_bpb = eval_val_ttt(args, ttt_model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log0)
        torch.cuda.synchronize()
        log0(f"final_ttt val_bpb:{ttt_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"final_ttt_exact val_bpb:{ttt_bpb:.8f}")
        del ttt_model
    if distributed: dist.destroy_process_group()
    if log_fh is not None: log_fh.close()
if __name__ == "__main__":
    main()
