"""Smoke test for the new train_gpt.py with depth recurrence + SD-Clip + forward(target=None)."""
import importlib.util
import subprocess
from types import SimpleNamespace

import torch

TRAIN_GPT_PATH = "/projects/parameter-golf/train_gpt.py"


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("train_gpt_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert torch.cuda.is_available(), "CUDA required for this smoke test"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    mod = load_module(TRAIN_GPT_PATH)
    GPT = mod.GPT

    model = GPT(
        vocab_size=100,
        num_layers=11,
        model_dim=64,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=4.0,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=4.0,
        num_loops=2,
        loop_start=4,
        loop_end=5,
    ).to(device).bfloat16()

    for m in model.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(model)

    print(f"effective_indices = {model.effective_indices}")
    print(f"encoder_indices  = {model.encoder_indices}")
    print(f"decoder_indices  = {model.decoder_indices}")
    print(f"num_skip_weights = {model.num_skip_weights}")

    B, T = 2, 32
    x = torch.randint(0, 100, (B, T), device=device, dtype=torch.int64)
    y = torch.randint(0, 100, (B, T), device=device, dtype=torch.int64)

    # 1. eager forward returning logits when target is None
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
    assert logits.shape == (B, T, 100), f"got {logits.shape}"
    print("1. eager logits ok", tuple(logits.shape))

    # 2. compiled forward
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    compiled.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits2 = compiled(x)
    assert logits2.shape == (B, T, 100), f"got {logits2.shape}"
    print("2. compiled logits ok", tuple(logits2.shape))

    # 3. forward + backward through recurrence (loops 4-5 ×2 must allow gradient through reused blocks)
    compiled.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = compiled(x, y)
    assert loss.ndim == 0, f"got {loss.shape}"
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "no gradients found"
    print("3. forward+backward ok loss=", float(loss.detach().cpu()), "grads=", grad_count)

    # 4. quantize/dequantize roundtrip with SD-Clip
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    qobj, qstats = mod.quantize_state_dict_int6(state)
    deq = mod.dequantize_state_dict_int8(qobj)
    assert state.keys() == deq.keys(), "state dict keys diverged"
    worst = 0.0
    worst_name = None
    for k in state:
        a = state[k].float()
        b = deq[k].float()
        if a.shape != b.shape:
            raise AssertionError(f"shape mismatch for {k}: {a.shape} vs {b.shape}")
        err = (a - b).abs().max().item()
        if err > worst:
            worst = err
            worst_name = k
    assert worst < 1.0, f"quant max error {worst} too high on {worst_name}"
    print("4. quant roundtrip ok max_err=", worst, "on", worst_name)

    # 5. no forward_logits anywhere
    rg = subprocess.run(
        ["rg", "-n", r"\bforward_logits\b", TRAIN_GPT_PATH],
        capture_output=True, text=True,
    )
    assert rg.returncode != 0, f"forward_logits still present:\n{rg.stdout}"
    print("5. forward_logits eliminated ok")

    torch.cuda.synchronize()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
