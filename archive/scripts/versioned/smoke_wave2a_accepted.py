"""Versioned smoke for git tag `wave2a-accepted` (commit bd67512).

Baked-in stack (committed defaults in train_gpt.py, no env var needed for values):
- ema_decay = 0.993
- muon_momentum = 0.995
- DEPTH_LR_FACTOR = 0.85
- attn_out_gate unconditional (AttnOutGate baked)

Feature flags still required (they toggle behaviour paths, not HP values):
- ENABLE_LOOPING_AT=0.25  — enables depth recurrence at 25% of wallclock
- MIXED_QUANT_B=1         — mixed-precision quant scheme B
- TWO_PASS_GPTQ=1         — two-pass GPTQ Hessian refresh
- DEPTH_LR=1              — turns on depth-specific LR scaling (default factor 0.85 now)

Delegates to the shared smoke harness at archive/scripts/current/smoke_clean_c39da7e.py,
which monkey-patches load_validation_tokens (500k token subset) and skips sliding eval.

Reference numbers (averaged W2A original + replication, GPU ~920ms/step):
  pre  1.4800
  post 1.4968
"""
import os, sys, runpy

os.environ.setdefault("ENABLE_LOOPING_AT", "0.25")
os.environ.setdefault("MIXED_QUANT_B", "1")
os.environ.setdefault("TWO_PASS_GPTQ", "1")
os.environ.setdefault("DEPTH_LR", "1")

_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "current", "smoke_clean_c39da7e.py")
runpy.run_path(_base, run_name="__main__")
