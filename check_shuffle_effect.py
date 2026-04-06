"""Test brotli with and without byte-shuffle on the actual 8xH100 weights."""
import torch, io, os, sys, lzma
sys.path.insert(0, os.path.dirname(__file__))
import brotli
from train_gpt import quantize_state_dict_int6, _byte_shuffle

sd = torch.load(sys.argv[1] if len(sys.argv) > 1 else "final_model.pt", map_location="cpu", weights_only=True)
print(f"Params: {sum(t.numel() for t in sd.values()):,}")
obj, _ = quantize_state_dict_int6(sd)
buf = io.BytesIO()
torch.save(obj, buf)
raw = buf.getvalue()
print(f"Raw pickle: {len(raw):,} bytes")

lz = lzma.compress(raw, preset=9)
b_plain = brotli.compress(raw, quality=11)
b_shuffle = brotli.compress(_byte_shuffle(raw), quality=11)
CODE = 72000
print(f"LZMA-9:             {len(lz)+CODE:,} bytes ({(len(lz)+CODE)/1e6:.2f}MB)")
print(f"Brotli-11:          {len(b_plain)+CODE:,} bytes ({(len(b_plain)+CODE)/1e6:.2f}MB)")
print(f"Brotli+shuffle:     {len(b_shuffle)+CODE:,} bytes ({(len(b_shuffle)+CODE)/1e6:.2f}MB)")
print(f"Best: {'shuffle' if len(b_shuffle) < len(b_plain) else 'plain'} brotli")
