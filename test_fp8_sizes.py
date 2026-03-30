import torch, time
from torchao.float8 import convert_to_float8_training

def bench(name, dim, mlp_mult=3, seq_len=2048, batch=16):
    hidden = dim * mlp_mult
    x = torch.randn(batch, seq_len, dim, device='cuda', dtype=torch.bfloat16)
    m_bf16 = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, dim)).cuda().bfloat16()
    m_bf16c = torch.compile(m_bf16)
    for _ in range(3): m_bf16c(x).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): m_bf16c(x).sum().backward()
    torch.cuda.synchronize()
    bf16_ms = (time.time() - t0) / 20 * 1000
    m_fp8 = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, dim)).cuda().bfloat16()
    convert_to_float8_training(m_fp8)
    m_fp8c = torch.compile(m_fp8)
    for _ in range(3): m_fp8c(x).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): m_fp8c(x).sum().backward()
    torch.cuda.synchronize()
    fp8_ms = (time.time() - t0) / 20 * 1000
    print(f'{name} dim={dim} hidden={hidden}: bf16={bf16_ms:.1f}ms fp8={fp8_ms:.1f}ms speedup={bf16_ms/fp8_ms:.2f}x')

bench('small', 512)
bench('medium', 768)
bench('large', 1024)
bench('xlarge', 1536)
bench('xxlarge', 2048)
bench('hourglass_mid', 768, mlp_mult=3, batch=8)
bench('hourglass_entry', 512, mlp_mult=3, batch=8)
print('\nDone.')
