import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    from torchao.float8 import convert_to_float8_training
    print('torchao float8: AVAILABLE')
except Exception as e:
    print(f'torchao float8: FAILED ({e})')

try:
    a = torch.randn(256, 512, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(1536, 512, device='cuda').to(torch.float8_e4m3fn).T.contiguous().T
    out = torch._scaled_mm(a, b, scale_a=torch.tensor(1.0, device='cuda', dtype=torch.float32), scale_b=torch.tensor(1.0, device='cuda', dtype=torch.float32), out_dtype=torch.bfloat16, use_fast_accum=True)
    print(f'_scaled_mm: WORKS shape={out.shape}')
except Exception as e:
    print(f'_scaled_mm: FAILED ({e})')

try:
    print(f'torch.library.custom_op: {hasattr(torch.library, "custom_op")}')
except:
    print('torch.library.custom_op: NOT AVAILABLE')

try:
    from torchao.float8 import convert_to_float8_training
    model = torch.nn.Sequential(torch.nn.Linear(512, 1536), torch.nn.Linear(1536, 512)).cuda().bfloat16()
    convert_to_float8_training(model)
    x = torch.randn(32, 2048, 512, device='cuda', dtype=torch.bfloat16)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f'torchao FP8 forward+backward: WORKS output={y.shape}')
except Exception as e:
    print(f'torchao FP8 forward+backward: FAILED ({e})')

try:
    from torchao.float8 import convert_to_float8_training
    model2 = torch.nn.Sequential(torch.nn.Linear(512, 1536), torch.nn.Linear(1536, 512)).cuda().bfloat16()
    convert_to_float8_training(model2)
    compiled = torch.compile(model2)
    x = torch.randn(32, 2048, 512, device='cuda', dtype=torch.bfloat16)
    y = compiled(x)
    loss = y.sum()
    loss.backward()
    print(f'compile + FP8: WORKS')
except Exception as e:
    print(f'compile + FP8: FAILED ({e})')

try:
    import time
    model3 = torch.nn.Sequential(torch.nn.Linear(512, 1536), torch.nn.Linear(1536, 512)).cuda().bfloat16()
    model3c = torch.compile(model3)
    x = torch.randn(32, 2048, 512, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        model3c(x).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        model3c(x).sum().backward()
    torch.cuda.synchronize()
    bf16_ms = (time.time() - t0) / 20 * 1000
    model4 = torch.nn.Sequential(torch.nn.Linear(512, 1536), torch.nn.Linear(1536, 512)).cuda().bfloat16()
    convert_to_float8_training(model4)
    model4c = torch.compile(model4)
    for _ in range(3):
        model4c(x).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        model4c(x).sum().backward()
    torch.cuda.synchronize()
    fp8_ms = (time.time() - t0) / 20 * 1000
    print(f'SPEED: bf16={bf16_ms:.1f}ms fp8={fp8_ms:.1f}ms speedup={bf16_ms/fp8_ms:.2f}x')
except Exception as e:
    print(f'SPEED TEST FAILED: {e}')

print('\nAll tests done.')
