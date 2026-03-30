import torch
sd = torch.load('final_model.pt', map_location='cpu')
for k in sd:
    if 'buf' in k.lower() or 'running' in k.lower() or 'num_batches' in k.lower():
        print(k, sd[k].shape)
print('Total keys:', len(sd))
print('Keys:', sorted(sd.keys())[:10])
