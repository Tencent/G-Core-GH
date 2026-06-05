import torch

for i in range(8):
    a = torch.load(f"test_data/rollout_0_rank{i}.pt", weights_only=False)
    for j in range(8):
        a["images"][j].save(f"test_data/image_{i}_{j}.png")
    print(f"caption {i} {a['captions'][0]}")
