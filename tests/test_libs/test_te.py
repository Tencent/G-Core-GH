import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


def test_linear():
    # Set dimensions.
    in_features = 768
    out_features = 3072
    hidden_size = 2048

    # Initialize model and inputs.
    model = te.Linear(in_features, out_features, bias=True)
    inp = torch.randn(hidden_size, in_features, device="cuda")

    out = model(inp)

    loss = out.sum()
    loss.backward()
