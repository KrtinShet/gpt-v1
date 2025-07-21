import torch
from src.models.gpt2 import GPT2


def test_gpt2_forward_pass():
    """Tests the forward pass of the GPT2 model."""
    config = {
        "vocab_size": 50257,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "block_size": 256,
        "dropout": 0.1,
        "bias": True,
        "tie_weights": True,
    }
    model = GPT2(**config)
    x = torch.randint(0, config["vocab_size"], (1, 256))
    logits, loss = model(x)
    assert logits.shape == (1, 1, config["vocab_size"])
    assert loss is None
