import torch
import hydra
from omegaconf import DictConfig
import tiktoken

from src.models.gpt2 import GPT2


@hydra.main(config_path="../configs", config_name="default")
def infer(cfg: DictConfig) -> None:
    """
    Generates text using the GPT-2 model.

    Args:
        cfg (DictConfig): The configuration object.
    """
    # Set device
    if cfg.train.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = cfg.train.device

    # Initialize model
    model = GPT2(
        vocab_size=cfg.model.vocab_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.data.sequence_length,
        dropout=0.1,  # Hardcoded for now
        bias=True,  # Hardcoded for now
        tie_weights=True,  # Hardcoded for now
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(cfg.inference.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Generate text
    start_ids = enc.encode(cfg.inference.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=cfg.inference.max_new_tokens)
        print(enc.decode(y[0].tolist()))


if __name__ == "__main__":
    infer()
