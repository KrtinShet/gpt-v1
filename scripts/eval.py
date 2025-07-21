import torch
import hydra
from omegaconf import DictConfig
import numpy as np

from src.models.gpt2 import GPT2
from src.data.data_loader import DataLoader


@hydra.main(config_path="../configs", config_name="default")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluates the GPT-2 model.

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

    # Initialize data loader
    data_loader = DataLoader(
        data_dir=cfg.data.data_dir,
        sequence_length=cfg.data.sequence_length,
        batch_size=cfg.train.batch_size,
        device=device,
    )

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
    checkpoint = torch.load(cfg.eval.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Calculate perplexity
    losses = []
    with torch.no_grad():
        for x, y in data_loader:
            _, loss = model(x, y)
            losses.append(loss.item())

    perplexity = np.exp(np.mean(losses))
    print(f"Perplexity: {perplexity}")


if __name__ == "__main__":
    evaluate()
