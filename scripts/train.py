import os
import torch
import hydra
from omegaconf import DictConfig
import wandb

from src.models.gpt2 import GPT2
from src.data.data_loader import DataLoader


@hydra.main(config_path="../configs", config_name="default")
def train(cfg: DictConfig) -> None:
    """
    Trains the GPT-2 model.

    Args:
        cfg (DictConfig): The configuration object.
    """
    # Initialize logging
    wandb.init(project=cfg.logging.project, name=cfg.logging.name, config=dict(cfg))

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

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    # Resume training if specified
    if cfg.train.resume:
        checkpoint = torch.load(os.path.join(cfg.train.checkpoint_dir, "latest.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        for i, (x, y) in enumerate(data_loader):
            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log to wandb
            if i % cfg.train.log_interval == 0:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": i})

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save(
            checkpoint, os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch}.pt")
        )
        torch.save(checkpoint, os.path.join(cfg.train.checkpoint_dir, "latest.pt"))


if __name__ == "__main__":
    train()
