import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is a simplified implementation of the one in GPT-2.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        """
        Initializes the CausalSelfAttention layer.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
            dropout (float): The dropout rate.
            bias (bool): Whether to use bias in the linear layers.
        """
        super().__init__()
        assert (
            n_embd % n_head == 0
        ), "Embedding dimension must be divisible by number of heads."
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = nn.Dropout(dropout)

        # Key, Query, Value projections for all heads at once
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the CausalSelfAttention layer.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: The output tensor of shape (B, T, C).
        """
        B, T, C = x.size()  # batch, sequence, embedding
        # Project input to Q, K, V and split heads
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape for multi-head: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    """
    A simple MLP with a single hidden layer and GELU activation.
    """

    def __init__(self, n_embd: int, dropout: float, bias: bool):
        """
        Initializes the MLP layer.

        Args:
            n_embd (int): The embedding dimension.
            dropout (float): The dropout rate.
            bias (bool): Whether to use bias in the linear layers.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP layer.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: The output tensor of shape (B, T, C).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        """
        Initializes the Block.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
            dropout (float): The dropout rate.
            bias (bool): Whether to use bias in the linear layers.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: The output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    The GPT-2 model.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
        bias: bool,
        tie_weights: bool,
    ):
        """
        Initializes the GPT-2 model.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_layer (int): The number of transformer blocks.
            n_head (int): The number of attention heads.
            n_embd (int): The embedding dimension.
            block_size (int): The sequence length of the model.
            dropout (float): The dropout rate.
            bias (bool): Whether to use bias in the linear layers.
            tie_weights (bool): Whether to tie the weights of the token embedding and the final linear layer.
        """
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            [Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if tie_weights:
            self.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass of the GPT-2 model.

        Args:
            idx (torch.Tensor): The input tensor of shape (B, T).
            targets (Optional[torch.Tensor]): The target tensor of shape (B, T).

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the logits and the loss.
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
