import time
import tiktoken
import torch 

from data_loader import DataLoaderLite
from gpt2 import GPT2
from gpt_config import GPTConfig

loader = DataLoaderLite(B=8, T=32, input_file="data/shakesphere/input.txt")

model = GPT2(GPTConfig())
model.to("mps")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print("Starting training...")
print(f"Total batches: {len(loader)}")
for i in range(len(loader)):
    start = time.time()
    x, y = loader.next_batch()
    x = torch.tensor(x, dtype=torch.long).to("mps")
    y = torch.tensor(y, dtype=torch.long).to("mps")
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end = time.time()
    print(f"Step {i+1}, Loss: {loss.item()}, Time: {end - start:.2f}s")
print("Training complete.")
