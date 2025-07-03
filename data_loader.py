import tiktoken

class DataLoaderLite:
  def __init__(self, B, T, input_file) -> None:
    self.B = B  # Batch size
    self.T = T  # Sequence length
    self.input_file = input_file
    self.current_batch = 0

    self.tokenizer = tiktoken.get_encoding("gpt2") 
    with open(input_file, 'r') as f:
      self.data = f.read()
    tokens = self.tokenizer.encode(self.data)
    self.num_batches: int = len(tokens) // (B * T)
    self.tokens = tokens[:self.num_batches * B * T]  # Trim to fit full batches

  def __len__(self):
    return self.num_batches

  def next_batch(self):
    batch_start = self.current_batch * self.B * self.T
    batch_end = batch_start + self.B * self.T
    batch_tokens = self.tokens[batch_start:batch_end]
    
    # Reshape to (B, T)
    batch = [batch_tokens[i:i + self.T] for i in range(0, len(batch_tokens), self.T)]
    x, y = batch[:-1], batch[1:]  # Shift for next token prediction 
    
    self.current_batch += 1
    return x, y