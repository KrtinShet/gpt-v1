import torch
import tiktoken
import os


class DataLoader:
    """
    Data loader for the GPT-2 model.
    Yields batches of (x, y) tensors.
    """

    def __init__(
        self, data_dir: str, sequence_length: int, batch_size: int, device: str = "cpu"
    ):
        """
        Initializes the DataLoader.

        Args:
            data_dir (str): The directory containing the input data.
            sequence_length (int): The sequence length of the model.
            batch_size (int): The batch size for training.
            device (str): The device to move the tensors to.
        """
        self.B = batch_size
        self.T = sequence_length
        self.device = device

        input_file_path = os.path.join(data_dir, "input.txt")
        with open(input_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.num_batches = len(self.tokens) // (self.B * self.T)

        # state
        self.current_pos = 0

    def __iter__(self):
        self.current_pos = 0
        return self

    def __next__(self):
        if self.current_pos + self.B * self.T + 1 > len(self.tokens):
            raise StopIteration

        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_pos += self.B * self.T

        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return self.num_batches
