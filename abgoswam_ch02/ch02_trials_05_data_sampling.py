from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print("tiktoken version:", version("tiktoken"))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def debug_dataloader():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=8, 
        max_length=4, 
        stride=4, 
        shuffle=False
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(raw_text[:32])
    print(ids)

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)


def debug_embeddings():
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    input_ids = torch.tensor([2, 3, 5, 1])
    x = embedding_layer(input_ids)
    print(x.shape)


def debug_dataloader_and_embeddings():

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    # token embedding.
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    # pos embedding.
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)





if __name__ == "__main__":

    # debug_dataloader()

    # debug_embeddings()

    debug_dataloader_and_embeddings()

    print("done")
