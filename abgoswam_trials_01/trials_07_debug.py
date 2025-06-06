import tiktoken
import torch

def debug_init(txt, tokenizer, max_length, stride):
    input_ids = []
    target_ids = []

    # Tokenize the entire text
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
    assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

    # Use a sliding window to chunk the book into overlapping sequences of max_length
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        
        input_ids.append(torch.tensor(input_chunk))
        target_ids.append(torch.tensor(target_chunk))

def debug_embeddings():
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    
    pass    

if __name__ == "__main__":

    # =========================
    debug_embeddings()
    
    # ==========================

    tokenizer = tiktoken.get_encoding("gpt2")
    text = "hellohellohellohello"

    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # token_ids = tokenizer.encode(text)
    print(token_ids)

    # Convert token IDs to their corresponding strings
    tokens = [tokenizer.decode([token]) for token in token_ids]
    print(tokens)

    debug_init(
        txt=text,
        tokenizer=tokenizer,
        max_length=4,
        stride=1
    )

    # ===============================

    print("done")
