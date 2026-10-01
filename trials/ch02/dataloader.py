"""Section 2.6: GPTDatasetV1 and create_dataloader_v1."""

import tiktoken
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # raise NotImplementedError

        """
        0 1 2 3 4 5 6 7
                [     )
                  [   ]
        """

        tokens = tokenizer.encode(txt)
        boundary = len(tokens) - max_length    #  (8 - 3) = 5
        self.src_data = []
        self.tgt_data = []

        for i in range(0, boundary, stride):
            src_window = tokens[i: i+max_length]       # [4: 4+3) => [4, 5, 6]
            tgt_window = tokens[i+1: i+1+max_length]   # [5: 5+3] => [5, ,6, 7]
            
            self.src_data.append(src_window)
            self.tgt_data.append(tgt_window)

        return

    def __len__(self):
        # raise NotImplementedError
        return len(self.src_data)

    def __getitem__(self, idx):
        # raise NotImplementedError
        return torch.tensor(self.src_data[idx]),  torch.tensor(self.tgt_data[idx])


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # raise NotImplementedError

    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset_v1 = GPTDatasetV1(
        txt=txt,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )

    dl = DataLoader(
        dataset=dataset_v1,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers
    )

    return dl
