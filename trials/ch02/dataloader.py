"""Section 2.6: GPTDatasetV1 and create_dataloader_v1."""

from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    raise NotImplementedError
