import torch
from torch.utils.data import random_split


def random_split_by_ratio(data, test_size):
    length = len(data)
    test_size = int(test_size * length)
    train_size = length - test_size
    train_data, val_data = random_split(data, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(42))
    return train_data, val_data
