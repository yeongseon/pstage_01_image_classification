import torch
from torch.utils.data import DataLoader

from dataset import MaskBaseDataset


class BaseDataLoader(DataLoader):
    def __init__(self, dataset: MaskBaseDataset, val_split: float, batch_size: int, num_workers: int = 2):
        super().__init__()

        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
        val_set.dataset.set_phase("test")  # todo : fix

        data_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
