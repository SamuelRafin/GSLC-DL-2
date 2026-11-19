import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


def make_loaders(root="data", batch_size=128, seed=42, num_workers=2, device=None):
    train_source = FashionMNIST(root, train=True, download=True, transform=ToTensor())
    test_source = FashionMNIST(root, train=False, download=True, transform=ToTensor())
    dataset = ConcatDataset((train_source, test_source))

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, (n_train, n_val, n_test), generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(seed + 1)
    use_cuda = device is not None and torch.device(device).type == "cuda"
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=loader_generator, **options
    )
    val_loader = DataLoader(val_set, shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)
    return train_loader, val_loader, test_loader
