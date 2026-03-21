import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_transforms() -> T.Compose:
    """
    Build image preprocessing pipeline.
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform


def build_datasets(data_root: str):
    """
        Build TinyImageNet train/val datasets.

        Expected structure:
        data_root/
            tiny-imagenet-200/
                train/
                val/
    """
    transform = get_transforms()

    train_path = os.path.join(data_root, "tiny-imagenet-200", "train")
    val_path = os.path.join(data_root, "tiny-imagenet-200", "val")

    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset = ImageFolder(root=val_path, transform=transform)

    return train_dataset, val_dataset


def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
        Build TinyImageNet train/val dataloaders.
        dataset logic stays here, not inside train.py / eval.py (for some unknown reasons)
    """
    train_dataset, val_dataset = build_datasets(data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader