"""Functions for processing datasets and model weights"""

from os.path import join
from typing import Tuple

from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet50_Weights
import torch
from torch.utils.data import DataLoader

from torch_musa.testing.integration.utils import (
    check_existent_directory,
    check_existent_file,
    get_dataset_root_dir,
    get_model_root_dir,
)

_DATASETCLS = datasets.CIFAR10


def get_cifar10_training_dataset(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Generate cifar10 dataset for resnet50 training"""
    data_dir: str = get_dataset_root_dir()
    data_dir = join(data_dir, "cifar10")
    check_existent_directory(
        data_dir, f"Not found cifar10 dataset directory `{data_dir}`"
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_set = _DATASETCLS(
        root=data_dir,
        train=True,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_set = _DATASETCLS(
        root=data_dir,
        train=False,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, test_loader


def get_imagenet2012_evaluation_small_dataset(batch_size: int) -> DataLoader:
    """Generate imagenet2012 small dataset for resnet50 evaluation"""
    data_dir: str = get_dataset_root_dir()
    data_dir = join(data_dir, "imagenet2012", "small_val_tensor")
    check_existent_directory(
        data_dir, f"Not found imagenet2012 eval small dataset directory `{data_dir}`"
    )

    def load_tensor(path):
        return torch.load(path, map_location="cpu")

    def is_valid_file(_):
        return True

    eval_set = datasets.ImageFolder(
        data_dir, loader=load_tensor, is_valid_file=is_valid_file
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return eval_loader


def get_resnet50_training_ckpt() -> str:
    """Find ckpt file for resnet50 training"""
    data_dir: str = get_model_root_dir()
    data_file = join(data_dir, "resnet50", "weights.pt")
    check_existent_file(
        data_file, f"Not found resnet50 training ckpt file: `{data_file}`"
    )
    return data_file


_EVAL_CKPT_URL = ResNet50_Weights.IMAGENET1K_V1.url
_EVAL_CKPT_NAME = _EVAL_CKPT_URL.rsplit("/", maxsplit=1)[-1]


def get_resnet50_evaluation_ckpt() -> str:
    """Find ckpt file for resnet50 evaluation"""
    data_dir: str = get_model_root_dir()
    data_file = join(data_dir, "resnet50", _EVAL_CKPT_NAME)
    check_existent_file(
        data_file, f"Not found resnet50 evaluation ckpt file: `{data_file}`"
    )
    return data_file
