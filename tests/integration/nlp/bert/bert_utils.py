"""Functions for processing datasets and model weights"""

# pylint: disable=C0301
from os.path import abspath, join
from typing import Tuple

from datasets import load_from_disk
from torch.utils.data import DataLoader

from torch_musa.testing.integration.utils import (
    check_existent_directory,
    get_dataset_root_dir,
    get_model_root_dir,
)


def get_bert_base_uncased_train_small_imdb_model_dir() -> str:
    """Find pretrained model root for bert-base-uncased small-imdb training"""
    model_dir: str = get_model_root_dir()
    model_dir = join(
        abspath(model_dir),
        "bert-base-uncased",
        "seq_binary_classification_train_small_imdb",
    )
    check_existent_directory(
        model_dir,
        f"Not found bert-base-uncased model directory `{model_dir}` for small-imdb training",
    )
    return model_dir


def get_bert_base_uncased_eval_small_imdb_model_dir() -> str:
    """Find pretrained model root for bert-base-uncased small-imdb evaluation"""
    model_dir: str = get_model_root_dir()
    model_dir = join(
        abspath(model_dir),
        "bert-base-uncased",
        "seq_binary_classification_eval_small_imdb",
    )
    check_existent_directory(
        model_dir,
        f"Not found bert-base-uncased model directory `{model_dir}` for small-imdb evaluation",
    )
    return model_dir


def get_bert_base_uncased_train_small_imdb_datasets(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Find preprocessed imdb small datasets for bert-base-uncased training"""
    dataset_dir: str = abspath(get_dataset_root_dir())
    train_dir = join(dataset_dir, "imdb", "bert_base_uncased_train_small")
    check_existent_directory(
        train_dir,
        f"Not found bert-base-uncased imdb small train dataset directory `{train_dir}` for training",
    )
    test_dir = join(dataset_dir, "imdb", "bert_base_uncased_test_small")
    check_existent_directory(
        test_dir,
        f"Not found bert-base-uncased imdb small test dataset directory `{test_dir}` for training",
    )

    small_train_dataset = load_from_disk(train_dir)
    small_test_dataset = load_from_disk(test_dir)

    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(small_test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


def get_bert_base_uncased_eval_small_imdb_dataset(
    batch_size: int,
) -> DataLoader:
    """Find preprocessed imdb small dataset for bert-base-uncased evalaution"""
    dataset_dir: str = abspath(get_dataset_root_dir())
    eval_dir = join(dataset_dir, "imdb", "bert_base_uncased_eval_small")
    check_existent_directory(
        eval_dir,
        f"Not found bert-base-uncased imdb small eval dataset directory `{eval_dir}` for evalaution",
    )

    small_eval_dataset = load_from_disk(eval_dir)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return eval_dataloader
