import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.fft as fft
import os
import numpy as np
from typing import Tuple

from .augmentations import DataTransform_FD, DataTransform_TD

def generate_freq(dataset, config):
    X_train = dataset["samples"]
    y_train = dataset['labels']
    # shuffle
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    data = data[:10000] # take a subset for testing.
    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs() #/(window_length) # rfft for real value inputs.
    return (X_train, y_train, x_data_f)

class Load_Dataset(Dataset):
    """
    Load dataset and perform augmentations.
    """
    def __init__(
            self, dataset: dict, config, training_mode: str,
            target_dataset_size: int=64, subset: bool=False
        ):
        """
        Parameters
        ----------
        dataset : dict
            A dictionary containing 'samples' and 'labels'.
        config : object
            Configuration object with necessary attributes.
            - TSlength_aligned : int, the aligned time series length.
            - batch_size : int, batch size for source dataset.
            - target_batch_size : int, batch size for target dataset.
        target_dataset_size : int
            The size of the target dataset, used when subset=True.
        training_mode : str
            'pre_train' or 'fine_tune_test'.
            'pre_train': self-supervised pre-training;
            'fine_tune_test': supervised fine-tuning and testing.
        subset : bool
            If True, only use a subset of the dataset for debugging.
        """
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        X_train, y_train = shuffle(X_train, y_train)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        # Subset for debugging
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]

        # Augmentation
        if training_mode == "pre_train":
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config)

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def shuffle(X_train, y_train):
    """
    Shuffle the dataset.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        The input samples.
    y_train : array-like, shape (n_samples,)
        The class labels.
    """
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    X_train, y_train = zip(*data)
    X_train = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X_train]
    y_train = [torch.tensor(y) if not isinstance(y, torch.Tensor) else y for y in y_train]
    X_train, y_train = torch.stack(X_train, dim=0), torch.stack(y_train, dim=0)

    return X_train, y_train


def data_generator(
        sourcedata_path: str, targetdata_path: str, configs,
        training_mode: str, subset: bool=True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate data loaders for source and target datasets.

    Parameters
    ----------
    sourcedata_path : str
        Path to the source dataset directory,
        with file train.pt inside.
    targetdata_path : str
        Path to the target dataset directory,
        with files train.pt and test.pt inside.
    configs : object
        Configuration object with necessary attributes.
        - batch_size : int, batch size for source dataset.
        - target_batch_size : int, batch size for target dataset.
        - drop_last : bool, whether to drop the last incomplete batch.
    training_mode : str
        'pre_train' or 'fine_tune_test'.
        'pre_train': self-supervised pre-training;
        'fine_tune_test': supervised fine-tuning and testing.
    subset : bool
        If True, only use a subset of the dataset for debugging.
    """
    training_path = os.path.join(sourcedata_path, "train.pt")
    finetune_path = os.path.join(targetdata_path, "train.pt")
    test_path = os.path.join(targetdata_path, "test.pt")

    if not os.path.exists(training_path):
        raise ValueError(
            f"Cannot find file train.pt in {sourcedata_path}. "
            "Please download the dataset first. "
        )
    if not os.path.exists(finetune_path):
        raise ValueError(
            f"Cannot find file train.pt in {targetdata_path}. "
            "Please download the dataset first. "
        )
    if not os.path.exists(test_path):
        raise ValueError(
            f"Cannot find file test.pt in {targetdata_path}. "
            "Please download the dataset first. "
        )

    train_dataset = torch.load(training_path, weights_only=False)
    finetune_dataset = torch.load(finetune_path, weights_only=False)
    test_dataset = torch.load(test_path, weights_only=False)

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(
        train_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset)
    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode,
                                target_dataset_size=configs.target_batch_size, subset=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, finetune_loader, test_loader
