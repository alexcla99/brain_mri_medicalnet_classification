from torch.utils.data import Dataset
import numpy as np
import os

DATA_DIR = "data" # The folder containing data

class AnoxiaDataset(Dataset):
    """Dataset loader for the Anoxia dataset."""

    def __init__(self, phase:str) -> None:
        """Instanciate the dataset loader."""
        self.phase = phase
        self.x_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
        self.x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
        self.x_val = np.load(os.path.join(DATA_DIR, "x_val.npy"))
        self.y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        self.y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
        self.y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    def __getitem__(self, idx:int) -> (np.ndarray, np.ndarray):
        """Return an item according to the current phase."""
        if self.phase == "train":
            return self.x_train[idx], self.y_train[idx]
        elif self.phase == "val":
            return self.x_val[idx], self.y_val[idx]
        elif self.phase == "test":
            return self.x_test[idx], self.y_test[idx]

    def __len__(self) -> int:
        """Length of the dataset."""
        if self.phase == "train":
            return len(self.x_train)
        elif self.phase == "val":
            return len(self.x_val)
        elif self.phase == "test":
            return len(self.x_test)