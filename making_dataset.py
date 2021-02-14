from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils    
import pandas as pd
import os
from PIL import Image
import numpy as npdwaita5.3

from tqdm import tqdm
from generateDatasets import getDataset


class DR_dataset(Dataset):
    def __init__(self):
        self.X = np.load("X_train.npy")
        self.y = np.load("y_train.npy")

        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class Net(nn.Module):
    

if __name__ == "__main__":
    dataSet = DR_dataset()
