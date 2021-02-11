from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils    
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

class DR_dataset(Dataset):

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.data_frame = pd.read_csv("data/train.csv")

        for f in tqdm(os.listdir("data/processed/train images")):
            im_frame = Image.open("data/processed/train images/"+f)

            np_frame = np.array(im_frame)
            self.X_train.append(np_frame)
        
        for f in tqdm(os.listdir("data/processed/test images")):
            im_frame = Image.open("data/processed/test images/"+f)
            np_frame = np.array(im_frame)
            self.X_test.append(np_frame)
        
        
        print(len(self.X_train))
        print(len(self.X_test))
        np.save("X_train.npy", np.array(self.X_train))
        np.save("X_test.npy",np.array(self.X_test))

        self.data_frame.set_index("id_code", inplace = True)
        for f in tqdm(os.listdir("data/processed/train images")):
            f = f[:-4]
            label = self.data_frame.loc[f, "diagnosis"]
            print(label)
            self.y_train.append(label)

        for f in tqdm(os.listdir("data/processed/test images")):
            f = f[:-4]

            label = self.data_frame.loc[f, "diagnosis"]
            print(label)
            self.y_test.append(label)

    
        
        print(len(self.y_train))
        print(len(self.y_test))
        np.save("y_train",np.array(self.y_train))
        np.save("y_test", np.array(self.y_test))


if __name__ == "__main__":
    dataSet = DR_dataset()
