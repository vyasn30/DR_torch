from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils    
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def getDataset():
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        data_frame = pd.read_csv("data/train.csv")

        for f in tqdm(os.listdir("data/processed/train images")):
            im_frame = Image.open("data/processed/train images/"+f)

            np_frame = np.array(im_frame)
            X_train.append(np_frame)
        
        for f in tqdm(os.listdir("data/processed/test images")):
            im_frame = Image.open("data/processed/test images/"+f)
            np_frame = np.array(im_frame)
            X_test.append(np_frame)
        
        
        print(len(X_train))
        print(len(X_test))
        X_train = np.array(X_train)
        X_test = np.array(X_test)

#        np.save("X_train.npy", X_train)
 #       np.save("X_test.npy",X_test)

        data_frame.set_index("id_code", inplace = True)
        for f in tqdm(os.listdir("data/processed/train images")):
            f = f[:-4]
            label = data_frame.loc[f, "diagnosis"]
            print(label)
            y_train.append(label)

        for f in tqdm(os.listdir("data/processed/test images")):
            f = f[:-4]

            label = data_frame.loc[f, "diagnosis"]
            print(label)
            y_test.append(label)

        y_train = np.array(y_train)
        y_test = np.array(y_test)
  #      np.save("y_train.npy", y_train)
   #     np.save("y_test.npy", y_test)
        


        return X_train, y_train
    


if __name__ == "__main__":
    getDataset()
