import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile

trainList = os.listdir("data/processed/train images")

df = pd.read_csv("data/train.csv")

labels = np.load("y_train.npy")


#for training data

df.set_index("id_code", inplace = True)

for f in tqdm(os.listdir("data/processed/train images")):
    f = f[:-4]
    label = df.loc[f, "diagnosis"]
    if label == 0:
        copyfile("data/processed/train images/"+f+".png", "train/0"+f+".png")

    elif label == 1:
        copyfile("data/processed/train images/"+f+".png", "train/1"+f+".png")

    elif label == 2:
        copyfile("data/processed/train images/"+f+".png", "train/2"+f+".png"
                )
    elif label == 3:
        copyfile("data/processed/train images/"+f+".png", "train/3"+f+".png")

    elif label == 4:
        copyfile("data/processed/train images/"+f+".png", "train/4"+f+".png")


for f in tqdm(os.listdir("data/processed/test images")):
    f = f[:-4]
    label = df.loc[f, "diagnosis"]
    if label == 0:
        copyfile("data/processed/test images/"+f+".png", "valid/0"+f+".png")

    elif label == 1:
        copyfile("data/processed/test images/"+f+".png", "valid/1"+f+".png")

    elif label == 2:
        copyfile("data/processed/test images/"+f+".png", "valid/2"+f+".png"
                )
    elif label == 3:
        copyfile("data/processed/test images/"+f+".png", "valid/3"+f+".png")

    elif label == 4:
        copyfile("data/processed/test images/"+f+".png", "valid/4"+f+".png")
