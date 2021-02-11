import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import cv2
import os
    
IMG_SIZE = 512


def crop_image_from_gray(img, tol=7):
    # performs gray scale conversion, and performs round cropping
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #changing colorspace
    image = crop_image_from_gray(image) 
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) #resizing the imagae
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)  #Weighted addition between the image and the filter

    return image


data_frame = pd.read_csv("data/train.csv")

X = data_frame["id_code"]
y = data_frame["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

NUM_SAMP = 7
fig = plt.figure(figsize=(25, 16))
ctr = 0

# performs Ben Graham's method but with colored images and generates new data


for id_code in tqdm(X_train):
    # applying changes for training data

    path = f"data/train_images/" + str(id_code) + ".png"
    array = load_ben_color(path, sigmaX=30)
    image = Image.fromarray(array)
    image.save("data/processed/train images/" + str(id_code) + ".png")

for id_code in tqdm(X_test):
    # applying chages for training data

    path = f"data/train_images/" + str(id_code) + ".png"
    array = load_ben_color(path, sigmaX=30)
    image = Image.fromarray(array)
    image.save("data/processed/test images/" + str(id_code) + ".png")
