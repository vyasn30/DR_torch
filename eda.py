import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        print(image_path)
        img = cv2.imread('data/train_images/' + str(image_path) +'.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
        plt.tight_layout()


train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

plot = train_df['diagnosis'].hist()


##plt.show()
print(train_df['diagnosis'].value_counts())

display_samples(train_df)
#plt.show()

