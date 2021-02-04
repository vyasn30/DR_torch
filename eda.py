import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

plot = train_df['diagnosis'].hist()


plt.show()
print(train_df['diagnosis'].value_counts())
