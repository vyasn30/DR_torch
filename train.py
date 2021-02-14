from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils    
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import models
import torch
from tqdm import tqdm
from generateDatasets import getDataset
from torchvision import models, transforms
import torch.optim as optim

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder("data/train", transform = transformations)
val_set = datasets.ImageFolder("data/valid", transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =16, shuffle=True)

model = models.densenet161(pretrained = True)

classifier_input = model.classifier.in_features
num_labels = 5

classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
if torch.cuda.is_available():
    model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0

    model.train()
    counter = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        counter += 1
        print(counter, "/", len(train_loader))

    model.eval()
    counter = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valloss = criterion(output, labels)
            val_loss += valloss.item()*inputs.size(0)
            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            counter += 1
            print(counter, "/", len(val_loader))
    
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))



torch.save(model.state_dict(), "drRichard.pth")
