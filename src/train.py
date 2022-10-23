import torch, torchvision
from torch.utils.data import Dataset, DataLoader, Subset 
from dataset import IndoorSceneDataset, tfms
import pandas as pd
import sys 
from engine import train_model

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 32

    train_df = pd.read_csv('dataset/train.csv')
    valid_df = pd.read_csv('dataset/valid.csv')

    # Take 20% of the dataset for experimentation
    train_ds = IndoorSceneDataset(train_df, tfms)
    valid_ds = IndoorSceneDataset(valid_df, tfms)

    # take 20% of the training dataset for experimentation
    train_ds = Subset(train_ds, range(0, len(train_ds), 5))
    valid_ds = Subset(valid_ds, range(0, len(valid_ds), 5))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 67)

    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(model, train_dl, valid_dl, optimizer, criterion, num_epochs=10)
    