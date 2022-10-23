from torch.utils.data import Dataset, DataLoader, Subset 
import pandas as pd 
import torch, torchvision
from PIL import Image 
from torchvision import transforms 

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class IndoorSceneDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label']

if __name__=='__main__':
    df = pd.read_csv('dataset/train.csv')
    ds = IndoorSceneDataset(df, transform=tfms)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for xb, yb in dl:
        print(xb.shape, yb.shape)
        break
