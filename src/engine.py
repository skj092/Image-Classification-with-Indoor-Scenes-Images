from tqdm import tqdm
import numpy as np 
import torch
from sklearn.metrics import accuracy_score

def train_model(model, train_dl, valid_dl, optimizer, criterion, num_epochs=10, device='cpu'):
    train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for x, y in tqdm(train_dl):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            accuracy = accuracy_score(y.detach().cpu().numpy(), y_pred.argmax(dim=1).detach().cpu().numpy())
            train_acc.append(accuracy)
        model.eval()
        for x, y in tqdm(valid_dl):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            accuracy = accuracy_score(y.detach().cpu().numpy(), y_pred.argmax(dim=1).detach().cpu().numpy())
            valid_losses.append(loss.item())
            valid_acc.append(accuracy)
        print(f'Epoch: {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Train Accuracy: {np.mean(train_acc):.4f}, Valid Loss: {np.mean(valid_losses):.4f}, Valid Accuracy: {np.mean(valid_acc):.4f}')
    
        