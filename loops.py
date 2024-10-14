import torch
from tqdm import tqdm
import numpy as np
import parameters as param
import os
from torch import nn

def train_loop(dataloader, model, loss_fn, optimizer, platform):
    size = len(dataloader.dataset)

    train_loss, correct = 0, 0
    model.train()

    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Move data on GPU
        X = X.to(platform)
        y = y.view(-1).type(torch.int64).to(platform)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    correct /= size
    loss, current = loss.item(), batch * len(X)
    print(f"Training Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    return correct, train_loss

def trainTriplet_loop(dataloader, model, loss_fn, optimizer, platform, lambd):
    size = len(dataloader.dataset)

    train_loss, correct = 0, 0
    loss_tr = nn.TripletMarginLoss()
    model.train()

    for batch, (X, y, pos, neg) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Move data on GPU
        X = X.to(platform)
        y = y.view(-1).type(torch.int64).to(platform)
        pos = pos.to(platform)
        neg = neg.to(platform)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred_anc, emb_anc = model(X)
        pred_pos, emb_pos = model(pos)
        pred_neg, emb_neg = model(neg)

        loss = loss_fn(pred_anc, y) + loss_tr(emb_anc, emb_pos, emb_neg)*lambd


        train_loss += loss.item()
        correct += (pred_anc.argmax(1) == y).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    correct /= size
    loss, current = loss.item(), batch * len(X)
    print(f"Training Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    return correct, train_loss

def val_loop(dataloader, model, loss_fn, platform):
    size = len(dataloader.dataset)

    val_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data on GPU
            X = X.to(platform)
            y = y.to(platform)
            # Prediction on the validation set
            pred, _ = model(X)
            loss = loss_fn(pred, y)
            val_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

    val_loss /= len(dataloader)
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return correct, val_loss

def test_loop(dataloader, model, loss_fn, platform, results, original_test, emb_dir):
    size = len(dataloader.dataset)

    test_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for batch, (X, y, ID) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data on GPU
            X = X.to(platform)
            y = y.to(platform)
            # Prediction on the validation set
            pred, emb = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            results = results.append({'ID': ID[0], 'original': original_test[ID[0]], 'label': y[0].cpu().numpy(), 'prediction': pred[0,1].cpu().numpy()}, ignore_index=True)
            # save embedding
            np.save(file=os.path.join(emb_dir, str(ID[0]) + '.npy'), arr=emb.cpu().numpy().flatten())

    test_loss /= len(dataloader)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss, results
