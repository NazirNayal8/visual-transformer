import argparse
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from models.vt_resnet import vt_resnet18
from models.vt_resnet18 import VTResNet18
from torch.optim import Optimizer
from typing import Any, Callable

parser = argparse.ArgumentParser()

def random_seed():
    """
    Seed all random generators.
    """
    np.random.seed(8)
    torch.manual_seed(8)
    random.seed(8)

def train_epoch(model: nn.Module, optimizer: Optimizer, data_loader: Any, device: torch.device):
    """
    Trains a model for a single epoch using the data provided
    in the dataloader.

    Input:
    - model: a pytorch model (nn.Module)
    - optimizer: a pytorch optimizer (e.g. torch.optim.Adam)
    - data_loader: iterable dataloader with data and targets
    - device: the device on which the training shall happen.

    Output:
    - loss_history: a list that contains the loss resulting from every batch
    """
    total_samples = len(data_loader.dataset)
    
    model.train()
    model.to(device)

    loss_history = []
    for i, (data, target) in enumerate(tqdm.tqdm(data_loader)):
        
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + 
                  '/' + '{:5}'.format(total_samples) + 
                  ' (' +'{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' + 
                  '{:6.4f}'.format(loss.item()))
    return loss_history

def evaluate(model: nn.Module, data_loader: Any, device: torch.device, comment: str = ""):
    """
    Evaluates the accuracy of a model on the validation data provided
    in the given data loader.
    
    Input:
    - model: a pytorch model (nn.Module)
    - data_loader: an iterable dataloader giving data and labels
    - comment: (optional) adds a comment on the status at the end of 
        the evaluation
    """
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    loss_history = []

    with torch.no_grad():
        for data, target in tqdm.tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage ' + comment + ' loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
    accuracy = 100.0 * correct_samples / total_samples
    return accuracy, loss_history

def train(
    model: nn.Module, 
    optimizer: Optimizer, 
    train_data: Any, 
    valid_data: Any, 
    epochs: int, 
    lr: float,
    lr_decay: float, 
    decay_every: int, 
    weight_decay: float,
    optim: Callable,
    device: torch.device,
    evaluate_every: bool = True,
    plot_every: bool = False,
    optimize: bool = True,
    threshold_acc: float = 100.0,
    threshold_itr: float = 5
):
    """
    This is the main function used for training the model. 
    
    Input:
    - model: a pytorch model (nn.Module)
    - optimizer: a pytorch optimizer (e.g. torch.optim.Adam)
    - train_data: a data loader of the training data
    - valid_data: a data loader of the validation data
    - epochs: number of training epochs
    - lr: learning rate
    - lr_decay: the amount of decay to be applier to learning rate
        new value is (lr * lr_decay).
    - decay_every: denotes the period at which the learning rate is decayed. i.e
        every $decay_every epochs, lr is decayed by $lr_decay.
    - weight_decay: the weight decay used by the optimizer.
    - optim: the constructor of the Optimizer instance.
    - evaluate_every: a boolean that allows the evaluation training and validation
        accuracies after every epoch if set to True.
    - plot_every: a boolean. If true, plots loss curve after every epoch.
    
    Output:
    - valid_acc: final validation accuracy
    - train_acc: final testing accuracy
    - all_history: the loss history of all epochs concatenated
    """

    all_history = []
    train_acc = 0
    valid_acc = 0
    for i in range(epochs):

        if i % decay_every == 0:
            lr *= lr_decay
            optimizer = optim(model.parameters(), lr=lr, weight_decay=weight_decay)

        history = train_epoch(model, optimizer, train_data, device)
        all_history = all_history + history
        if plot_every:
            plt.plot(history)
            plt.show()
        
        print("Epoch " + str(i) + " done.")
        
        if evaluate_every:
            valid_acc, valid_hist = evaluate(model, valid_data, device, 'valid')
            train_acc, train_hist = evaluate(model, train_data, device, 'train')
            
            if optimize and i + 1 > threshold_itr and valid_acc < threshold_acc:
                print('Training Aborted due to Poor Performance.')
                break
                 
          
        
    valid_acc, valid_hist = evaluate(model, valid_data, device, 'valid')
    train_acc, train_hist = evaluate(model, train_data, device, 'train')

    return valid_acc, train_acc, all_history
    
