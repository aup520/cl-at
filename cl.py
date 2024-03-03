import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from loader import get_loaders
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
data_list=[]
data_labellt=[]
train_loader, test_loader= get_loaders(dir_="./cifar-data",batch_size=128)
for cln_data, true_label in train_loader:
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    data_list.append(cln_data)
    data_labellt.append(true_label)
torch.save({'advdata_list': data_list, 'advdata_labellt': data_labellt}, './data/clntr.pth')
data_list=[]
data_labellt=[]
for cln_data, true_label in test_loader:
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    data_list.append(cln_data)
    data_labellt.append(true_label)
torch.save({'advdata_list': data_list, 'advdata_labellt': data_labellt}, './data/clnte.pth')
