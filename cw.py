import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from loader import get_loaders
from resnet import ResNet50
import torch.nn.functional as F
TRAINED_MODEL_PATH="./model"
filename="cr_clntrained.pt"
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet50()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
model.to(device)
model.eval()
loaded_data = torch.load('./data/cwte.pth')
tedata_list = loaded_data['advdata_list']
telabel_list = loaded_data['advdata_labellt']
loaded_data = torch.load('./data/cwtr.pth')
trdata_list =  loaded_data['advdata_list']
trlabel_list = loaded_data['advdata_labellt']
advdata_list=[]
advdata_labellt=[]

for cln_data, true_label in zip(trdata_list,trlabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    new_label=model(cln_data)
    new_label=torch.argmax(new_label,1)
    advdata_list.append(cln_data)
    advdata_labellt.append(new_label)
    print(1)
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/cw_tr.pth')
advdata_list=[]
advdata_labellt=[]
for cln_data, true_label in zip(tedata_list,telabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    new_label=model(cln_data)
    new_label=torch.argmax(new_label,1)
    advdata_list.append(cln_data)
    advdata_labellt.append(new_label)
    print(1)
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/cw_te.pth')