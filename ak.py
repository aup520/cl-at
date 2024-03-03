import os
import argparse
# from absl import app, flags
# from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from resnet import ResNet50
from carlini_wagner_l2 import carlini_wagner_l2
# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# from cleverhans.torch.attacks.projected_gradient_descent import (
#     projected_gradient_descent,
# )

# FLAGS = flags.FLAGS

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
loaded_data = torch.load('./data/clnte.pth')
tedata_list = loaded_data['advdata_list']
telabel_list = loaded_data['advdata_labellt']
loaded_data = torch.load('./data/clntr.pth')
trdata_list =  loaded_data['advdata_list']
trlabel_list = loaded_data['advdata_labellt']
advdata_list=[]
advdata_labellt=[]

for cln_data, true_label in zip(trdata_list,trlabel_list):
    # print(set(cln_data.keys()))
    # print(label)
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    # x_fgm = fast_gradient_method(model,cln_data,8/255, np.inf)
    x_cw=carlini_wagner_l2(model,cln_data,10,targeted=False, y=true_label)
    new_label=model(x_cw)
    new_label=torch.argmax(new_label,1)
    # advdata_list.append(x_fgm)
    advdata_list.append(x_cw)
    advdata_labellt.append(true_label)
    print(1)
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/cw1tr.pth')
advdata_list=[]
advdata_labellt=[]
for cln_data, true_label in zip(tedata_list,telabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    # x_fgm = fast_gradient_method(model,cln_data, 8/255, np.inf)
    x_cw=carlini_wagner_l2(model,cln_data,10)
    new_label=model(x_cw)
    new_label=torch.argmax(new_label,1)
    # advdata_list.append(x_fgm)
    advdata_list.append(x_cw)
    # advdata_labellt.append(true_label)
    advdata_labellt.append(true_label)
    print(1)
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/cw1te.pth')