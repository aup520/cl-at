import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from loader import get_loaders
from resnet import ResNet50


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
import advertorch
loaded_data = torch.load('./data/clnte.pth')
tedata_list = loaded_data['advdata_list']
telabel_list = loaded_data['advdata_labellt']
loaded_data = torch.load('./data/clntr.pth')
trdata_list =  loaded_data['advdata_list']
trlabel_list = loaded_data['advdata_labellt']
from autoattack import AutoAttack
# adversary3 = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
# adversary= advertorch.attacks.GradientAttack(
#     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255, clip_min=0.0, clip_max=1.0,
#     targeted=False)
# adversary1 = advertorch.attacks.LinfPGDAttack(
#     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
#     nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0.0, clip_max=1.0,
#     targeted=False)
adversary1 = advertorch.attacks.L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.0,
    nb_iter=10, eps_iter=0.25, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)
adversary = advertorch.attacks.L1PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=10.0,
    nb_iter=10, eps_iter=0.5, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)
# adversary2 = advertorch.attacks.SparseL1DescentAttack(
#     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
#     nb_iter=30, eps_iter=2/255, rand_init=False, clip_min=0.0, clip_max=1.0,l1_sparsity=0.95,
#     targeted=False)
# adversary3= advertorch.attacks.MomentumIterativeAttack(
#     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
#     nb_iter=10, eps_iter=2/255, clip_min=0.0, clip_max=1.0,
#     targeted=False)
# adversary3= advertorch.attacks.CarliniWagnerL2Attack(
#     model, num_classes=10, confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9, max_iterations=20, abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0)
# adversary3= advertorch.attacks.LBFGSAttack(
#      model, num_classes=10,batch_size=128, binary_search_steps=9, max_iterations=100, initial_const=0.01, clip_min=0, clip_max=1, loss_fn=None, targeted=False)
# adversary3=advertorch.attacks.LocalSearchAttack(
#     model, clip_min=0.0, clip_max=1.0, p=1.0, r=1.5, loss_fn=None, d=5, t=5, k=1, round_ub=10, seed_ratio=0.1, max_nb_seeds=128, comply_with_foolbox=False, targeted=False
# )
advdata_list=[]
advdata_list1=[]
advdata_list2=[]
advdata_list3=[]
advdata_labellt=[]
advdata_labellt1=[]
advdata_labellt2=[]
advdata_labellt3=[]

i=0
for cln_data, true_label in zip(trdata_list,trlabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    advdata = adversary.perturb(cln_data, true_label)
    advdata_list.append(advdata)
    advdata_labellt.append(true_label)
    
    advdata1 = adversary1.perturb(cln_data, true_label)
    advdata_list1.append(advdata1)
    advdata_labellt1.append(true_label)
   
    # advdata2 = adversary2.perturb(cln_data, true_label)
    # advdata_list2.append(advdata2)
    # advdata_labellt2.append(true_label)
    
    # advdata3 = adversary3.run_standard_evaluation(cln_data, true_label, bs=128)
    # advdata_list3.append(advdata3)
    # type(advdata3)
    # advdata_labellt3.append(true_label)
    i=i+1
    print(i)
    
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/pgd_1tr.pth')
torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/pgd_2tr.pth')
# # torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/gd.pth')
advdata_list=[]
advdata_labellt=[]
# # torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/pgdtr.pth')
advdata_list1=[]
advdata_labellt1=[]
# # torch.save({'advdata_list': advdata_list2, 'advdata_labellt': advdata_labellt2}, './data/SparseL1tr.pth')
# # advdata_list2=[]
# # advdata_labellt2=[]
# # torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/Momentumtr.pth')
# torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/AAktr.pth')

advdata_list3=[]
advdata_labellt3=[]

for cln_data, true_label in zip(tedata_list,telabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    advdata = adversary.perturb(cln_data, true_label)
    advdata_list.append(advdata)
    advdata_labellt.append(true_label)
    advdata1 = adversary1.perturb(cln_data, true_label)
    advdata_list1.append(advdata1)
    advdata_labellt1.append(true_label)
    # advdata2 = adversary2.perturb(cln_data, true_label)
    # advdata_list2.append(advdata2)
    # advdata_labellt2.append(true_label)
    # advdata3 = adversary3.perturb(cln_data, true_label)
    # advdata3 = adversary3.run_standard_evaluation(cln_data, true_label, bs=128)
    # advdata_list3.append(advdata3)
    # advdata_labellt3.append(true_label)
    i=i+1
    print(i)
# # torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/pgdte.pth')
# # advdata_list=[]
# # advdata_labellt=[]
# # torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/gdte.pth')
# # advdata_list1=[]
# # advdata_labellt1=[]
# # torch.save({'advdata_list': advdata_list2, 'advdata_labellt': advdata_labellt2}, './data/SparseL1te.pth')
# # advdata_list2=[]
# # advdata_labellt2=[]
# torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/AAkte.pth')
torch.save({'advdata_list': advdata_list, 'advdata_labellt': advdata_labellt}, './data/pgd_1te.pth')
torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/pgd_2te.pth')
#
# torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/cwte.pth')
# loaded_data = torch.load('./data/pgd.pth')
# data_list = loaded_data['advdata_list']
# label_list = loaded_data['advdata_labellt']




