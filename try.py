from __future__ import print_function

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wideresnet import WideResNet
from advertorch.context import ctx_noparamgrad_and_eval
# from advertorch.test_utils import LeNet5
# from advertorch_examples.utils import get_mnist_train_loader
# from advertorch_examples.utils import get_mnist_test_loader
# from advertorch_examples.utils import TRAINED_MODEL_PATH
TRAINED_MODEL_PATH="./model"
from loader import get_loaders
from resnet import ResNet18

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train contiune attack')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")
    nb_epoch = 20
    x=["cw_"]
    x=["cln","fgsm","pgd_1","pgd_2","pgdf","cw","AAk"]
    # x=["cln","fgsm","pgdf","cw","AAk"]
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
    # train_loader, test_loader= get_loaders(
    #     dir_="./cifar-data",batch_size=args.train_batch_size)
    
    # model = WideResNet(34, 10, widen_factor=10, dropRate=0.0, normalize=False,
    #         activation='ReLU', softplus_beta=1.0)
    model = ResNet18()
    model.to(device)
    params = model.parameters()
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    test_list=[]
    # num=int(50000/128/50)
    # print(num)
    # memory=[]
    # gener=[]
    # p=range(5)
    for t in range(len(x)):
        loaded_data = torch.load('./data/{}te.pth'.format(x[t]))
        tedata_list = loaded_data['advdata_list']
        telabel_list = loaded_data['advdata_labellt']
        test_list.append([tedata_list,telabel_list])
    for t in range(len(x)):
        model = ResNet18()
        model.to(device)
        params = model.parameters()
        optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
        loaded_data = torch.load('./data/{}tr.pth'.format(x[t]))
        trdata_list =  loaded_data['advdata_list']
        trlabel_list = loaded_data['advdata_labellt']
        # datadl=list(zip(trdata_list, trlabel_list))
        # gener=datadl+memory
        for epoch in range(nb_epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(zip(trdata_list, trlabel_list)):
                data, target = data.to(device), target.to(device)
                ori = data
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(
                    output, target, reduction='mean')
                loss.backward()
                optimizer.step()
                if batch_idx % args.log_interval == 0 or batch_idx == 49999:
                    print('TASK{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        t,epoch, batch_idx *
                        len(data), 50000,
                        100. * batch_idx / len(trdata_list), loss.item()))
            
            model.eval()
            for i in range(len(x)):
                test_clnloss = 0
                clncorrect = 0
                for batch_idx, (clndata, target) in enumerate(zip(test_list[i][0], test_list[i][1])):
                    clndata, target = clndata.to(device), target.to(device)
                    with torch.no_grad():
                        output = model(clndata)
                    test_clnloss += F.cross_entropy(
                        output, target, reduction='sum').item()
                    pred = output.max(1, keepdim=True)[1]
                    clncorrect += pred.eq(target.view_as(pred)).sum().item()
                test_clnloss /= 10000
                print('\nTASK{} Test set: avg cln loss: {:.4f},'
                    ' cln acc: {}/{} ({:.0f}%)'.format(
                        i,test_clnloss, clncorrect, 10000,
                        100. * clncorrect / 10000))
            print()
        