from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
# from advertorch.test_utils import LeNet5
# from advertorch_examples.utils import get_mnist_train_loader
# from advertorch_examples.utils import get_mnist_test_loader
# from advertorch_examples.utils import TRAINED_MODEL_PATH
TRAINED_MODEL_PATH="./model"
from loader import get_loaders
from resnet import ResNet50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln", help="cln | adv")
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 20
        model_filename = "cr_clntrained.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 90
        model_filename = "cr_advtrained.pt"
    else:
        raise

    train_loader, test_loader= get_loaders(
        dir_="./cifar-data",batch_size=args.train_batch_size)
    

    model = ResNet50()
    model.to(device)
    params = model.parameters()
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)

    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
    max1=0
    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            max1=max(torch.max(data),max1)
            # data=data.float()/255.0
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should NOT be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data = adversary.perturb(data, target)
                

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        model.eval()
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    
    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))