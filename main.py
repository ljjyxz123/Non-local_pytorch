'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.autograd import Variable

from dataloaders import make_data_loader
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                        training (default: auto)')
parser.add_argument('--base-size', type=int, default=128, # 513,
                        help='base image size')
parser.add_argument('--crop-size', type=int, default=128, # 513,
                        help='crop image size')
args = parser.parse_args()

print('torch.cuda.is_available(): ', torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# batch_size = 1
# c, h, w = 1, 10, 10
# nb_classes = 3
# x = torch.randn(batch_size, c, h, w)
# target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)
#
# model = nn.Conv2d(c, nb_classes, 3, 1, 1)
# criterion = nn.CrossEntropyLoss()
#
# output = model(x)
# print('size: ', output.size(), target.size())
# loss = criterion(output, target)
# loss.backward()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomCrop(128, 64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_seg = transforms.Compose([
    # transforms.RandomCrop(128, 64), # This can cause very low accuracy, pay attention!!!
    transforms.ToTensor(),
])

# load PASCAL VOC 2012 Segmentation dataset
seg_dataset = VOCSegmentation('~/DeLightCMU/CVPR-Prep/Non-local_pytorch/data',
                              year = "2012",
                              image_set='train',
                              download=False,
                              transform=transform_seg,
                              target_transform=transform_seg)
print('VOCSeg ends.')
seg_loader  = DataLoader(seg_dataset, batch_size=1)
print('seg_loader ends.')

# input_num = 0
# for input, target in seg_loader:
#     # print('for loop.')
#     print(input.size(), target.size())
#     input_num = input_num + 1
# print('input_num: ', input_num)
# exit(-1)

# for i, data in enumerate(seg_loader):
#     # load from dataloader
#     inputs, labels = data
#
#     # transform to Variable type
#     inputs, labels = Variable(inputs), Variable(labels)
#
#     # print(i, "inputs", inputs.data.size(), "labels", labels.data.size())
#     print(i, "labels", labels.data)
# exit(-1)

# For CIFAR10
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)


# For PASCAL VOC 2012
trainset = VOCSegmentation(root='~/DeLightCMU/CVPR-Prep/Non-local_pytorch/data',
                           year = "2012",
                           image_set='train',
                           download=False,
                           transform=transform_seg,
                           target_transform=transform_seg)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)

testset = VOCSegmentation(root='~/DeLightCMU/CVPR-Prep/Non-local_pytorch/data',
                           year = "2012",
                           image_set='val',
                           download=False,
                           transform=transform_seg,
                           target_transform=transform_seg)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)

# Define Dataloader
kwargs = {'num_workers': 2, 'pin_memory': True}
train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
# print('device: ', device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     targets = torch.squeeze(targets, 1)
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)
    for batch_idx, sample in enumerate(tbar):
        image, targets = sample['image'], sample['label']
        if_use_cuda = False
        if if_use_cuda:
            image, targets = image.cuda(), targets.cuda()
        optimizer.zero_grad()
        # print('image.size(): ', image.size())
        outputs = net(image)
        # targets = torch.where(targets > 30, torch.full_like(targets, 0), targets)
        # print('outputs: ', outputs)
        # print('targets: ', torch.max(targets), torch.min(targets))
        # print('main output: ', outputs.size())
        # print('main targets: ', targets.size())
        targets = targets.long()
        # loss = criterion(outputs, targets)
        # print('outputs and targets: ', outputs.size(), targets.size())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss / (batch_idx + 1)))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        #     inputs, targets = inputs.to(device), targets.to(device)
        tbar = tqdm(test_loader)
        num_img_tr = len(test_loader)
        for batch_idx, sample in enumerate(tbar):
            image, targets = sample['image'], sample['label']
            if_use_cuda = False
            if if_use_cuda:
                image, targets = image.cuda(), targets.cuda()

            # outputs = net(inputs)
            outputs = net(image)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
