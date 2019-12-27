#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune the fc layer only for bilinear CNN.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python ./src/bilinear_cnn_fc.py --base_lr 0.01 \
        --batch_size 32 --epochs 100 --weight_decay 5e-4
"""


import os

import torch
import torchvision
import mvtec
import cub200
from efficientnet_pytorch import EfficientNet
from torch import nn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


from lib.non_local_embedded_gaussian import NONLocalBlock2D

class CELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.add(y1,y2)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SElayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SElayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BCNN(torch.nn.Module):
    """B-CNN for CUB200.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        #self.model = torchvision.models.resnet34(pretrained=True)#.features
        #print(self.model)
        self.node = 1920
        self.pixes = 14
        #model = EfficientNet.from_pretrained('efficientnet-b0')
        #model._fc = nn.Linear(in_features=1280,out_features=45,bias=True)
        
        self.features1 = torchvision.models.densenet201(pretrained=True).features[:6]
        self.features2 = torchvision.models.densenet201(pretrained=True).features[6:]
        #self.features3 = torchvision.models.densenet201(pretrained=True).features[8:]

        #self.features2 = torchvision.models.vgg19(pretrained=True).features
        #self.features2 = torch.nn.Sequential(self.features2.conv1, self.features2.bn1, self.features2.relu,
        #                                    self.features2.layer1, self.features2.layer2, self.features2.layer3,self.features2.layer4)
        #self.features1 = torch.nn.Sequential(*list(self.features1.children())
        #                                    [:-1])  # Remove pool5.
        #self.features2 = torch.nn.Sequential(*list(self.features2.children())
        #                                    [:-1])  # Remove pool5.
        #self.features2 = torch.nn.Sequential(list(self.features1)[-2], list(self.features1)[-1], list(self.features1)[-2],
        #                                     list(self.features1)[-1])
        #print(self.features2)
        #self.features1 = torch.nn.Sequential(model._conv_stem, model._bn0, *model._blocks,
        #                                    model._conv_head, model._bn1, model._avg_pooling, model._dropout)

        # Linear classifier.
        #self.relu0 = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ce1 = CELayer(channel=128)
        self.se = SElayer()
        self.nnl = NONLocalBlock2D(in_channels=1920)
        #self.ce2 = CELayer(channel=256)
        #self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=(1))
        #self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=(8, 8), stride=(8, 8), padding=(0, 0))
        self.fc1 = torch.nn.Linear(1920, 200)
        #self.relu1 = torch.nn.ReLU(inplace=True)
        #self.fc2 = torch.nn.Linear(1000, 200)
        #self.relu = torch.nn.ReLU(inplace=True)
        
        '''
        #print(layer_name)
        for features in [self.features1,self.features2]:
            n = len(list(features))
            print(n)
            for i,param in enumerate(features.parameters()):
            #print(i,param)
            #param.data  = self.features1[layer_name[i]]
                param.requires_grad = False
                if i > n:
                    break
        '''
        
        #freeze all previous layers.
        for param in self.features1.parameters():
            param.requires_grad = False
        #for param in self.features2.parameters():
        #    param.requires_grad = False
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal(self.fc1.weight.data)
        if self.fc1.bias is not None:
            torch.nn.init.constant(self.fc1.bias.data, val=0)
        #torch.nn.init.kaiming_normal(self.fc2.weight.data)
        #if self.fc2.bias is not None:
        #    torch.nn.init.constant(self.fc2.bias.data, val=0)

    def attenmap(self, X):
        N,c,w,h = X.size()
        X = X.view(N,c,w*h)
        X_atten = torch.bmm(torch.transpose(X, 1, 2),X) / (w*h)
        X_atten = torch.softmax(X_atten,dim=-1)
        #print(X_atten.size())
        X = torch.bmm(X,X_atten).view(N,c,w,h)
        return X

    def passthrough(self, X):
        X1 = X[:,:,::2,::2]
        X2 = X[:,:,1::2,::2]
        X3 = X[:,:,::2,1::2]
        X4 = X[:,:,1::2,1::2]
        X = torch.cat((X1,X2,X3,X4),0)
        return X

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        #Y = self.model(X)
        X = self.features1(X)
        #print(X.size())
        #for i in range(len(self.features2)):
        #    X = self.features2[i](X)
        #X = self.attenmap(X)
        X = self.ce1(X)
        X = self.se(X)*X
        X = self.features2(X)
        X = self.nnl(X)
        #print(X.size(1))
        #X = self.se(X)
        #X_avg1 = self.avgpool(X1).view(N,512)
        #X_avg2 = self.avgpool(X2).view(N,512)
        #print(torch.mean(X_avg1))
        #print(torch.mean(X_avg2))

        #assert X.size() == (N, self.node, self.pixes, self.pixes)
        #X = X.view(N, self.node, self.pixes**2)
        #X2 = X2.view(N, self.node, self.pixes**2)
        #print(torch.transpose(X,1,2).size())
        #X = torch.cat((X1,X2),1)
        
        #X_atten = torch.bmm(torch.transpose(X, 1, 2),X)
        #X_atten = torch.softmax(X_atten,dim=-1)
        #print(X_atten.size())
        #X = torch.bmm(X,X_atten).view(N,1920,14,14)
        
        #X = torch.bmm(X,torch.transpose(X, 1, 2)) / (self.pixes**2)  # Bilinear
        #X = torch.bmm(X,X1)
        #X_diag = torch.bmm(X1,torch.transpose(X2, 1, 2)) / (self.pixes**2)  # Bilinear
        #X = torch.bmm(X_acc,X1)
        #X_avgacc = self.avgpool1(X_acc).view(N,512)
        #X_avgdiag = self.avgpool1(X_diag).view(N,512)
        #print(torch.mean(X_avgacc))
        #print(torch.mean(X_avgdiag))
        
        #X_pcadiag = torch.tensor([]).cuda()
        #for i in range(X_acc.size()[-1]):
        #    #X_diag[:,i,i] = X_acc[:,i,i]
        #    a = X_acc[:,i,i].view(N,1) 
        #    c = X_diag[:,i,i].view(N,1)
        #    X_pcacc = torch.cat((X_pcacc,a),1)
        #    X_pcadiag = torch.cat((X_pcadiag,c),1)
        #X = torch.cat((X_avg1,X_avg2,X_pcacc*0.2,X_pcadiag*4),1)
        #print(torch.mean(X_pcacc))
        #print(torch.mean(X_pcadiag))
        #u,v,d = torch.svd(X)
        #X_diag = self.avgpool(X_diag)

        #assert X.size() == (N, self.node, self.node)
        #X = X.view(N, 1920*14*14)
        #X = self.avgpool(X).view(N,512)
        #X = torch.chunk(X, 64, dim=2)
        #print(X[0].size())
        #X = self.avgpool(X[0]).view(N,512)
        #X = self.conv1(X)
        #print(X.size())
        #X = torch.sqrt(X + 1e-5)
        #X = torch.nn.functional.normalize(X)
        #X = self.passthrough(X)
        X = self.attenmap(X)
        X = self.avgpool(X)
        X = self.fc1(X.view(N,1920))
        #X = self.fc2(self.relu1(X))
        assert X.size() == (N, 200)
        return X


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            [{'params': self._net.module.fc1.parameters()},
                            {'params': self._net.module.features2.parameters(), 'lr': 0.005}],
            #self._net.module.fc1.parameters(), 
            lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
        #train_data = mvtec.MVTEC(
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
        #test_data = mvtec.MVTEC(
            root=self._path['cub200'], train=False, download=True,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16,
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda()) #async=True))

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                # Save model onto disk.
                torch.save(self._net.state_dict(),
                           os.path.join(self._path['model'],
                                        'res_vgg_epoch_%d.pth' % (t + 1)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda()) #async=True))

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total

    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        train_data = cub200.CUB200(
        #train_data = mvtec.MVTEC(
            root=self._path['cub200'], train=True,
            transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4,
            pin_memory=True)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        required=True, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': os.path.join(project_root, 'data/cub200'),
        'model': os.path.join(project_root, 'model'),
    }
    #for d in path:
    #    assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
