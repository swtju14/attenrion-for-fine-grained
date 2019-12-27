#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_all.py --base_lr 0.05 \
        --batch_size 64 --epochs 100 --weight_decay 5e-4
"""


import os

import torch
import torchvision

import cub200

from torch.autograd import Variable
import torch.onnx
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



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
        self.features1 = torchvision.models.densenet201(pretrained=False).features
        #self.features2 = torchvision.models.vgg19(pretrained=False).features
        #self.features = torch.nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
        #                                               self.model.layer1, self.model.layer2, self.model.layer3)
        #self.features1 = torch.nn.Sequential(*list(self.features1.children())
        #                                    [:-1])  # Remove pool5.
        #self.features2 = torch.nn.Sequential(*list(self.features2.children())
        #                                    [:-1])  # Remove pool5.
        # Linear classifier.
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size = (1,1))
        #self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size = (1))
        self.fc1 = torch.nn.Linear(1920, 200)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features1(X)
        #X2 = self.features2(X)
        #X_avg1 = self.avgpool(X1).view(N,512)
        #X_avg2 = self.avgpool(X2).view(N,512)
        #assert X1.size() == (N, 512, 28, 28)
        X = X.view(N, 1920, 14**2)
        #X2 = X2.view(N, 512, 28**2)
        X_atten = torch.bmm(torch.transpose(X,1,2),X)
        X_atten = torch.softmax(X_atten,dim=-1)
        X = torch.bmm(X,X_atten).view(N,1920,14,14)

        #X = torch.bmm(X,torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        #X = torch.bmm(X,X1)
        #X_diag = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (28**2)  # Bilinear
        #X_avgacc = self.avgpool1(X_acc).view(N,512)
        #X_avgdiag = self.avgpool1(X_diag).view(N,512)
        #X1 = torch.bmm(X_cc,X1)
        #X_pcacc = torch.tensor([]).cuda()
        #X_pcadiag = torch.tensor([]).cuda()
        #for i in range(X_acc.size()[-1]):
        #    a = X_acc[:,i,i].view(N,1)
        #    c = X_diag[:,i,i].view(N,1)
        #    X_pcacc = torch.cat((X_pcacc,a),1)
        #    X_pcadiag = torch.cat((X_pcadiag,a),1)
        #assert X2.size() == (N, 512, 512)
        #X1 = self.avgpool(X1)
        #X = torch.cat((X_avg1,X_avg2,X_avgacc*4,X_avgdiag*4,X_pcacc*0.2,X_pcadiag*4),1)
        #X = self.avgpool(X).view(N, 512)
        #X = X.view(N,512**2)
        #X = torch.sqrt(X + 1e-5)
        #X = torch.nn.functional.normalize(X)
        X = self.avgpool(X).view(N,1920)
        X = self.fc1(X)
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
        # Load the model from disk.
        self._net.load_state_dict(torch.load(self._path['model']))
        #self._net = torch.load(self._path).cuda()
        print(self._net)
        #dummy_input = Variable(torch.randn(10, 3, 448, 448)).cuda()
        #model = torchvision.models.vgg16(pretrained=True).cuda()
        #torch.onnx.export(self._net, dummy_input, "/root/disk/kevin/models/bcnn.onnx", verbose=True)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            #[{'params': self._net.features1[8:12].parameters()},
            #    {'params': self._net.fc1.parameters(), 'lr': 1e-2}],
            self._net.parameters(), 
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
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
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
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model for fine-tuning.')
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
        'model': os.path.join(project_root, 'model', args.model),
    }
    #for d in path:
    #    if d == 'model':
    #        assert os.path.isfile(path[d])
    #    else:
    #        assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()

