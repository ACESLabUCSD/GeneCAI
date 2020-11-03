'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd.function import Function
from torch.nn.parameter import Parameter

import numpy as np
import sys
from layers import Mask_Layer, Conv2d_sparse, Linear_sparse, decomposed_conv, svd_conv, wrap_layer

#---------------------- non-pruned ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.mask1 = Mask_Layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = nn.ReLU(inplace=True)
        self.mask2 = Mask_Layer(planes)
        self.layers = [self.conv1, self.bn1, self.relu1, self.mask1, self.conv2, self.bn2, self.relu2, self.mask2]

    def forward(self, x):
        out = self.mask1(self.relu1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        if hasattr(self,'shortcut'):
            out += self.shortcut(x)
        elif hasattr(self,'downsample'):
            out += self.downsample(x)
        out = self.mask2(self.relu2(out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Cifar10ResNet(nn.Module):
    def __init__(self, block, n, num_classes=10):
        super(Cifar10ResNet, self).__init__()
        self.in_planes = 16
        num_blocks = n
        

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.mask1 = Mask_Layer(16)
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
        self.avg_pool2d1 = nn.AvgPool2d(8)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        # self.features = nn.Sequential(*[self.conv1, self.bn1, self.relu1, self.mask1, self.layer1, self.layer2, self.layer3, self.avg_pool2d1])
        # self.classifier = nn.Sequential(*[self.linear])


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.mask1(self.relu1(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool2d1(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#---------------------- structured pruning
class BasicBlockMask(nn.Module):
    def __init__(self, orig_block, decomposed = False, wrap=False):
        super(BasicBlockMask, self).__init__()
        if decomposed:
            if wrap:
                self.conv1 = wrap_layer(orig_block.conv1)
            else:
                self.conv1 = decomposed_conv(orig_block.conv1)
            self.bn1 = orig_block.bn1
            self.relu1 = orig_block.relu1
            self.mask1 = orig_block.mask1
            if wrap:
                self.conv2 = wrap_layer(orig_block.conv2)
            else:
                self.conv2 = decomposed_conv(orig_block.conv2)
            self.bn2 = orig_block.bn2
            self.shortcut = orig_block.shortcut

            self.relu2 = orig_block.relu2
            self.mask2 = orig_block.mask2
            self.layers = [self.conv1, self.bn1, self.relu1, self.mask1, self.conv2, self.bn2, self.relu2, self.mask2]
        
        else:
            if wrap:
                self.conv1 = wrap_layer(orig_block.conv1)
            else:
                self.conv1 = orig_block.conv1
            self.bn1 = orig_block.bn1
            self.relu1 = orig_block.relu1
            self.mask1 = orig_block.mask1
            if wrap:
                self.conv2 = wrap_layer(orig_block.conv2)
            else:
                self.conv2 = orig_block.conv2
            self.bn2 = orig_block.bn2
            self.shortcut = orig_block.shortcut

            self.relu2 = orig_block.relu2
            self.mask2 = orig_block.mask2
            self.layers = [self.conv1, self.bn1, self.relu1, self.mask1, self.conv2, self.bn2, self.relu2, self.mask2]
            
    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        if self.mask1 is not None:
            out = self.mask1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        if self.mask2 is not None:
            out = self.mask2(out)
        return out
    

class BottleneckMask(nn.Module):
    def __init__(self, orig_block, decomposed = False, wrap=False):
        super(BottleneckMask, self).__init__()
        self.decomposed = decomposed
        
        if decomposed:
            if wrap:
                self.conv1 = wrap_layer(orig_block.conv1)
            else:
                self.conv1 = svd_conv(orig_block.conv1)
            self.bn1 = orig_block.bn1
            if hasattr(orig_block.conv1,'out_channels'):
                self.mask1 = Mask_Layer(orig_block.conv1.out_channels, True)
            else:
                self.mask1 = Mask_Layer(orig_block.conv1.n_channel_out, True)
                
            if wrap:
                self.conv2 = wrap_layer(orig_block.conv2)
            else:
                self.conv2 = decomposed_conv(orig_block.conv2)
            self.bn2 = orig_block.bn2
            if hasattr(orig_block.conv2,'out_channels'):
                self.mask2 = Mask_Layer(orig_block.conv2.out_channels, True)
            else:
                self.mask2 = Mask_Layer(orig_block.conv2.n_channel_out, True)
            if wrap:
                self.conv3 = wrap_layer(orig_block.conv3)
            else:
                self.conv3 = svd_conv(orig_block.conv3)
            self.bn3 = orig_block.bn3
            if hasattr(orig_block.conv3,'out_channels'):
                self.mask3 = Mask_Layer(orig_block.conv3.out_channels, True)
            else:
                self.mask3 = Mask_Layer(orig_block.conv3.n_channel_out, True)
            
            self.downsample = orig_block.downsample
        else:
            if wrap:
                self.conv1 = wrap_layer(orig_block.conv1)
            else:
                self.conv1 = orig_block.conv1
            self.bn1 = orig_block.bn1
            if hasattr(orig_block.conv1,'out_channels'):
                self.mask1 = Mask_Layer(orig_block.conv1.out_channels, True)
            else:
                self.mask1 = Mask_Layer(orig_block.conv1.n_channel_out, True)
            if wrap:
                self.conv2 = wrap_layer(orig_block.conv2)
            else:
                self.conv2 = orig_block.conv2
            self.bn2 = orig_block.bn2
            if hasattr(orig_block.conv2,'out_channels'):
                self.mask2 = Mask_Layer(orig_block.conv2.out_channels, True)
            else:
                self.mask2 = Mask_Layer(orig_block.conv2.n_channel_out, True)
            if wrap:
                self.conv3 = wrap_layer(orig_block.conv3)
            else:
                self.conv3 = orig_block.conv3
            self.bn3 = orig_block.bn3
            if hasattr(orig_block.conv3,'out_channels'):
                self.mask3 = Mask_Layer(orig_block.conv3.out_channels, True)
            else:
                self.mask3 = Mask_Layer(orig_block.conv3.n_channel_out, True)
            
            self.downsample = orig_block.downsample
        
    def forward(self, x):
        residual = x
        out = self.mask1(F.relu(self.bn1(self.conv1(x))))
        out = self.mask2(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        if not(self.downsample is None):
            residual = self.downsample(x)
        out += residual
        out = self.mask3(F.relu(out))
        return out


class Cifar10ResNetMask(nn.Module):
    def __init__(self, orig_blocks, decomposed=False):
        super(Cifar10ResNetMask, self).__init__()
        
        self.conv1 = orig_blocks['conv1']
        
        if decomposed:
            assert type(self.conv1) == decomposed_conv 
        self.bn1 = orig_blocks['bn1']
        self.relu1 = orig_blocks['relu1']
        self.mask1 = orig_blocks['mask1']
        self.layer1 = orig_blocks['layer1']
        self.layer2 = orig_blocks['layer2']
        self.layer3 = orig_blocks['layer3']
        self.avg_pool2d1 = orig_blocks['avg_pool2d1']
        self.linear = orig_blocks['linear']

    def forward(self, x):
        out = self.mask1(self.relu1(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool2d1(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetMask(nn.Module):
    def __init__(self, orig_blocks, decomposed=False):
        super(ResNetMask, self).__init__()
        print(orig_blocks.keys())
    
        self.conv1 = orig_blocks['conv1']
        if decomposed:
            assert type(self.conv1) == decomposed_conv 
        self.bn1 = orig_blocks['bn1']
        if not 'mask1' in orig_blocks.keys():
            if decomposed:
                self.mask1 = Mask_Layer(orig_blocks['conv1'].last_conv.out_channels, True)
            else:
                self.mask1 = Mask_Layer(orig_blocks['conv1'].out_channels, True)
        else:
            self.mask1 = orig_blocks['mask1']
        self.maxpool = orig_blocks['maxpool']
        self.layer1 = orig_blocks['layer1']
        self.layer2 = orig_blocks['layer2']
        self.layer3 = orig_blocks['layer3']
        self.layer4 = orig_blocks['layer4']
        self.avgpool = orig_blocks['avgpool']
        self.fc = orig_blocks['fc']

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mask1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#----------------------- non-structured pruning
class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(SparseBottleneck, self).__init__()
        self.conv1 = Conv2d_sparse(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_sparse(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_sparse(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_sparse(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SparseResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d_sparse(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#------------------------ model construction
def SparseResNet50(num_classes=10):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet56(num_classes=10):
    return Cifar10ResNet(BasicBlock, n=9, num_classes=num_classes)


def ResNet110(num_classes=10):
    return Cifar10ResNet(BasicBlock, n=18, num_classes=num_classes)