import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd.function import Function
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import sys

def convert_to_np(p):
    if not type(p)==np.ndarray:
        return p.detach().cpu().numpy()
    else:
        return p


def wrap_layer(layer):
    layers = []
    if isinstance(layer, decomposed_conv):
        r0 = int(np.sum(np.ravel(convert_to_np(layer.mask1.mask.data))+0.01))
        r1 = int(np.sum(np.ravel(convert_to_np(layer.mask2.mask.data))+0.01))
        layer.ranks = [r0,r1]

        w1 = convert_to_np(layer.first_conv.weight.data)[0:layer.ranks[0],:,:,:]
        w2 = convert_to_np(layer.core_conv.weight.data)[0:layer.ranks[1],0:layer.ranks[0],:,:]
        w3 = convert_to_np(layer.last_conv.weight.data)[:,0:layer.ranks[1],:,:]
        
        if layer.last_conv.bias is not None:
            bias = convert_to_np(layer.last_conv.bias.data)
        else:
            bias = None
        
        if layer.ranks[0]==layer.n_channel_in and layer.ranks[1]==layer.n_channel_out:
            w = np.dot(w2.transpose(0,2,3,1),np.squeeze(w1)) #r2,k,k,r1 * r1,c -> r2,k,k,c
            w = np.dot(w.transpose(1,2,3,0),np.squeeze(w3).transpose()) #k,k,c,r2 * r2,f ->k,k,c,f
            w = w.transpose(3,2,0,1)
            
            out_layer = nn.Conv2d(in_channels = w.shape[1],
                                  out_channels = w.shape[0],
                                  kernel_size = w.shape[2],
                                  stride = layer.core_conv.stride,
                                  padding = layer.core_conv.padding,
                                  bias = not(bias is None))
            set_module_params(out_layer, {'weight':w,'bias':bias})
            return out_layer
            
        elif layer.ranks[0]==layer.n_channel_in:
            w = np.dot(w2.transpose(0,2,3,1),np.squeeze(w1)) #r2,k,k,r1 * r1,c -> r2,k,k,c
            w = w.transpose(0,3,1,2)#r2,c,k,k
            
            core_layer = nn.Conv2d(in_channels = w.shape[1],
                                  out_channels = w.shape[0],
                                  kernel_size = w.shape[2],
                                  stride = layer.core_conv.stride,
                                  padding = layer.core_conv.padding,
                                  bias = None)
            set_module_params(core_layer, {'weight':w})
            mask1 = Mask_Layer(w.shape[0])
            
            last_layer = nn.Conv2d(in_channels = w3.shape[1],
                                  out_channels = w3.shape[0],
                                  kernel_size = w3.shape[2],
                                  stride = layer.last_conv.stride,
                                  padding = layer.last_conv.padding,
                                  bias = not(bias is None))
            set_module_params(last_layer, {'weight':w3, 'bias':bias})
            
            
            
            return decomposed_conv(core_conv=core_layer,last_conv=last_layer)#nn.Sequential(core_layer,mask1,last_layer)
            
            
        elif layer.ranks[1]==layer.n_channel_out:
            w = np.dot(w2.transpose(1,2,3,0),np.squeeze(w3).transpose()) #r1,k,k,r2 * r2,f ->r1,k,k,f
            w = w.transpose(3,0,1,2)
            first_layer = nn.Conv2d(in_channels = w1.shape[1],
                                  out_channels = w1.shape[0],
                                  kernel_size = w1.shape[2],
                                  stride = layer.first_conv.stride,
                                  padding = layer.first_conv.padding,
                                  bias = None)
            set_module_params(first_layer, {'weight':w1})
            mask1 = Mask_Layer(w1.shape[0])
            core_layer = nn.Conv2d(in_channels = w.shape[1],
                                  out_channels = w.shape[0],
                                  kernel_size = w.shape[2],
                                  stride = layer.core_conv.stride,
                                  padding = layer.core_conv.padding,
                                  bias = bias is not None)
            set_module_params(core_layer, {'weight':w, 'bias':bias})
            return decomposed_conv(first_conv=first_layer,core_conv=core_layer)#nn.Sequential(first_layer, mask1, core_layer)
        
        else:
            first_layer = nn.Conv2d(in_channels = w1.shape[1],
                                  out_channels = w1.shape[0],
                                  kernel_size = w1.shape[2],
                                  stride = layer.first_conv.stride,
                                  padding = layer.first_conv.padding,
                                  bias = None)
            set_module_params(first_layer, {'weight':w1})
            mask1 = Mask_Layer(w1.shape[0])
            core_layer = nn.Conv2d(in_channels = w2.shape[1],
                                  out_channels = w2.shape[0],
                                  kernel_size = w2.shape[2],
                                  stride = layer.core_conv.stride,
                                  padding = layer.core_conv.padding,
                                  bias = None)
            set_module_params(core_layer, {'weight':w2})
            mask2 = Mask_Layer(w2.shape[0])
            last_layer = nn.Conv2d(in_channels = w3.shape[1],
                                  out_channels = w3.shape[0],
                                  kernel_size = w3.shape[2],
                                  stride = layer.last_conv.stride,
                                  padding = layer.last_conv.padding,
                                  bias = not(bias is None))
            set_module_params(last_layer, {'weight':w3, 'bias':bias})
            
            return decomposed_conv(first_conv=first_layer,core_conv=core_layer,last_conv=last_layer)#nn.Sequential(first_layer, mask1, core_layer, mask2, last_layer)
        
    elif isinstance(layer, svd_conv):
        r0 = int(np.sum(np.ravel(convert_to_np(layer.mask1.mask.data))+0.01))       
        layer.rank = r0
        
        w1 = convert_to_np(layer.first_conv.weight.data)[0:r0,:,:,:]
        w2 = convert_to_np(layer.last_conv.weight.data)[:,0:r0,:,:]
        
        if layer.last_conv.bias is not None:
            bias = convert_to_np(layer.last_conv.bias.data)
        else:
            bias = None
        
        if r0==layer.max_rank:
            w = np.dot(w2.transpose(0,2,3,1),np.squeeze(w1)) #f,1,1,r1 * r1,c -> f,1,1,c
            w = w.transpose(0,3,1,2)

            out_layer = nn.Conv2d(in_channels = w.shape[1],
                                  out_channels = w.shape[0],
                                  kernel_size = 1,
                                  stride = layer.last_conv.stride,
                                  padding = layer.last_conv.padding,
                                  bias = not(bias is None))
            set_module_params(out_layer, {'weight':w,'bias':bias})
            return out_layer
                        
        else:
            first_layer = nn.Conv2d(in_channels = w1.shape[1],
                                  out_channels = w1.shape[0],
                                  kernel_size = w1.shape[2],
                                  stride = layer.first_conv.stride,
                                  padding = layer.first_conv.padding,
                                  bias = None)
            mask1 = Mask_Layer(w1.shape[0])
            set_module_params(first_layer, {'weight':w1})
            last_layer = nn.Conv2d(in_channels = w2.shape[1],
                                  out_channels = w2.shape[0],
                                  kernel_size = w2.shape[2],
                                  stride = layer.last_conv.stride,
                                  padding = layer.last_conv.padding,
                                  bias = not(bias is None))
            
            set_module_params(last_layer, {'weight':w2,'bias':bias})
            return svd_conv(first_conv=first_layer,last_conv=last_layer)#nn.Sequential(first_layer,mask1,last_layer)
            

def set_module_params(module, params):
    for key, p in params.items():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        p = params[key]
        if p is None:
            continue
        if isinstance(p, np.ndarray):
            src = torch.Tensor(p).to(device)
        else:
            src = p
               
        if key=='running_mean':    # batchnormalization layer
            module.running_mean.data.copy_(src)
        elif key=='running_var':   # batchnormalization layer
            module.running_var.data.copy_(src)
        else:
            module._parameters[key].data.copy_(src)

class apply_mask(Function):
    @staticmethod
    def forward(self, weight, mask):
        self.save_for_backward(mask)
        return torch.mul(weight, mask)#weight * mask

    @staticmethod
    def backward(self, grad_output):
        mask = self.saved_tensors[0]
        
        grad_weight = grad_mask = None

        if self.needs_input_grad[0]:
            grad_weight = torch.mul(mask,grad_output)#mask * grad_output

        return grad_weight, grad_mask


class Mask_Layer(Module):
    def __init__(self, n_out, is_conv=True):
        super(Mask_Layer,self).__init__()
        self.n_out = n_out
        if is_conv:
            mask = np.ones((1,n_out,1,1))
            self.mask_shape = (1,n_out,1,1)
            self.mask = Parameter(torch.Tensor(1,n_out,1,1), requires_grad=False)
        else:
            mask = np.ones((1,n_out))
            self.mask_shape = (1,n_out)
            self.mask = Parameter(torch.Tensor(1,n_out), requires_grad=False)
        
        src = torch.Tensor(mask)
        self.mask.data.copy_(src)
        self.pruning_rate = 0
        
    def forward(self, input):
        self.masked_inp = apply_mask.apply(input, self.mask)
        return self.masked_inp
    
    def set_mask(self, mask):
        self.pruning_rate = 1 - np.mean(mask) 
        src = torch.Tensor(mask)
        self.mask.data.copy_(src)


class Conv2d_sparse(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, mask=None, coarse_grained=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
    
        super(Conv2d_sparse, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        if coarse_grained:
            self.mask = Parameter(torch.Tensor(self.out_channels, 1, 1, 1), requires_grad=False)
            self.mask_shape = (self.out_channels, 1, 1, 1)
        else:
            self.mask = Parameter(torch.Tensor(self.out_channels, self.in_channels, 
                                           self.kernel_size[0], self.kernel_size[1]), requires_grad=False)
            self.mask_shape = (self.out_channels, self.in_channels, 
                               self.kernel_size[0], self.kernel_size[1])

        if not mask is None:
            src = torch.Tensor(mask)
            self.mask.data.copy_(src)
            print('mask is set')
        else:
            size = self.mask.size()
            mask = np.ones((size[0], size[1], size[2], size[3]))
            src = torch.Tensor(mask)
            self.mask.data.copy_(src)
        self.pruning_rate = 1 - np.mean(mask)
        
    def forward(self, input):  
        self.masked_weight = apply_mask.apply(self.weight, self.mask)
        return F.conv2d(input, self.masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def set_mask(self, mask):
        self.pruning_rate = 1 - np.mean(mask)
        src = torch.Tensor(mask)
        self.mask.data.copy_(src)
   

class Linear_sparse(Module):
    def __init__(self, in_features, out_features, bias=True, mask=None, coarse_grained=False):
        super(Linear_sparse, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        if coarse_grained:
            self.mask = Parameter(torch.Tensor(out_features, 1), requires_grad=False)
            self.mask_shape = (out_features, 1)
        else:
            self.mask = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
            self.mask_shape = (out_features, in_features)
        if not mask is None:
            src = torch.Tensor(mask)
            self.mask.data.copy_(src)
            print('mask is set')
        else:
            size = self.mask.size()
            mask = np.ones((size[0], size[1]))
            src = torch.Tensor(mask)
            self.mask.data.copy_(src)
        
        self.pruning_rate = 1 - np.mean(mask)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        self.masked_weight = apply_mask.apply(self.weight, self.mask)
        return F.linear(input, self.masked_weight, self.bias)
    
    def set_mask(self, mask):
        self.pruning_rate = 1 - np.mean(mask) 
        src = torch.Tensor(mask)
        self.mask.data.copy_(src)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class decomposed_conv(nn.Module):
    def __init__(self, original_module=None, first_conv=None, core_conv=None, last_conv=None, has_masks=True):
        super(decomposed_conv, self).__init__()
        if isinstance(original_module, nn.Conv2d):
            self.has_first = True
            self.has_core = True
            self.has_last = True
            self.ranks = [original_module.in_channels, original_module.out_channels]
            
            w = convert_to_np(original_module.weight.data)           
            b = original_module.bias
            if b is not None:
                b = convert_to_np(b)
           
            self.n_channel_in = w.shape[1]
            self.in_channels = w.shape[1]
            self.n_channel_out = w.shape[0]
            self.out_channels = w.shape[0]

            first = np.eye(w.shape[1]).reshape(w.shape[1],w.shape[1],1,1)
            core = w
            last = np.eye(w.shape[0]).reshape(w.shape[0],w.shape[0],1,1)
            
            self.first_conv=nn.Conv2d(in_channels=first.shape[1],
                                        out_channels=first.shape[0],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False)
            self.mask1 = Mask_Layer(first.shape[0])
            
            set_module_params(self.first_conv,{'weight':first})

            self.core_conv=nn.Conv2d(in_channels=core.shape[1],
                                        out_channels=core.shape[0],
                                        kernel_size=core.shape[2],
                                        stride=original_module.stride,
                                        padding=original_module.padding,
                                        bias=False)
            self.mask2 = Mask_Layer(core.shape[0])

            set_module_params(self.core_conv,{'weight':core})

            self.last_conv=nn.Conv2d(in_channels=last.shape[1],
                                        out_channels=last.shape[0],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=not(b is None))
            set_module_params(self.last_conv,{'weight':last,'bias':b})
        else:            
            assert core_conv is not None
            self.ranks = [core_conv.in_channels, core_conv.out_channels]

            self.has_first = not(first_conv is None)
            self.has_last = not(last_conv is None)
            
            self.first_conv = first_conv
            if self.has_first:
                self.n_channel_in = first_conv.in_channels
                self.in_channels = first_conv.in_channels
                if has_masks:
                    self.mask1 = Mask_Layer(self.first_conv.out_channels)
                else:
                    self.mask1 = None
            else:
                self.n_channel_in = core_conv.in_channels
                self.in_channels = core_conv.in_channels
                self.mask1 = None
            
            self.core_conv = core_conv
            
            if self.has_last:
                if has_masks:
                    self.mask2 = Mask_Layer(self.core_conv.out_channels)
                else:
                    self.mask2 = None
                
                self.n_channel_out = last_conv.out_channels
                self.out_channels = last_conv.out_channels
            
            else:
                self.mask2 = None
                self.n_channel_out = core_conv.out_channels
                self.out_channels = core_conv.out_channels
                
            self.last_conv = last_conv

    def forward(self, x):
        out = x
        if self.has_first:
            out = self.first_conv(out)
        if not self.mask1 is None:
            out = self.mask1(out)
        out = self.core_conv(out)
        if not(self.mask2 is None):
            out = self.mask2(out)
        if self.has_last:
            out = self.last_conv(out)
        return out
    
    def set_config(self, first, core, last):
        r0 = first.shape[0]
        r0core = core.shape[1]
        r1 = last.shape[1]
        r1core = core.shape[0]
        
        assert r0==r0core
        assert r1==r1core
        
        self.ranks = [r0,r1]
        
        first_pad = np.zeros(self.first_conv.weight.size())
        first_pad[0:r0,:,:,:] = convert_to_np(first)
        core_pad = np.zeros(self.core_conv.weight.size())
        core_pad[0:r1,0:r0,:,:] = convert_to_np(core)
        last_pad = np.zeros(self.last_conv.weight.size())
        last_pad[:,0:r1,:,:] = convert_to_np(last)


        set_module_params(self.first_conv,{'weight':first_pad})

        set_module_params(self.core_conv,{'weight':core_pad})

        set_module_params(self.last_conv,{'weight':last_pad})
        
        maskfirst = np.ones(self.mask1.mask.size())
        maskfirst[:,r0:,:,:] = 0
        self.mask1.set_mask(maskfirst)
        
        maskcore = np.ones(self.mask2.mask.size())
        maskcore[:,r1:,:,:] = 0
        self.mask2.set_mask(maskcore)


class svd_conv(nn.Module):
    def __init__(self, original_module=None, first_conv=None, last_conv=None, has_masks=True):
        super(svd_conv, self).__init__()
        if isinstance(original_module, nn.Conv2d):
            w = convert_to_np(original_module.weight.data)
            assert w.shape[2]==1 and w.shape[3]==1 #this layer only works for 1by1 (pointwise) convolutions
           
            b = original_module.bias
            if b is not None:
                b = convert_to_np(b)
           
            self.in_channels=w.shape[1]
            self.out_channels=w.shape[0]

            self.max_rank = min(w.shape[0],w.shape[1])
            self.rank = copy.deepcopy(self.max_rank)
            
            if w.shape[1]<w.shape[0]:
                self.first_conv = nn.Conv2d(in_channels=w.shape[1],
                                                out_channels=w.shape[1],
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False)
                
                self.mask1 = Mask_Layer(w.shape[1])
                self.last_conv = nn.Conv2d(in_channels=w.shape[1],
                                            out_channels=w.shape[0],
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=(b is not None))
                
                first = np.eye(w.shape[1]).reshape(w.shape[1],w.shape[1],1,1)
                last = w
            else:
                self.first_conv = nn.Conv2d(in_channels=w.shape[1],
                                            out_channels=w.shape[0],
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False)
                
                self.mask1 = Mask_Layer(w.shape[0])
                
                self.last_conv = nn.Conv2d(in_channels=w.shape[0],
                                            out_channels=w.shape[0],
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=(b is not None))
                
                first = w
                last = np.eye(w.shape[0]).reshape(w.shape[0],w.shape[0],1,1)
            
            set_module_params(self.first_conv,{'weight':first})

            set_module_params(self.last_conv,{'weight':last,'bias':b})
        else:
            self.rank = first_conv.out_channels
            self.first_conv = first_conv
            if has_masks:
                self.mask1 = Mask_Layer(self.first_conv.out_channels)
            else:
                self.mask1 = None
            self.last_conv = last_conv
            
            self.in_channels = first_conv.in_channels
            self.out_channels = last_conv.out_channels
            
            
    def forward(self, x):
        out = self.first_conv(x)
        if not(self.mask1) is None:
            out = self.mask1(out)
        out = self.last_conv(out)
        return out

    def set_config(self, rank):
        assert rank<=self.max_rank
        self.rank = rank
        # assuming that the SVD decomposition is stored in self.first_conv and self.last_conv, 
        # we can simply mask principal vectors. This is equivalent to the following line:
        maskfirst = np.ones(self.mask1.mask.size())
        maskfirst[:,rank:,:,:] = 0
        self.mask1.set_mask(maskfirst)

