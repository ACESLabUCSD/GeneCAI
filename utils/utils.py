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
from tqdm import tqdm

from tensorly.decomposition import parafac, partial_tucker
import tensorly as tl

from models.layers import Conv2d_sparse, Linear_sparse, Mask_Layer, decomposed_conv, svd_conv, wrap_layer
from models.resnet import ResNetMask, Cifar10ResNetMask, BasicBlockMask, BottleneckMask, BasicBlock, Bottleneck
from train_eval_utils import test

class time_meter(object):
	def __init__(self):
		self.n = 0
		self.avg = 0
	def add(self,t):
		sm = self.n*self.avg + t
		self.n = self.n + 1
		self.avg = sm/self.n


def get_data(val_ratio=0.02):
	print('==> Preparing data..')
	transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	val_data_idx = int((1-val_ratio) * len(trainset))
	trainset.data = trainset.data[0:val_data_idx]
	trainset.targets = trainset.targets[0:val_data_idx]
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
	valset.data = valset.data[val_data_idx:]
	valset.targets = valset.targets[val_data_idx:]
	valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	return trainloader, valloader, testloader

class net_description(nn.Module):
	def __init__(self,modules):
		super(net_description, self).__init__()
		if 'features' in list(modules.keys()):
			features = modules['features']
			self.features = nn.Sequential(*features)
		else:
			self.features = None
		classifier = modules['classifier']
		
		self.classifier = nn.Sequential(*classifier)

	def forward(self, x):
		if self.features:
			x = self.features(x)
		x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
		x = self.classifier(x)
		return x

def convert_to_masked_model(model, decomposed=False, dataset='CIFAR10', arch='VGG'):
	while model._modules.keys()[0]=='module':
		model = model._modules['module']
	
	prev_shape = None
	isconv = True
	top_modules = collections.OrderedDict({})
	for key, module in model._modules.items():
		if type(module)==torch.nn.DataParallel:
			module = module.module
		top_modules[key], prev_shape, isconv = module_replicator(module, key=key,
																 prev_shape=prev_shape,
																 isconv=isconv, 
																 decomposed=decomposed, 
																 add_mask=True)
		top_modules[key] = top_modules[key][0]
	
	if dataset=='ImageNet':
		if 'ResNet' in arch:
			masked_model = ResNetMask(top_modules, decomposed)
		else :
			masked_model = net_description(top_modules)
	else:
		if 'ResNet' in arch:
			masked_model = Cifar10ResNetMask(top_modules, decomposed)
		else:
			masked_model = net_description(top_modules)
	return masked_model

def module_replicator(module, key='',  prev_shape=None, isconv=True, decomposed=False, wrap=False, add_mask=False):
	new_modules_list = []

	if isinstance(module, nn.Sequential):
		modules = []  
		for subkey,submodule in module._modules.items():
			new_modules, prev_shape, isconv = module_replicator(submodule,
																key+'-'+subkey, 
																prev_shape=prev_shape, 
																isconv=isconv,
																decomposed=decomposed,
																wrap=wrap, 
																add_mask=add_mask)
			modules = modules + new_modules
		if isinstance(module, nn.Sequential):
			new_module = nn.Sequential(*modules)
		else:
			new_module = nn.Container(*modules)
		new_modules_list.append(new_module)
	
	elif isinstance(module, BasicBlock) or isinstance(module, BasicBlockMask):		
		new_module = BasicBlockMask(module, decomposed=decomposed, wrap=wrap)
		new_modules_list.append(new_module)
		if hasattr(module.conv2,'out_channels'):
			prev_shape = module.conv2.out_channels
		else:
			prev_shape = module.conv2.last_conv.out_channels
		
	elif isinstance(module, Bottleneck) or isinstance(module, BottleneckMask):		
		new_module = BottleneckMask(module, decomposed=decomposed,wrap=wrap)
		new_modules_list.append(new_module)
		if hasattr(module.conv3,'out_channels'):
			prev_shape = module.conv3.out_channels
		else:
			prev_shape = module.conv3.last_conv.out_channels
		
	elif isinstance(module, nn.Conv2d):
		if not decomposed:
			new_module = module
			
		else:
			kernel_size = module.weight.data.size()[2]
			if kernel_size>1:
				new_module = decomposed_conv(module)
			else:
				new_module = module
			
		prev_shape = module.out_channels
		new_modules_list.append(new_module)
				
	elif isinstance(module, nn.Linear):
		new_module = module
		new_modules_list.append(new_module)
		prev_shape = module.out_features
		isconv = False
		
	elif isinstance(module, nn.ReLU):
		new_module = nn.ReLU()
		new_modules_list.append(new_module)
		if add_mask:
			new_modules_list.append(Mask_Layer(prev_shape,isconv))
	
	elif isinstance(module, Mask_Layer):
		new_module = module
		new_modules_list.append(new_module)
	
	elif isinstance(module, decomposed_conv):
		if not wrap:
			new_module = module
			new_modules_list.append(new_module)
		else:
			new_module = wrap_layer(module)
			new_modules_list.append(new_module)
	
	elif isinstance(module, svd_conv):
		if not wrap:
			new_module = module
			new_modules_list.append(new_module)
		else:
			new_module = wrap_layer(module)
			new_modules_list.append(new_module)
	   
	else:
		new_module = module
		new_modules_list.append(new_module)
		
	return new_modules_list, prev_shape, isconv

def svd_weight(w):
	U, s, V = np.linalg.svd(w)
	s = np.diag(s)
	if w.shape[1]<w.shape[0]:
		appendix = np.zeros((w.shape[0]-s.shape[0],s.shape[0]))
		s = np.concatenate((s,appendix),axis=0)
		last = np.dot(U,s)
		first = V
	else:
		appendix = np.zeros((s.shape[0],w.shape[1]-s.shape[0]))
		s = np.concatenate((s,appendix),axis=1)
		last = U
		first = np.dot(s,V)

	return first, last

def decompose_cnn_weight_using_tucker(w,ranks):
	if ranks[0]==w.shape[1] and ranks[1]==w.shape[0]:
		first = np.eye(w.shape[1]).reshape(w.shape[1],w.shape[1],1,1)
		core = w
		last = np.eye(w.shape[0]).reshape(w.shape[0],w.shape[0],1,1)
		return first, core, last
	
	w = convert_to_np(w)
	w = w.transpose(2,3,1,0)

	w = tl.tensor(w)
	t1 = time.time()
	
	core, [first, last] = partial_tucker(w,modes=[2,3],ranks=ranks,init='svd',n_iter_max=100)

	core_sh = [int(s) for s in core.shape]
	first_sh = [1,1]+[int(s) for s in first.shape]
	last_sh = [1,1]+[int(s) for s in last.shape]
	core = np.reshape(tl.to_numpy(core),core_sh)
	core = core.astype(np.float32)
	first = np.reshape(tl.to_numpy(first),first_sh)
	first = first.astype(np.float32)
	last = np.reshape(tl.to_numpy(last),last_sh)
	last = last.astype(np.float32)
	last = last.transpose(0,1,3,2)

	core = core.transpose(3,2,0,1)
	first = first.transpose(3,2,0,1)
	last = last.transpose(3,2,0,1)

	return first, core, last

def sweep_decompose_weight(w, num_quant=8):
	r0list = [int(w.shape[1]*1.0*i/num_quant+0.001) for i in range(1,num_quant+1)]
	r1list = [int(w.shape[0]*1.0*i/num_quant+0.001) for i in range(1,num_quant+1)]
	for i in range(num_quant):
		if r0list[i]==0:
			r0list[i] = 1
		if r1list[i]==0:
			r1list[i] = 1
	dict_of_decomposed = collections.OrderedDict({})
	with tqdm(total=num_quant*num_quant) as pbar:
		for i, r0 in enumerate(r0list):
			for j, r1 in enumerate(r1list):
				dict_of_decomposed[(i,j)] = decompose_cnn_weight_using_tucker(w,(r0,r1))
				pbar.update(1)

	return dict_of_decomposed

def get_decomposed_parameters(args, decomposed_layers, num_quant):
	path_to_save = os.path.join('artifacts', args.dataset, args.arch, 'decomposed_weights')
	if not os.path.exists(path_to_save):
		os.makedirs(path_to_save)
	
	all_decomposed_params = []
	for idx, layer in enumerate(decomposed_layers):		 
		file = os.path.join(path_to_save, str(idx)+'_decomposed.pkl')   
		if os.path.exists(file):
			# print('loading decomposed parameters for layer %d out of %d' % (idx, len(decomposed_layers)))
			with open(file, 'rb') as f:
				decomposed_params = pickle.load(f)
				if isinstance(decomposed_params, torch.Tensor):
					decomposed_params = convert_to_np(decomposed_params)
		else:
			print('decomosing layer %d out of %d' % (idx, len(decomposed_layers)))
			if isinstance(layer, decomposed_conv):
				w = layer.core_conv.weight.data
				decomposed_params = sweep_decompose_weight(w, num_quant=num_quant)
			else:
				w1 = np.squeeze(np.asarray(layer.first_conv.weight.data))
				w2 = np.squeeze(np.asarray(layer.last_conv.weight.data))
				
				if w1.shape[0]==w1.shape[1] and w2.shape[0]==w2.shape[1]:
					err1 = np.sum(np.absolute(w1-np.eye(w1.shape[0])))
					err2 = np.sum(np.absolute(w2-np.eye(w2.shape[0])))
					if err1==0:
						decomposed_params = svd_weight(w2)
					elif err2==0:
						decomposed_params = svd_weight(w1)
					else:
						raise('at least one of the weights should be eye')
				elif w1.shape[0]==w1.shape[1]:
					err1 = np.sum(np.absolute(w1-np.eye(w1.shape[0])))
					assert err1==0
					decomposed_params = svd_weight(w2)
				elif w2.shape[0]==w2.shape[1]:
					err2 = np.sum(np.absolute(w2-np.eye(w2.shape[0])))
					assert err2==0
					decomposed_params = svd_weight(w1)
				else:
					raise('at least one of the weights should be sqare shaped')
					
				first = decomposed_params[0]
				first = first.reshape(first.shape[0],first.shape[1],1,1)
				last = decomposed_params[1]
				last = last.reshape(last.shape[0],last.shape[1],1,1)
				set_module_params(layer.first_conv,{'weight':first})
				set_module_params(layer.last_conv,{'weight':last})
				
			with open(file, 'wb') as f:
				pickle.dump(decomposed_params, f)

		all_decomposed_params.append(decomposed_params)
	
	return all_decomposed_params

def get_valid_ranks(args, decomposed_layers, decomposed_params, num_quant, valloader, masked_net, acc_threshold):
	path_to_pkl = os.path.join('artifacts', args.dataset, args.arch, 'boundries_D.pkl')
	
	if os.path.exists(path_to_pkl):
		print('loading valid decomposition ranks from %s'%path_to_pkl)
		with open(path_to_pkl, 'rb') as f:
			valid_r0_per_layer, valid_r1_per_layer = pickle.load(f)
	else:
		num_quant_svd = int(num_quant**2)
		genes_decomp_r0 = len(decomposed_layers)
		genes_decomp_r1 = len(decomposed_layers)


		valid_r0_per_layer = []
		valid_r1_per_layer = []	
		for idx, layer in enumerate(decomposed_layers):
			valid_r0 = []
			valid_r1 = []
			
			if isinstance(layer, decomposed_conv):
				first, core, last = decomposed_params[idx][(num_quant-1, num_quant-1)]
				layer.set_config(first, core, last)
				orig_flops = get_flops(layer)
				for r0 in range(0, num_quant-1):
					first, core, last = decomposed_params[idx][(r0, num_quant-1)]
					layer.set_config(first, core, last)
					flops = get_flops(layer)
					acc = test(valloader, masked_net)
					if acc>args.acc_threshold:
						for j in range(r0, num_quant):
							first, core, last = decomposed_params[idx][(r0, num_quant-1)]
							layer.set_config(first, core, last)
							flops = get_flops(layer)
							valid_r0.append(j)
						break
					
				first, core, last = decomposed_params[idx][(num_quant-1, num_quant-1)]
				layer.set_config(first, core, last)


				for r1 in range(0, num_quant-1):
					first, core, last = decomposed_params[idx][(num_quant-1, r1)]
					layer.set_config(first, core, last)
					flops = get_flops(layer)
					acc = test(valloader, masked_net)
					if acc>args.acc_threshold:
						for j in range(r1, num_quant):
							first, core, last = decomposed_params[idx][(num_quant-1, r1)]
							layer.set_config(first, core, last)
							flops = get_flops(layer)
							valid_r1.append(j)
						break

				if not(num_quant-1 in valid_r0):
					valid_r0.append(num_quant-1)
				if not(num_quant-1 in valid_r1):
					valid_r1.append(num_quant -1)

				first, core, last = decomposed_params[idx][(num_quant-1, num_quant-1)]
				layer.set_config(first, core, last)
			
			elif isinstance(layer, svd_conv):
				layer.set_config(layer.max_rank)
				orig_flops = get_flops(layer)
				for r0 in range(0, num_quant_svd):
					rank = int(layer.max_rank * (r0+1)*1.0/(num_quant_svd)+0.01)
					
					layer.set_config(rank)
					flops = get_flops(layer)
					acc = test(valloader, masked_net)
					if acc>args.acc_threshold:
						for j in range(r0, num_quant_svd):
							rank = int(layer.max_rank * (j+1)*1.0/(num_quant_svd)+0.01)
							layer.set_config(rank)
							flops = get_flops(layer)
							if flops<orig_flops:
								valid_r0.append(j)
								valid_r1.append(j)
						break
								
				if not(num_quant_svd-1 in valid_r0):
					valid_r0.append(num_quant_svd-1)
				if not(num_quant_svd-1 in valid_r1):
					valid_r1.append(num_quant_svd-1)
					
					
				layer.set_config(layer.max_rank)

			print('layer {0:d} valid_r0: {1}, valid_r1: {2}' .format(idx, valid_r0, valid_r1))
			valid_r0_per_layer.append(valid_r0)
			valid_r1_per_layer.append(valid_r1)
				
			assert len(valid_r0)==len(np.unique(valid_r0))
			assert len(valid_r1)==len(np.unique(valid_r1))
						
		masked_net.eval()
		_ = test(valloader, masked_net, verbose=True)

		with open(path_to_pkl, 'wb') as f:
			pickle.dump([valid_r0_per_layer,valid_r1_per_layer], f)
		print('saved max pruning rates to %s' % path_to_pkl)

		decompose_all_layers(decomposed_layers, decomposed_params, np.zeros(len(decomposed_layers)), np.zeros(len(decomposed_layers)), num_quant_svd)

	return valid_r0_per_layer, valid_r1_per_layer

def decompose_all_layers(decomposed_layers, decomposed_params, individual_r0, individual_r1, num_quant_svd=64):
	for idx, layer in enumerate(decomposed_layers):
		if isinstance(layer, decomposed_conv):
			r0 = individual_r0[idx]
			r1 = individual_r1[idx]
			first, core, last = decomposed_params[idx][(r0,r1)]
			layer.set_config(first, core, last)
		
		elif isinstance(layer, svd_conv):
			r0 = individual_r0[idx]
			rank = int(layer.max_rank*(1.0+r0)/num_quant_svd+0.01)
			layer.set_config(rank)

def set_module_params(module, params):
	for key, p in params.items():
		p = params[key]
		if p is None:
			continue
		if isinstance(p, np.ndarray):
			if torch.cuda.is_available():
				src = torch.Tensor(p).cuda()
			else:
				src = torch.Tensor(p)
		else:
			src = p
			   
		if key=='running_mean':	# batchnormalization layer
			module.running_mean.data.copy_(src)
		elif key=='running_var':   # batchnormalization layer
			module.running_var.data.copy_(src)
		else:
			module._parameters[key].data.copy_(src)  

def wrap_decomposed_model(model, dataset='CIFAR10', arch='VGG16'):
	while model._modules.keys()[0]=='module':
		model = model._modules['module']

	prev_shape = None
	isconv = True
	top_modules = collections.OrderedDict({})
	for key, module in model._modules.items():
		if type(module)==torch.nn.DataParallel:
			module = module.module
		top_modules[key], prev_shape, isconv = module_replicator(module, key=key,
															  prev_shape=prev_shape,
															  isconv=isconv,
															  wrap=True)
		top_modules[key] = top_modules[key][0]
		 
	if dataset=='ImageNet':
		if 'ResNet' in arch:
			masked_model = ResNetMask(top_modules, decomposed=False)
		else :
			masked_model = net_description(top_modules)
	else:
		if 'ResNet' in arch:
			masked_model = Cifar10ResNetMask(top_modules, decomposed=False)
		else:
			masked_model = net_description(top_modules)
	return masked_model

def prune_layer(layer, sorted_args, rate):
	mask = np.ones(layer.mask_shape)
	orig_size = mask.shape
	mask = np.ravel(mask)
	
	assert len(sorted_args)==len(mask)
	
	to_be_pruned = int(rate * len(mask))
	mask[sorted_args[0:to_be_pruned]] = 0
	mask = mask.reshape(list(orig_size))
	
	layer.set_mask(mask)
	
def prune_all_masked_layers(layer_rates, layers_to_prune, all_sorted_args):
	assert len(layers_to_prune)==len(layer_rates)
	assert len(layer_rates)==len(all_sorted_args)
	
	index = 0
	for layer in layers_to_prune:
		prune_layer(layer, all_sorted_args[index], layer_rates[index])
		index = index + 1

def get_pruning_priorities(layer):
	w = convert_to_np(layer.weight.data)
	if len(w.shape)==2:
		#-------------- FC layer
		sums = np.sum(np.absolute(w),axis=1)
		inds = np.argsort(sums)
	elif len(w.shape)==4:
		#-------------- CONV layer
		sums = np.sum(np.absolute(w),axis=(1,2,3))
		inds = np.argsort(sums)
	return inds

def get_all_pruning_priorities(net, valloader, device, gradbased=False, layers_to_prune=None):
	if not gradbased:
		layers_to_prune = get_list_of_layers(net, layerType=[nn.Conv2d])
		
		corresponding_masks = get_list_of_layers(net, layerType=[Mask_Layer])

		assert len(layers_to_prune)==len(corresponding_masks)

		all_priorities = []
		index = 0
		for layer in layers_to_prune:
			all_priorities.append(get_pruning_priorities(layer))
			index = index + 1
	else:
		if layers_to_prune is None:
			grads = get_layer_grads(net, valloader, device, layerType=Mask_Layer)
		else:
			 grads = get_layer_grads(net, valloader, device, layers_to_add_hook=layers_to_prune)
			
		all_priorities = []
		index = 0
		for g in grads:
			all_priorities.append(np.argsort(g))
			index = index + 1.

			
	return all_priorities

def get_list_of_layers(module, layerType=None):
	if not isinstance(layerType, list):
		layerType = [layerType]
	
	layers = []
	children = list(module.children())
	
	for lt in layerType:
		if isinstance(module, lt):
			return [module]

	if len(children)>0:
		for c in children:
			layers = layers + get_list_of_layers(c, layerType)

	return layers

def get_max_prune_rates(args, layers_to_prune, all_sorted_args, valloader, masked_net, device, acc_threshold):
	path_to_pkl = os.path.join('artifacts', args.dataset, args.arch, 'boundries_CP.pkl')
	
	if os.path.exists(path_to_pkl):
		print('loading max pruning rates from %s'%path_to_pkl)
		with open(path_to_pkl, 'rb') as f:
			max_prune_rates = pickle.load(f)
	else:
		genes_per_individual = len(layers_to_prune)
		max_prune_rates = np.zeros(genes_per_individual)
		index = 0
		for layer in layers_to_prune:
			prune_layer(layer, all_sorted_args[index], 0)
			index = index + 1
		index = 0
		for i, layer in enumerate(layers_to_prune):
			for rate in range(100,-1,-5):
				r = rate/100.0
				prune_layer(layer, all_sorted_args[index], r)
				acc = test(valloader, masked_net, verbose=False)
				if acc >= acc_threshold:
					break
			print(i, r)
			max_prune_rates[index] = r
			prune_layer(layer, all_sorted_args[index], 0)
			index = index + 1

		with open(path_to_pkl, 'wb') as f:
			pickle.dump(max_prune_rates, f)
		print('saved max pruning rates to %s'%path_to_pkl)
	
	return max_prune_rates

def get_layers_with_flops(net):
	layers_with_flops = get_list_of_layers(net, layerType=[nn.Conv2d, nn.Linear, decomposed_conv, svd_conv])
	mask_layers = get_list_of_layers(net, layerType=[Mask_Layer])
	
	flop_layers = [l for l in layers_with_flops]
	m_layers_before = [None]+[l for l in mask_layers]
	m_layers_after = [l for l in mask_layers]+[None] 
	
	return flop_layers, m_layers_before, m_layers_after

def complexity(layer, prune_rate_prev=0, prune_rate_next=0):   
	if isinstance(layer, decomposed_conv):
		ranks = layer.ranks
		assert ranks[0]<=layer.n_channel_in
		assert ranks[1]<=layer.n_channel_out
		if ranks[0]<layer.n_channel_in:
			first_shape=list(layer.first_conv.output_size)
			first_shape[0]=layer.ranks[0]
			first_in_channels=layer.first_conv.in_channels
			num_dot_prods=np.prod(first_shape)
			per_dot_prod=np.prod(layer.first_conv.kernel_size)*first_in_channels
			complexity_first=num_dot_prods*per_dot_prod*(1-prune_rate_prev)
		else:
			complexity_first=0


		core_shape=list(layer.core_conv.output_size)
		core_shape[0]=layer.ranks[1]
		core_in_channels=layer.ranks[0]
		num_dot_prods=np.prod(core_shape)
		per_dot_prod=np.prod(layer.core_conv.kernel_size)*core_in_channels
		complexity_core=num_dot_prods*per_dot_prod
		if complexity_first==0:
			complexity_core = complexity_core*(1-prune_rate_prev)
		

		if ranks[1]<layer.n_channel_out:
			last_shape=list(layer.last_conv.output_size)
			last_in_channels=layer.ranks[1]
			num_dot_prods=np.prod(last_shape)
			per_dot_prod=np.prod(layer.last_conv.kernel_size)*last_in_channels
			complexity_last=num_dot_prods*per_dot_prod*(1-prune_rate_next)
		   
		else:
			complexity_last=0

		if complexity_last==0:
			complexity_core = complexity_core*(1-prune_rate_next)
		complexity_all = complexity_last+complexity_core+complexity_first

		return complexity_all
	
	elif isinstance(layer, svd_conv):
		first_shape=list(layer.first_conv.output_size)
		first_shape[0]=layer.rank
		first_in_channels=layer.first_conv.in_channels
		num_dot_prods=np.prod(first_shape)
		per_dot_prod=np.prod(layer.first_conv.kernel_size)*first_in_channels
		complexity_first=num_dot_prods*per_dot_prod

		last_shape=list(layer.last_conv.output_size)
		last_in_channels=layer.rank
		num_dot_prods=np.prod(last_shape)
		per_dot_prod=np.prod(layer.last_conv.kernel_size)*last_in_channels
		complexity_last=num_dot_prods*per_dot_prod
		
		if layer.rank==layer.max_rank:
			return max(complexity_first,complexity_last)*(1-prune_rate_prev)*(1-prune_rate_next)
		else:
			return complexity_first*(1-prune_rate_prev)+complexity_last*(1-prune_rate_next)
		

def get_flops(l, prune_rate_prev=0, prune_rate_next=0):
	if isinstance(l, nn.Conv2d):
		per_dot_product = np.prod(l.weight.data.size()[1:])
		assert len(l.output_size)==3
		num_dot_products = np.prod(l.output_size)
		flops = per_dot_product * num_dot_products *(1-prune_rate_prev)*(1-prune_rate_next)

	elif isinstance(l, nn.Linear):
		flops = np.prod(l.weight.data.size())*(1-prune_rate_prev)*(1-prune_rate_next)

	elif isinstance(l, decomposed_conv) or isinstance(l, svd_conv):
		flops = complexity(l, prune_rate_prev, prune_rate_next)
	else:
		assert False, "flops calculation not possible for %s layer type" % type(l)
	
	return flops

def get_all_flops(net, return_mask=True):
	layers = get_list_of_layers(net, layerType=[nn.Conv2d, nn.Linear, Mask_Layer, decomposed_conv])

	all_flops = []
	masks_before = []
	masks_after = []
	index = 0
	for index, l in enumerate(layers):
		if index==0:
			prev_layer = None
		else:
			prev_layer = layers[index-1]
		
		if index==len(layers)-1:
			next_layer = None
		else:
			next_layer = layers[index+1]
			
		if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear) or isinstance(l, decomposed_conv):
			all_flops.append(get_flops(l))
			
			if isinstance(prev_layer, Mask_Layer):
				masks_before.append(prev_layer)
			else:
				masks_before.append(None)
				
			if isinstance(next_layer, Mask_Layer):
				masks_after.append(next_layer)
			else:
				if isinstance(l, decomposed_conv) and (isinstance(next_layer, nn.Conv2d) or isinstance(next_layer, svd_conv)):
					assert isinstance(layers[index+2], Mask_Layer)
					masks_after.append(layers[index+2])
				else:
					masks_after.append(None)

	if return_mask:
		return np.asarray(all_flops).astype(np.float32), masks_before, masks_after
	else:
		return np.asarray(all_flops).astype(np.float32)

def compute_flops_pruned_net(per_layer_flops, masks_before, masks_after):
	assert len(masks_before)==len(per_layer_flops)
	assert len(masks_after)==len(per_layer_flops)
	overall_flop = 0
	
	for i, f in enumerate(per_layer_flops):	   
		if not masks_before[i] is None:
			p_prev = masks_before[i].pruning_rate
		else:
			p_prev = 0
		
		if not masks_after[i] is None:
			p_this = masks_after[i].pruning_rate
		else:
			p_this = 0
			
		overall_flop = overall_flop + f*(1 - p_this)*(1 - p_prev)
		
	return overall_flop

def get_in_out_shapes(self, inp, out):
	if not hasattr(self,'input_size'):
		self.input_size = inp[0].size()
	if not hasattr(self,'output_size'):
		self.output_size = out[0].data.size()
	if isinstance(self, decomposed_conv):
		print('here1')
		core_computed = False
		if hasattr(layer, 'first_conv'):
			if not(layer.first_conv is None):
				out_first = layer.first_conv(inp)
				layer.first_conv.output_size = copy.deepcopy(np.ravel(out_first[0].size()))
		
				out_core = layer.core_conv(out_first)
				layer.core_conv.output_size = copy.deepcopy(np.ravel(out_core[0].size()))
				
				core_computed = True
		
		if not core_computed:
			out_core = layer.core_conv(inp)
			layer.core_conv.output_size = copy.deepcopy(np.ravel(out_core[0].size()))
		
		if hasattr(layer, 'last_conv'):
			if not (layer.last_conv is None):
				out_last = layer.last_conv(out_core)
				layer.last_conv.output_size = copy.deepcopy(np.ravel(out_last[0].size()))
				assert np.sum(layer.last_conv.output_size - layer.output_size)==0

	elif isinstance(self, svd_conv):
		out_first = layer.first_conv(inp)
		out_last = layer.last_conv(out_first)
		layer.first_conv.output_size = copy.deepcopy(np.ravel(out_first[0].size()))
		layer.last_conv.output_size = copy.deepcopy(np.ravel(out_last[0].size()))
		assert np.sum(layer.last_conv.output_size - layer.output_size)==0

def save_inp_out_size(model, inp_size):
	assert len(inp_size)==3
	layer_list = get_list_of_layers(model, layerType=[nn.Conv2d, nn.Linear, Conv2d_sparse, Linear_sparse])
	inp_size = list(inp_size)
	handles = []
	for submodule in layer_list:
		handles.append(submodule.register_forward_hook(get_in_out_shapes))
	
	model.eval()

	# run the network so that the forward hooks (defined above) gather the ReLU outputs:
	inp = np.random.rand(1, inp_size[0], inp_size[1], inp_size[2])
	inp = torch.Tensor(inp)
	if torch.cuda.device_count()>0:
		inp = inp.cuda()
	outputs = model(inp)

	#remove all forward hooks so that it is no longer executed:
	for h in handles:
		h.remove()

	model.train()

def grad_store(self, grad_input, grad_output):
	grad = grad_output[0]
	
	if not hasattr(self, 'sum_of_grads'):
		self.sum_of_grads = torch.zeros(grad.size()[1]).cuda()
		
	grad_abs = torch.abs(grad)
	
	if len(grad_abs.size())==4:
		sum_for_this_batch = torch.sum(grad_abs, dim=(0,2,3)).cuda()
	elif len(grad_abs.size())==2:
		sum_for_this_batch = torch.sum(grad_abs, dim=(0)).cuda()

	self.sum_of_grads += sum_for_this_batch

def convert_to_np(p):
	if not type(p)==np.ndarray:
		return p.detach().cpu().numpy()
	else:
		return p

def get_layer_grads(net,  loader, device,layerType=None, layers_to_add_hook=None):
	if layers_to_add_hook is None:
		layers_to_add_hook = get_list_of_layers(net, layerType=layerType)
		
	handles_to_backward_hooks = []
	for l in layers_to_add_hook:
		handles_to_backward_hooks.append(l.register_backward_hook(grad_store))
		
	criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=0, momentum=0, weight_decay=0)

	for batch_idx, (inputs, targets) in enumerate(loader):
		inputs, targets = inputs.to(device), targets.to(device)
		# optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
	
	for h in handles_to_backward_hooks:
		h.remove()
		
	grad_sums = []
	for l in layers_to_add_hook:
		grad_sums.append(convert_to_np(l.sum_of_grads))
	
	return grad_sums


def heatmap(data, fname, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	if not ax:
		# ax = plt.gca()
		fig , ax = plt.subplots()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)
	

	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False,
				   labeltop=True, labelbottom=False)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
			 rotation_mode="anchor")

	# Turn spines off and create white grid.
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
	ax.tick_params(which="minor", bottom=False, left=False)


	plt.xlabel('Individuals')
	plt.ylabel('Layers')

	plt.rcParams["figure.figsize"] = (20*data.shape[1]/50,40*data.shape[1]/50)
	

	divider = make_axes_locatable(ax)
	cax2 = divider.append_axes("right", size="5%", pad=0.05)
	cbar = ax.figure.colorbar(im, cax=cax2, **cbar_kw)
	fig.savefig(fname, bbox_inches='tight')
	return im, cbar  

