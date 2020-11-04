from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from models import *
from utils.utils import *
from utils.train_eval_utils import *
from utils.genetic_utils import init_population, run_genetic_pruning, run_genetic_decomposition

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

parser.add_argument('--phase', default='D',
					help='Execution phases: D (decomposition) => FT_D (finetuning after D) => CP (channel pruning) => CP_FT (fine-tuning after CP)')
parser.add_argument('--arch', default='VGG16',
					help='model architecture: [VGG16, ResNet56, ResNet110] (default: VGG16)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
					help='initial learning rate (default: 0.1)')
parser.add_argument('--val_ratio', default=0.02, type=float,
					help='portion of training data used for optimization validation set (default: 0.02)')
parser.add_argument('--num_population', default=50, type=int,
					help='genetic population size (default: 50)')
parser.add_argument('--acc_threshold', default=90, type=float,
					help='accuracy constraint for optimization (default: 90)')
parser.add_argument('--iter', default=50, type=int,
					help='number of genetic iterations (default: 50)')
parser.add_argument('--p_cross', default=0.2, type=float,
					help='probability of crossover (default: 0.2)')
parser.add_argument('--p_swap', default=0.2, type=float,
					help='per-bit exchange probability (default: 0.2)')
parser.add_argument('--p_mutate', default=0.8, type=float,
					help='probability of mutate (default: 0.8)')
parser.add_argument('--p_tweak', default=0.05, type=float,
					help='per-bit tweaking probability (default: 0.05)')
parser.add_argument('--mutate_s', default=0.2, type=float,
					help='mutation scale (default: 0.2)')
parser.add_argument('--coeff', default=100., type=float,
					help='flops coefficient for the score function (default: 100.)')
parser.add_argument('--config', default='', type=str,
					help='name of the desired per-layer configuration for FT phases (default: '')')
parser.add_argument('--num_bins', default=8, type=int,
					help='number of bins for decomposition ranks (default: 8)')
parser.add_argument('--compressed_ckpt', default='', type=str,
					help='path to previously compressed checkpoint (default: '')')
args = parser.parse_args()
args.dataset = 'CIFAR10'


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load data
	trainloader, valloader, testloader = get_data(val_ratio=args.val_ratio)
	if args.dataset=='CIFAR10':
			inp_size = [3,32,32]
	else:
		inp_size = [3,224,224]

	#----------------------- load model
	path_to_checkpoint = os.path.join('artifacts', args.dataset, args.arch, 'checkpoint')

	net = get_model(args, device)	
	print('==> Resuming from checkpoint..')
	assert os.path.isdir(path_to_checkpoint), 'Error: no checkpoint found at %s.'%path_to_checkpoint
	checkpoint = torch.load(path_to_checkpoint + '/ckpt.t7')
	net.load_state_dict(checkpoint['net'])

	save_inp_out_size(net, inp_size)
	original_layers = get_list_of_layers(net, layerType=[nn.Conv2d, nn.Linear])

	flops_init = np.sum([get_flops(l) for l in original_layers])
	print('original flops was %d' % flops_init)

	if len(args.compressed_ckpt)>0:
		path_to_ckpt = args.compressed_ckpt
		masked_net = torch.load(path_to_ckpt)
	else:
		masked_net = convert_to_masked_model(net, decomposed='D' in args.phase, dataset=args.dataset, arch=args.arch)
		print("network architecture:")
	
	print(masked_net)
	if device == 'cuda':
		masked_net = torch.nn.DataParallel(masked_net)
		cudnn.benchmark = True

	optimizer = optim.SGD(masked_net.parameters(), lr=0, momentum=0, weight_decay=0)
	criterion = nn.CrossEntropyLoss()

	acc_val, _, _ = validate(valloader, masked_net, args)
	acc_original = acc_val.data.cpu()
	acc_test, _, _ = validate(testloader, masked_net, args)
	print('Validation accuracy before compression = %.2f%%' % (acc_original))
	print('Test accuracy before compression = %.2f%%' % (acc_test))
		
	save_inp_out_size(masked_net, inp_size)

	if 'D' in args.phase:
		#------------------ gather layer logistics for decomposition
		decomposed_layers = get_list_of_layers(masked_net, layerType=[decomposed_conv, svd_conv])
		
		genes_decomp_r0 = len(decomposed_layers)
		genes_decomp_r1 = len(decomposed_layers)
		num_quant = args.num_bins
		num_quant_svd = int(num_quant**2)

		decomposed_params = get_decomposed_parameters(args, decomposed_layers, num_quant)

		flops_original = get_all_flops(masked_net)
		print('total flops before decomposition:  (%.1f%%)' % (np.sum(flops_original), np.sum(flops_original)*100./flops_init))

		if args.phase=='D':
			# ------------------ boundary extraction and directed initialization
			valid_r0_per_layer, valid_r1_per_layer = get_valid_ranks(args, decomposed_layers, decomposed_params, num_quant, 
																	valloader, masked_net, acc_threshold=acc_original-1)
			r0_population_init, r1_population_init = init_population(args, valid_rates=[valid_r0_per_layer, valid_r1_per_layer], layers=decomposed_layers, 
												logistics=decomposed_params, valloader=valloader, masked_net=masked_net, device=device)

			# ------------------ run the genetic algorithm
			configs = run_genetic_decomposition(args, masked_net, valloader, r0_population_init, r1_population_init, flops_original, acc_original, 
									decomposed_layers, decomposed_params, valid_r0_per_layer, valid_r1_per_layer, flops_threshold=None)

			path = os.path.join('artifacts', args.dataset, args.arch, 'best_configs/decomposition')		
			if not os.path.exists(path):
				os.makedirs(path)
			for i in range(len(configs['best_configs'])):		
				pickle_name = os.path.join(path, 'iter_'+str(i)+'_acc_' + '%.2f'%(configs['accs'][i]) + \
									'_flops_' + '%.2f'%(configs['flops'][i]/np.sum(flops_original)) + '.pkl')
				with open(pickle_name, 'wb') as f:
					pickle.dump(configs['best_configs'][i], f)
			print('best configs saved')

		elif args.phase=='D_FT':
			assert len(args.config)>0, 'please provide the path to the desired per-layer configuration for fine-tuning' 
			print('loading configuration %s'%(os.path.abspath(args.config)))
			
			path_to_config = os.path.join('artifacts', args.dataset, args.arch, 'best_configs/decomposition', args.config)
			assert os.path.exists(path_to_config), 'no config file found at %s'%path_to_config
			with open(path_to_config, 'rb') as f:
				individual_r0, individual_r1 = pickle.load(f)

			for i, layer in enumerate(decomposed_layers):
			    r0 = individual_r0[i]
			    r1 = individual_r1[i]
			    first, core, last = decomposed_params[i][(r0,r1)]
			    layer.set_config(first, core, last)

			# val_acc = test(valloader, masked_net, verbose=False)
			# print('validation accuracy before fine-tuning = %.2f%%'%(val_acc))
			# test_acc = test(testloader, masked_net, verbose=False)
			# print('test accuracy before fine-tuning = %.2f%%'%(test_acc))
			
			# flops = np.sum(get_all_flops(masked_net, return_mask=False))
			# ratio = flops*1.0/np.sum(flops_original)
			# print('flops ratio = %.2f'%ratio)

			wrapped_net = wrap_decomposed_model(masked_net, dataset=args.dataset, arch=args.arch).to(device)
			save_inp_out_size(wrapped_net, inp_size)

			val_acc = test(valloader, wrapped_net, verbose=False)
			print('validation accuracy before fine-tuning = %.2f%%'%(val_acc))
			test_acc = test(testloader, wrapped_net, verbose=False)
			print('test accuracy before fine-tuning = %.2f%%'%(test_acc))
			
			flops = np.sum(get_all_flops(wrapped_net, return_mask=False))
			ratio = flops*1.0/flops_init
			print('flops ratio = %.2f'%ratio)

			lr_init = 1e-3
			optimizer = optim.SGD(masked_net.parameters(), lr=lr_init, momentum=0.9, weight_decay=0)
			criterion = nn.CrossEntropyLoss()
					
			best_acc = test_acc
			path_to_save = os.path.join('artifacts', args.dataset, args.arch, 'checkpoint/decomposition')
			if not os.path.exists(path_to_save):
				os.makedirs(path_to_save)
				torch.save(wrapped_net, os.path.join(path_to_save, 'ckpt.t7'))
			for epoch in range(1, 5):
				if epoch==3:
					lr = lr_init/10
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr
				
				train(trainloader, wrapped_net, criterion, optimizer, epoch, args)
				acc_test, _, _ = validate(testloader, wrapped_net, args, criterion)
				#-------------- Save checkpoint
				if acc_test > best_acc:
					print('Saving..')
					torch.save(wrapped_net, os.path.join(path_to_save, 'ckpt.t7'))
					best_acc = acc_test

			print('fine-tuned checkpoint saved to %s'%os.path.join(path_to_save, 'flops_%.2f'%(ratio)+'_acc_%.2f'%(best_acc)+'.t7'))
			os.rename(os.path.join(path_to_save, 'ckpt.t7'), os.path.join(path_to_save, 'flops_%.2f'%(ratio)+'_acc_%.2f'%(best_acc)+'.t7'))
	
	elif 'CP' in args.phase:
		#------------------ gather layer logistics for pruning
		all_sorted_args = get_all_pruning_priorities(masked_net, valloader, device, gradbased=True)
		layers_to_prune = get_list_of_layers(masked_net, layerType=[Mask_Layer])
		print(len(layers_to_prune))
		assert len(all_sorted_args)==len(layers_to_prune)

		if 'ResNet' in args.arch:
			new_sorted_args = []
			new_layers_to_prune = []
			to_be_removed = range(2, len(layers_to_prune), 2)
			for i, a in enumerate(all_sorted_args):
				if not (i in to_be_removed):
					new_sorted_args.append(a)
					new_layers_to_prune.append(layers_to_prune[i])
			all_sorted_args = new_sorted_args
			layers_to_prune = new_layers_to_prune

		masked_net.eval()
		flops_original, masks_before, masks_after = get_all_flops(masked_net)

		print('total flops before pruning: %d (%.1f%%)' % (np.sum(flops_original), np.sum(flops_original)*100./flops_init))

		if args.phase=='CP':
			# ------------------ boundary extraction and directed initialization
			max_prune_rates = get_max_prune_rates(args, layers_to_prune, all_sorted_args, valloader, masked_net, device, acc_threshold=acc_original-1)
			population_init = init_population(args, valid_rates=max_prune_rates, layers=layers_to_prune, 
												logistics=all_sorted_args, valloader=valloader, masked_net=masked_net, device=device)	

			# ------------------ run the genetic algorithm
			configs = run_genetic_pruning(args, masked_net, valloader, population_init, 
										flops_original, acc_original, layers_to_prune, masks_before, masks_after, all_sorted_args, flops_threshold=None)
			
			path = os.path.join('artifacts', args.dataset, args.arch, 'best_configs/channel_pruning')		
			if not os.path.exists(path):
				os.makedirs(path)
			for i in range(len(configs['best_configs'])):		
				pickle_name = os.path.join(path, 'iter_'+str(i)+'_acc_' + '%.2f'%(configs['accs'][i]) + \
									'_flops_' + '%.2f'%(configs['flops'][i]/np.sum(flops_original)) + '.pkl')
				with open(pickle_name, 'wb') as f:
					pickle.dump(configs['best_configs'][i], f)
			print('best configs saved')

		elif args.phase=='CP_FT':
			assert len(args.config)>0, 'please provide the path to the desired per-layer configuration for fine-tuning' 
			print('loading configuration %s'%(os.path.abspath(args.config)))
			
			path_to_config = os.path.join('artifacts', args.dataset, args.arch, 'best_configs/channel_pruning', args.config)
			assert os.path.exists(path_to_config), 'no config file found at %s'%path_to_config
			with open(path_to_config, 'rb') as f:
				curr_model = pickle.load(f)
			prune_all_masked_layers(curr_model, layers_to_prune, all_sorted_args)

			val_acc = test(valloader, masked_net, verbose=False)
			print('validation accuracy before fine-tuning = %.2f%%'%(val_acc))
			test_acc = test(testloader, masked_net, verbose=False)
			print('test accuracy before fine-tuning = %.2f%%'%(test_acc))

			flops = compute_flops_pruned_net(flops_original, masks_before, masks_after)	
			ratio = flops*1.0/flops_init
			print('flops ratio = %.2f'%ratio)

			lr_init = 1e-3
			optimizer = optim.SGD(masked_net.parameters(), lr=lr_init, momentum=0.9, weight_decay=0)
			criterion = nn.CrossEntropyLoss()
					
			best_acc = test_acc
			path_to_save = os.path.join('artifacts', args.dataset, args.arch, 'checkpoint/pruned')
			if not os.path.exists(path_to_save):
				os.makedirs(path_to_save)
				torch.save(masked_net, os.path.join(path_to_save, 'ckpt.t7'))
			for epoch in range(1, 5):
				if epoch==3:
					lr = lr_init/10
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr
				
				train(trainloader, masked_net, criterion, optimizer, epoch, args)
				acc_test, _, _ = validate(testloader, masked_net, args, criterion)
				#-------------- Save checkpoint
				if acc_test > best_acc:
					print('Saving..')
					torch.save(masked_net, os.path.join(path_to_save, 'ckpt.t7'))
					best_acc = acc_test

			print('fine-tuned checkpoint saved to %s'%os.path.join(path_to_save, 'flops_%.2f'%(ratio)+'_acc_%.2f'%(best_acc)+'.t7'))
			os.rename(os.path.join(path_to_save, 'ckpt.t7'), os.path.join(path_to_save, 'flops_%.2f'%(ratio)+'_acc_%.2f'%(best_acc)+'.t7'))

		else:
			assert False, 'invalid phase'

	


if __name__ == '__main__':
	main()