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
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import *

prune_meter = time_meter()
eval_meter = time_meter()
reload_meter = time_meter()
step_meter = time_meter()

def init_population(args, valid_rates, layers, logistics, valloader, masked_net, device):
	pop_init_name = 'pop_init_phase_' + args.phase + '_' + str(args.num_population) + '_' + str(args.acc_threshold) + '.pkl'
	path_to_pkl = os.path.join('artifacts', args.dataset, args.arch, pop_init_name)
	if os.path.exists(path_to_pkl):
		print('loading population from %s'%path_to_pkl)
		with open(path_to_pkl, 'rb') as f:
			population_init = pickle.load(f)
			return population_init
			# if 'D' in args.phase:
			# 	population_r0, population_r1 = pickle.load(f)
			# 	return population_r0, population_r1
			# else:
			# 	population_init = pickle.load(f)
			# 	return population_init
	else:
		if 'D' in args.phase:
			decomposed_params = logistics
			valid_r0_per_layer, valid_r1_per_layer = valid_rates

			genes_decomp_r0 = len(layers)
			genes_decomp_r1 = len(layers)
			num_quant_svd = int(args.num_bins**2)
			
			r0_population_init = np.ones((args.num_population, genes_decomp_r0)).astype(np.int32)
			r1_population_init = np.ones((args.num_population, genes_decomp_r1)).astype(np.int32)
			pop = 0
			while pop<args.num_population:
				individual_r0 = np.asarray([np.random.choice(valid_r0_per_layer[i]) for i in range(len(layers))])
				individual_r1 = np.asarray([np.random.choice(valid_r1_per_layer[i]) for i in range(len(layers))])
				decompose_all_layers(layers, decomposed_params, individual_r0, individual_r1, num_quant_svd)
				acc = test(valloader, masked_net, verbose=False)
				# flops = np.sum(np.asarray([get_flops(l) for l in layers_with_flops]))
				# print('acquired accuracy: %.2f, flops: %.2f' %(acc, flops*1./np.sum(flops_original)))
				if acc>=args.acc_threshold:
					r0_population_init[pop] = individual_r0
					r1_population_init[pop] = individual_r1
					pop += 1
					print(pop)

			print('population initialized')
			with open(path_to_pkl, 'wb') as f:
				pickle.dump([r0_population_init, r1_population_init], f)
				print('saving population to %s'%path_to_pkl)

			return r0_population_init, r1_population_init
		
		elif 'CP' in args.phase: 
			max_prune_rates = valid_rates
			all_sorted_args = logistics
			
			genes_per_individual = len(layers)
			
			population_init = np.zeros((args.num_population, genes_per_individual))
			pop = 0
			while pop < args.num_population:
				individual = np.random.normal(loc=max_prune_rates/2, scale=max_prune_rates/4, size=genes_per_individual)
				individual = np.clip(individual, 0, 1)
				prune_all_masked_layers(individual, layers, all_sorted_args)
				acc = test(valloader, masked_net)
				if acc >= args.acc_threshold:
					population_init[pop] = individual
					pop += 1
					print(pop)
			
			print('population initialized')
			with open(path_to_pkl, 'wb') as f:
				pickle.dump(population_init, f)
				print('saving population to %s'%path_to_pkl)

			return population_init

		else:
			assert False, 'phase not supported'


def accuracy_bound_score(original_acc, original_flops, new_acc, new_flops, acc_threshold, coeff=100., saving_threshold=None):
	saving = original_flops - new_flops
	saving = saving*1./original_flops
	if saving<0:
		return 0
	error = np.absolute(original_acc - new_acc)+1
	
	if new_acc < acc_threshold:
		error = error + np.exp(acc_threshold - new_acc)
	
	if saving_threshold is None:
		reward = np.exp(saving*coeff)/error
	else:
		reward = np.exp((saving-saving_threshold)*coeff)/error

	# print('saving: %.2f, new_acc: %.2f, original_acc: %.2f, threshold: %.2f, score: %.3f'%(saving, new_acc, original_acc, acc_threshold, reward))
		
	return reward


def get_best_individual(accs, flops, acc_threshold):
	f = np.Infinity
	index = -1
	for i, a in enumerate(accs):
		if a >= acc_threshold:
			if flops[i] <= f:
				f = flops[i]
				index = i
	if index==-1:
		index = np.argmax(accs)
	
	return index, accs[index], flops[index]


def crossover(father, mother, prob_cross, per_bit_exchange_prob):
	p = np.random.rand()
	if p <= prob_cross:	
		#------------------- do crossover
		for i in range(len(father)):
			p = np.random.rand()
			if p < per_bit_exchange_prob:
				#-------------- swap element of mother & father
				temp = father[i]
				father[i] = mother[i]
				mother[i] = temp
	return father, mother


def mutate_discrete(individual, prob_mutate, per_bit_mutation_prob, valid_r0_per_layer, valid_r1_per_layer, mutation_ranges=None):
	
	if mutation_ranges is None:
		mutation_ranges=np.ones(len(individual[0])).astype(np.int32)
	p = np.random.rand()
	r0_vec = individual[0]
	r1_vec = individual[1]
	 
	if p < prob_mutate:
		for i in range(len(r0_vec)):
			assert r0_vec[i] in valid_r0_per_layer[i]
			p = np.random.rand()
			if p <= per_bit_mutation_prob:
				coin_flip = np.random.rand()
				if coin_flip>0.5:
					for j in range(len(valid_r0_per_layer[i])):
						if valid_r0_per_layer[i][j]==r0_vec[i]:
							break;
					to_add = np.random.randint(1,mutation_ranges[i]+1)
					if j+to_add>=len(valid_r0_per_layer[i]):
						r0_vec[i]=copy.deepcopy(valid_r0_per_layer[i][-1])
					else:
						r0_vec[i]=copy.deepcopy(valid_r0_per_layer[i][j+to_add])
					
				else:
					for j in range(len(valid_r0_per_layer[i])):
						if valid_r0_per_layer[i][j]==r0_vec[i]:
							break;
					to_dec = np.random.randint(1,mutation_ranges[i]+1)
					if j-to_dec<0:
						r0_vec[i]=copy.deepcopy(valid_r0_per_layer[i][0])
					else:
						r0_vec[i]=copy.deepcopy(valid_r0_per_layer[i][j-to_dec])

		for i in range(len(r1_vec)):
			assert r1_vec[i] in valid_r1_per_layer[i]
			p = np.random.rand()
			if p <= per_bit_mutation_prob:
				coin_flip = np.random.rand()
				if coin_flip>0.5:
					to_add = np.random.randint(1,mutation_ranges[i]+1)
					for j in range(len(valid_r1_per_layer[i])):
						if valid_r1_per_layer[i][j]==r1_vec[i]:
							break;
					if j+to_add>=len(valid_r1_per_layer[i]):
						r1_vec[i]=copy.deepcopy(valid_r1_per_layer[i][-1])
					else:
						r1_vec[i]=copy.deepcopy(valid_r1_per_layer[i][j+to_add])
					
				else:
					for j in range(len(valid_r1_per_layer[i])):
						if valid_r1_per_layer[i][j]==r1_vec[i]:
							break;
					to_dec = np.random.randint(1,mutation_ranges[i]+1)
					if j-to_dec<0:
						r1_vec[i]=copy.deepcopy(valid_r1_per_layer[i][0])
					else:
						r1_vec[i]=copy.deepcopy(valid_r1_per_layer[i][j-to_dec])
  
	return [r0_vec, r1_vec]


def mutate_continuous(individual, prob_mutate, per_bit_mutation_prob, scale):
	p = np.random.rand()
	if p < prob_mutate:
		#--------------- do mutation
		for i in range(len(individual)):
			p = np.random.rand()
			if p <= per_bit_mutation_prob:
				#----------- mutate this gene
				noise = np.random.normal(loc=0.0, scale=scale[i])
				individual[i] = individual[i] + noise
	
	individual = np.clip(individual, 0, 1)
	return individual


def run_genetic_decomposition(args, masked_net, valloader, population_r0, population_r1, original_flops, original_acc, 
				decomposed_layers, decomposed_params, valid_r0_per_layer, valid_r1_per_layer, flops_threshold=None):
	path_to_heatmap = os.path.join('artifacts', args.dataset, args.arch, 'heatmap/decomposition')
	if not os.path.exists(path_to_heatmap):
		os.makedirs(path_to_heatmap)

	#-----------------evaluate all individuals:
	fitness_scores = []
	accuracies = []
	flops_all = []
	for index in range(args.num_population):
		for i, layer in enumerate(decomposed_layers):
			first, core, last = decomposed_params[i][(population_r0[index][i], population_r1[index][i])]
			layer.set_config(first, core, last)		

		new_acc = test(valloader, masked_net)
		new_flops = np.sum(get_all_flops(masked_net, return_mask=False))
		# new_flops = compute_flops_pruned_net(original_flops, masks_before, masks_after)
		score = accuracy_bound_score(original_acc, np.sum(original_flops), new_acc, new_flops, args.acc_threshold, args.coeff, flops_threshold)
		
		fitness_scores.append(score)
		accuracies.append(new_acc)
		flops_all.append(new_flops)
			   
	fitness_scores = np.asarray(fitness_scores)
	accuracies = np.asarray(accuracies)
	flops_all = np.asarray(flops_all)

	best_id, best_acc, best_flops = get_best_individual(accuracies, flops_all, args.acc_threshold)

	print('Initial population => average flops: %.3f, average accuracy:%0.2f%%, best flops: %0.3f, best accuracy:%0.2f%%'
		  %(np.mean(flops_all*1./np.sum(original_flops)), np.mean(accuracies), best_flops*1./np.sum(original_flops), best_acc))

	configs = {'best_configs':[], 'accs':[], 'flops':[], 'population_r0':[], 'population_r1':[]}
	for iter in range(args.iter):
		fitness_scores = fitness_scores - np.min(fitness_scores) 
		
		if np.sum(fitness_scores) == 0:
			fitness_scores = np.ones(len(fitness_scores))
		normalized_scores = fitness_scores/np.sum(fitness_scores)
				
		#-----------------random selection:
		indices = np.arange(args.num_population)
		choices = np.random.choice(indices, size=args.num_population, replace=True, p=normalized_scores)
		
		new_population_pruning = []
		new_population_r0 = []
		new_population_r1 = []		
		for index in range(args.num_population/2):
			#----------------- crossover
			father = [population_r0[choices[2*index]],population_r1[choices[2*index]]]
			mother = [population_r0[choices[2*index+1]],population_r1[choices[2*index+1]]]
			kid1, kid2 = crossover(father, mother, args.p_cross, args.p_swap)
			
			#----------------- mutation
			kid1 = mutate_discrete(kid1, args.p_mutate, args.p_tweak, valid_r0_per_layer, valid_r1_per_layer)
			kid2 = mutate_discrete(kid2, args.p_mutate, args.p_tweak, valid_r0_per_layer, valid_r1_per_layer)
			new_population_r0 = new_population_r0 + [kid1[0], kid2[0]]
			new_population_r1 = new_population_r1 + [kid1[1], kid2[1]]
		
		
		population_r0 = np.asarray(new_population_r0)
		population_r1 = np.asarray(new_population_r1)
		  
		#-----------------evaluate all individuals:
		fitness_scores = []
		accuracies = []
		flops_all = []
		t2=time.time()
		for index in range(args.num_population):
			for i, layer in enumerate(decomposed_layers):
				first, core, last = decomposed_params[i][(population_r0[index][i], population_r1[index][i])]
				layer.set_config(first, core, last)		

			new_acc = test(valloader, masked_net)
			new_flops = np.sum(get_all_flops(masked_net, return_mask=False))
			# new_flops = compute_flops_pruned_net(original_flops, masks_before, masks_after)
			score = accuracy_bound_score(original_acc, np.sum(original_flops), new_acc, new_flops, args.acc_threshold, args.coeff, flops_threshold)
			
			fitness_scores.append(score)
			accuracies.append(new_acc)
			flops_all.append(new_flops)
		t3 = time.time()
		step_meter.add(t3-t2)
		fitness_scores = np.asarray(fitness_scores)
		accuracies = np.asarray(accuracies)
		flops_all = np.asarray(flops_all)
		
		best_id, best_acc, best_flops = get_best_individual(accuracies, flops_all, args.acc_threshold)
		
		configs['population_r0'].append(copy.deepcopy(population_r0))
		configs['population_r1'].append(copy.deepcopy(population_r1))
		configs['best_configs'].append(copy.deepcopy([population_r0[best_id], population_r1[best_id]]))
		configs['accs'].append(best_acc)
		configs['flops'].append(best_flops)
		
		print('iter: %d, average flops: %.3f, average accuracy:%0.2f%%, best flops: %0.3f, best accuracy:%0.2f%%'
		  		%(iter+1, np.mean(flops_all*1./np.sum(original_flops)), np.mean(accuracies), best_flops*1./np.sum(original_flops), best_acc))
		
		print('average time per step: %.2f' % step_meter.avg)

	return configs


def run_genetic_pruning(args, masked_net, valloader, population, flops_original, acc_original,
					layers_to_prune, masks_before, masks_after, all_sorted_args, flops_threshold=None):
	genes_per_individual = len(layers_to_prune)
	mutate_scales = np.ones(genes_per_individual)*args.mutate_s
	
	path_to_heatmap = os.path.join('artifacts', args.dataset, args.arch, 'heatmap/pruning')
	if not os.path.exists(path_to_heatmap):
		os.makedirs(path_to_heatmap)

	#-----------------evaluate all individuals:
	fitness_scores = []
	accuracies = []
	flops_all = []
	for index in range(args.num_population):
		t0 = time.time()
		prune_all_masked_layers(population[index], layers_to_prune, all_sorted_args)
		flops_after_pruning = compute_flops_pruned_net(flops_original, masks_before, masks_after)
		prune_meter.add(time.time()-t0)

		t0 = time.time()
		acc_after_pruning = test(valloader, masked_net)
		eval_meter.add(time.time()-t0)

		score = accuracy_bound_score(acc_original, np.sum(flops_original), acc_after_pruning, flops_after_pruning, args.acc_threshold, args.coeff, flops_threshold)

		fitness_scores.append(score)
		accuracies.append(acc_after_pruning)
		flops_all.append(flops_after_pruning)
	print(fitness_scores)	
	fitness_scores = np.asarray(fitness_scores)
	accuracies = np.asarray(accuracies)
	flops_all = np.asarray(flops_all)

	fname = os.path.join(path_to_heatmap, 'itr_0.png')
	heatmap(1-population.transpose(), fname,
			[str(xx+1) for xx in range(genes_per_individual)],
			[str(xx+1) for xx in range(args.num_population)], 
			cmap="YlGn", 
			cbarlabel="Density")

	best_id, best_acc, best_flops = get_best_individual(accuracies, flops_all, args.acc_threshold)

	print('Initial population => average flops: %.3f, average accuracy:%0.2f%%, best flops: %0.3f, best accuracy:%0.2f%%'
		  %(np.mean(flops_all/np.sum(flops_original)), np.mean(accuracies), best_flops/np.sum(flops_original), best_acc))


	configs = {'best_configs':[],'accs':[],'flops':[], 'population': []}
	for iter in range(args.iter):	
		fitness_scores = fitness_scores - np.min(fitness_scores) 
		
		if np.sum(fitness_scores) == 0:
			fitness_scores = np.ones(len(fitness_scores))
		normalized_scores = fitness_scores/np.sum(fitness_scores)
		
		#-----------------random selection:
		indices = np.arange(args.num_population)
		choices = np.random.choice(indices, size=args.num_population, replace=True, p=normalized_scores)
		
		new_population = []
		for index in range(args.num_population/2):
			#----------------- crossover
			father = population[choices[2*index]]
			mother = population[choices[2*index+1]]
			kid1, kid2 = crossover(father, mother, args.p_cross, args.p_swap)
			
			#----------------- mutation
			kid1 = mutate_continuous(kid1, args.p_mutate, args.p_tweak, mutate_scales)
			kid2 = mutate_continuous(kid2, args.p_mutate, args.p_tweak, mutate_scales)
			new_population = new_population + [kid1, kid2]
		
		population = np.asarray(new_population)
		  
		#-----------------evaluate all individuals:
		fitness_scores = []
		accuracies = []
		flops_all = []
		t2 = time.time()
		for index in range(args.num_population):
			t0 = time.time()
			prune_all_masked_layers(population[index], layers_to_prune, all_sorted_args)
			flops_after_pruning = compute_flops_pruned_net(flops_original, masks_before, masks_after)
			prune_meter.add(time.time()-t0)

			t0 = time.time()
			acc_after_pruning = test(valloader, masked_net)
			eval_meter.add(time.time()-t0)

			score = accuracy_bound_score(acc_original, np.sum(flops_original), acc_after_pruning, flops_after_pruning, args.acc_threshold, args.coeff, flops_threshold)

			fitness_scores.append(score)
			accuracies.append(acc_after_pruning)
			flops_all.append(flops_after_pruning)
		print(fitness_scores)
		t3 = time.time()
		step_meter.add(t3-t2)
		fitness_scores = np.asarray(fitness_scores)
		accuracies = np.asarray(accuracies)
		flops_all=np.asarray(flops_all)
		
		fname = os.path.join(path_to_heatmap, 'itr_'+str(iter)+'.png')
		heatmap(1-population.transpose(), fname,
				[str(xx+1) for xx in range(genes_per_individual)],
				[str(xx+1) for xx in range(args.num_population)], 
				cmap="YlGn", 
				cbarlabel="Density")
		best_id, best_acc, best_flops = get_best_individual(accuracies, flops_all, args.acc_threshold)

		configs['best_configs'].append(copy.deepcopy(population[best_id]))
		configs['accs'].append(best_acc)
		configs['flops'].append(best_flops)
		configs['population'].append(copy.deepcopy(population))
		
		print('iter: %d, average flops: %.3f, average accuracy:%0.2f%%, best flops: %0.3f, best accuracy:%0.2f%%'
		  		%(iter+1, np.mean(flops_all/np.sum(flops_original)), np.mean(accuracies), best_flops/np.sum(flops_original), best_acc))
		
		print('average time per step: %.2f' % step_meter.avg)

	return configs
