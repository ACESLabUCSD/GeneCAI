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

from models import *
from utils.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

parser.add_argument('--arch', metavar='ARCHITECTURE', default='VGG16',
					choices=model_names,
					help='model architecture: [VGG16, ResNet56, ResNet110] (default: VGG16)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--e', '--epochs', default=300, type=float,
					help='number of epochs to train')
parser.add_argument('--resume', action='store_true',
						help='resume from latest checkpoint (default: False)')
parser.add_argument('--train', action='store_true',
					help='train the model (default: False)')
args = parser.parse_args()
args.dataset = 'CIFAR10'


def main():
	path_to_save = os.path.join(args.dataset, args.arch, 'checkpoint')
	if not os.path.exists(path_to_save):
		os.makedirs(path_to_save)
	
	# ----------------- Data
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
	trainset.data = trainset.data[0:49000]
	trainset.targets = trainset.targets[0:49000]
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
	valset.data = valset.data[49000:]
	valset.targets = valset.targets[49000:]
	valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	#---------------- Model
	print('==> Building model..')
	if args.arch == 'VGG16':
		net = VGG('VGG16')
	elif args.arch == 'ResNet56':
		net = ResNet56()
	elif args.arch == 'ResNet110':
		net = ResNet110()
	print(net)

	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	
	if args.resume:
		#---------------- Load checkpoint.
		print('==> Resuming from checkpoint..')
		assert os.path.isdir(path_to_save), 'Error: no checkpoint directory found at %s'%path_to_save
		checkpoint = torch.load(os.path.join(path_to_save, 'ckpt.t7'))
		net.load_state_dict(checkpoint['net'])
		best_acc = checkpoint['acc']
		start_epoch = checkpoint['epoch']
	else:
		best_acc = 0	
		start_epoch = 0


	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	if args.train:

		rain(train_loader, model, criterion, optimizer, epoch, args)

		for epoch in range(start_epoch, args.epochs):
			adjust_learning_rate(optimizer, epoch, args)
			
			train_loss, train_top1, train_top5 = train(trainloader, net, criterion, optim, epoch, args)
			
			test_loss, test_top1, test_top5 = validate(testloader, net, criterion, args)
			print('test accuracy: %.2f' % test_top1)
			
			val_loss, val_top1, val_top5 = validate(valloader, net, criterion, args)
			print('validation accuracy: %.2f' % acc_val)
			
			#-------------- Save checkpoint
			if val_top1 > best_acc:
				print('Saving..')
				state = {
					'net': net.state_dict(),
					'acc': acc_test,
					'epoch': epoch,
				}
				torch.save(os.path.join(path_to_save, 'ckpt.t7'))
				best_acc = val_top1

	elif args.evaluate:
		test_loss, test_top1, test_top5 = validate(testloader, net, criterion, args)
		print('test accuracy: %.2f' % test_top1)


if __name__ == '__main__':
	main()