from vgg import *
from dpn import *
from lenet import *
from senet import *
from pnasnet import *
from densenet import *
from googlenet import *
from shufflenet import *
from resnet import *
from resnext import *
from preact_resnet import *
from mobilenet import *
from mobilenetv2 import *
from layers import *

def get_model(args, device='cuda'):
	if args.arch == 'VGG16':
	    net = VGG('VGG16')
	elif args.arch == 'ResNet56':
	    net = ResNet56()
	elif args.arch == 'ResNet110':
	    net = ResNet110()
	    
	net = net.to(device)
	if device == 'cuda':
	    net = torch.nn.DataParallel(net)
	    cudnn.benchmark = True

	return net
