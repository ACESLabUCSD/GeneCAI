import shutil
import time
import torch
from tqdm import tqdm

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    reverseChannels = False
    if args.dataset=='ImageNet':
        reverseChannels = not(args.arch=='alexnet')
        revIdx = [2,1,0]
        revIdx = torch.cuda.LongTensor(revIdx)
    
    with tqdm(total=len(train_loader)) as pbar:
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if reverseChannels:
                input= input.index_select(1, revIdx)
            if not isinstance(input, torch.cuda.FloatTensor):
                input = input.cuda()
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.update(1)
            pbar.set_description('Epoch: %d | Loss: %.3f | Acc: %.3f%%'% (epoch, losses.avg, top1.avg))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, args, criterion=None , num_batches=None,verbose=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    reverseChannels = args.dataset=='ImageNet' and not args.arch=='alexnet'
    revIdx = torch.cuda.LongTensor([2,1,0])
    
    end = time.time()
    with tqdm(total=len(val_loader)) as pbar:
        for i, (input, target) in enumerate(val_loader):
            if num_batches:
                if i>num_batches:
                    break
            if reverseChannels:
                input = input.index_select(1, revIdx)
            if not isinstance(input, torch.cuda.FloatTensor):
                input = input.cuda()
            target = target.cuda(async=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)

            # measure accuracy and record loss
            if criterion is not None:
                loss = criterion(output, target_var)
                losses.update(loss.data.item(), input.size(0))
            else:
                losses.update(0.0, input.size(0))
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))            
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.update(1)

    if verbose:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def test(loader, net, verbose=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
     
    acc = 100.*correct/total
    if verbose:
        print(' * Prec@1 {:.2f}'.format(acc))
    return acc


def save_checkpoint(state, is_best, filename='checkpoint', save_architecture=False,model=None):
    if not save_architecture:
        torch.save(state, filename+'.pth.tar')
        if is_best:
            shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')
    else:
        torch.save(model, filename+'.pth.tar')
        if is_best:
            shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset=='CIFAR10':
        if args.arch=='vgg16':
            if epoch<90:
                lr = 0.1
            elif epoch<136:
                lr = 0.01
            else:
                lr = 0.001
        elif 'ResNet' in args.arch:
            if epoch<150:
                lr = 0.1
            elif epoch<250:
                lr = 0.01
            else:
                lr = 0.001
        else:
            raise('architecture not in pre-specified list')
        # lr = args.lr * (0.1 ** (epoch // 100))
    else:
        raise('no learning rate scheduler avialable for dataset %s'%args.dataset)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

