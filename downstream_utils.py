import argparse
import builtins
from codecs import namereplace_errors
import math
from mimetypes import init
from modulefinder import Module
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import tllib.vision.datasets as datasets
import torchvision.models as torchvision_models
from sklearn.metrics import r2_score
import vits
import inspect
from torch.utils.data import ConcatDataset
from torch.autograd import Variable
import hyper_resnet
# Test dataset info
from test_datasets import CelebA, FacesInTheWild300W, LeedsSportsPose
import wandb
import json


dataset_info = {
    'dtd': {
        'class': datasets.DTD, 'dir': 'dtd', 'num_classes': 47,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'pets': {
        'class': datasets.OxfordIIITPets, 'dir': 'pets', 'num_classes': 37,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'cub200': {
        'class': datasets.CUB200, 'dir': 'CUB200', 'num_classes': 200,
        'splits': ['train', 'train',  'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'cifar10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10/', 'num_classes': 10,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'cifar100': {
        'class': datasets.CIFAR100, 'dir': 'CIFAR100', 'num_classes': 100,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'flowers': {
        'class': datasets.OxfordFlowers102, 'dir': 'flowers/', 'num_classes': 102,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.5,
        'mode': 'classification'
    }, 
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 40,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'target_type': 'landmarks',
        'mode': 'regression'
    },
    '300w': {
        'class': FacesInTheWild300W, 'dir': '300W', 'num_classes': 136,
        'splits': ['train', 'val', 'test'], 'split_size': 0.5,
        'mode': 'regression'
    },
    'caltech101': {
        'class': datasets.Caltech101, 'dir': 'Caltech101', 'num_classes': 102,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': 28,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'regression'
    }
}

# Helper functions to load test datasets
def get_dataset(args, c, d, s, t, shots):
    if args.test_dataset == 'CelebA':
        return c(os.path.join(args.data_root, d), split=s, target_type=dataset_info[args.test_dataset]['target_type'], transform=t, download=False, shots = shots)
    elif args.test_dataset in ['dtd', 'pets', 'cifar10', 'cifar100', 'flowers', 'caltech101', 'cub200']:
        return c(os.path.join(args.data_root, d), split=s, transform=t, download=False)
    else:
        if 'split' in inspect.getfullargspec(c.__init__)[0]:
            if s == 'val':
                try:
                    return c(os.path.join(args.data_root, d),  split=s, transform=t, shots = shots)
                except:
                    return c(os.path.join(args.data_root, d), split='valid', transform=t, shots = shots)
            else:
                return c(os.path.join(args.data_root, d), split=s, transform=t, shots = shots)
        else:
            return c(os.path.join(args.data_root, d), train=s == 'train', transform=t, shots = shots)


def prepare_data(args, norm, shots=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    if dataset_info[args.test_dataset]['splits'][1] in ['valid', 'val', 'validation']:
        train_dataset = get_dataset(args, dataset_info[args.test_dataset]['class'],
                                    dataset_info[args.test_dataset]['dir'], 'train', transform, shots)
        val_dataset = get_dataset(args, dataset_info[args.test_dataset]['class'],
                                    dataset_info[args.test_dataset]['dir'], dataset_info[args.test_dataset]['splits'][1], transform, shots)
        trainval = ConcatDataset([train_dataset, val_dataset])

    elif dataset_info[args.test_dataset]['splits'][1] == 'train':
        trainval = get_dataset(args, dataset_info[args.test_dataset]['class'],
                              dataset_info[args.test_dataset]['dir'], 'train', transform, shots)

    test = get_dataset(args, dataset_info[args.test_dataset]['class'],
                       dataset_info[args.test_dataset]['dir'], 'test', transform, shots)

    return trainval, test

def load_backbone(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        if args.baseline:
            model = vits.__dict__[args.arch](num_classes = dataset_info[args.test_dataset]['num_classes'])
            linear_keyword = 'head'
        else:
            model = vits.PromptVisionTransformerMoCo(num_classes = dataset_info[args.test_dataset]['num_classes'])
            linear_keyword = 'head'
    else:
        if args.baseline:
            model = torchvision_models.__dict__[args.arch](num_classes = dataset_info[args.test_dataset]['num_classes'])
            linear_keyword = 'fc'
        else:
            model = hyper_resnet.resnet50(num_classes  = dataset_info[args.test_dataset]['num_classes'], inv_dim=2)
            linear_keyword = 'fc'

    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            # print("####################### LAST EPOCH ######################", checkpoint['epoch'])
            if args.baseline:
                if args.arch == 'resnet50':
                    base_str = 'encoder_q'
                else:
                    base_str = "base_encoder"
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.%s' % base_str) and not k.startswith('module.%s.%s' % (base_str, linear_keyword)):
                        # remove prefix
                        state_dict[k[len('module.%s' % base_str):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
            else:
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            print(state_dict.keys())
            msg = model.load_state_dict(state_dict, strict=False)

            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    return model, linear_keyword

def train(train_loader, model, criterion, optimizer, epoch, args, learn_inv):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avg_acc = 0.0
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.baseline:
            output = model(images)
        else:
            output = model(images, learn_inv) 
            
        # Normalize outputs or targets for regression datasets for better comparison
        if args.test_dataset in ['leeds_sports_pose', 'celeba', '300w']:
            output = nn.functional.normalize(output, dim=1)
            target = nn.functional.normalize(target, dim=1)

        loss = criterion(output, target) 

        # measure accuracy and record loss
        if dataset_info[args.test_dataset]['mode'] == 'classification':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.test_dataset in ["caltech101", "pets", "flowers"]:
                acc1 = mean_per_class_accuracy(dataset_info[args.test_dataset]['num_classes'], output, target)
        else:
            acc1 = [r2_score(target.flatten().detach().cpu().numpy(), output.flatten().detach().cpu().numpy())]
            acc5 = [criterion(output, target).item()]

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
       
        # compute gradient and do SGD step
        if args.baseline:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer[0].zero_grad() # Update model parameters
            optimizer[1].zero_grad() # Update invariances
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            learn_inv.data.clamp_(0.0, 1.0) # Clamp invariances between 0 and 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.item(), top5.avg

def validate(val_loader, model, criterion, args, learn_inv):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            with  torch.no_grad():
                # compute output
                if args.baseline:
                    output = model(images)
                else:
                    output = model(images, learn_inv) # mode='downstream')

                if args.test_dataset in ['leeds_sports_pose', 'celeba', '300w']:
                    output = nn.functional.normalize(output, dim=1)
                    target = nn.functional.normalize(target, dim=1)

                loss = criterion(output, target)

            # measure accuracy and record loss
            if dataset_info[args.test_dataset]['mode'] == 'classification':
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                if args.test_dataset in ["caltech101", "pets", "flowers"]:
                    acc1 = mean_per_class_accuracy(dataset_info[args.test_dataset]['num_classes'], output, target)
            else:
                acc1 = [r2_score(target.flatten().detach().cpu().numpy(), output.flatten().detach().cpu().numpy())]
                acc5 = [criterion(output, target).item()]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg.item(), top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mean_per_class_accuracy(num_classes, outputs, targets):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    _, preds = torch.max(outputs, 1)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix.diag()/confusion_matrix.sum(1)