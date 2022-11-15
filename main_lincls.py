#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
# from sklearn.metrics import r2_score
from r2score import r2_score
import vits
import inspect

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.autograd import Variable
import hyper_resnet
from torchvision.datasets import DTD, SUN397, ImageFolder, OxfordIIITPet
# Test dataset info
from test_datasets import Flowers, CelebA, CIFAR10_rotation, FacesInTheWild300W, Caltech101, LeedsSportsPose
import wandb
import json

dataset_info = {
    'imagenet': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 100,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'dtd': {
        'class': DTD, 'dir': 'dtd', 'num_classes': 47,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'sun': {
        'class': SUN397, 'dir': 'sun', 'num_classes': 397,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'pets': {
        'class': OxfordIIITPet, 'dir': 'pets', 'num_classes': 37,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'rotation': {
        'class': CIFAR10_rotation, 'dir': 'CIFAR10', 'num_classes': 25,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'cub200': {
        'class': ImageFolder, 'dir': 'CUB200', 'num_classes': 200,
        'splits': ['train', 'train',  'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'cifar10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 10,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'cifar100': {
        'class': ImageFolder, 'dir': 'CIFAR100', 'num_classes': 100,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'flowers': {
        'class': Flowers, 'dir': 'flowers', 'num_classes': 102,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'mode': 'classification'
    }, 
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 10,
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
        'class': Caltech101, 'dir': 'caltech-101', 'num_classes': 102,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': 28,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'regression'
    }
}

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = torch.abs(error) < 1
    squared_loss = torch.square(error) / 2
    linear_loss  = torch.abs(error) - 0.5
    return torch.where(is_small_error, squared_loss, linear_loss)

# Helper functions to load test datasets
def get_dataset(args, c, d, s, t, shots):
    print("LOADING DATASET", d)
    if d == 'CelebA':
        return c(os.path.join(args.data_root, d), split=s, target_type=dataset_info[args.test_dataset]['target_type'], transform=t, download=False, shots = shots)
    elif d == 'CIFAR10' or d=='rotation':
        return c(os.path.join(args.data_root, d), train=s == 'train', transform=t, download=True)
    elif d == 'dtd' or d == 'pets':
        return c(os.path.join(args.data_root, d), split=s, transform=t, download=True)
    elif d=='sun':
        return c(os.path.join(args.data_root, d), transform=t, download=True)
    elif d == 'CUB200' or d == 'CIFAR100':
        return c(os.path.join(args.data_root, d, s), transform=t)
    else:
        if 'split' in inspect.getfullargspec(c.__init__)[0]:
            print("split in get_dataset", s)
            if s == 'val':
                try:
                    return c(os.path.join(args.data_root, d),  split=s, transform=t, shots = shots)
                except:
                    return c(os.path.join(args.data_root, d), split='valid', transform=t, shots = shots)
            else:
                print("get_dataset", s)
                return c(os.path.join(args.data_root, d), split=s, transform=t, shots = shots)
        else:
            return c(os.path.join(args.data_root, d), train=s == 'train', transform=t, shots = shots)


def prepare_data(args, norm, shots=None):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    if dataset_info[args.test_dataset]['splits'][1] == 'val':
        train_dataset = get_dataset(args, dataset_info[args.test_dataset]['class'],
                                    dataset_info[args.test_dataset]['dir'], 'train', transform, shots)
        val_dataset = get_dataset(args, dataset_info[args.test_dataset]['class'],
                                    dataset_info[args.test_dataset]['dir'], 'val', transform, shots)
        trainval = ConcatDataset([train_dataset, val_dataset])

    elif dataset_info[args.test_dataset]['splits'][1] == 'train':
        trainval = get_dataset(args, dataset_info[args.test_dataset]['class'],
                              dataset_info[args.test_dataset]['dir'], 'train', transform, shots)

    test = get_dataset(args, dataset_info[args.test_dataset]['class'],
                       dataset_info[args.test_dataset]['dir'], 'test', transform, shots)

    return trainval, test

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--baseline', action='store_true', help="Use resnet or hyper-resnet")
parser.add_argument('--discretize', action='store_true', help="Discretize invariances before testing")
parser.add_argument('--test_dataset', default='cifar10', type=str)
parser.add_argument('--data_root', default='/raid/s2265822/TestDatasets/', type = str)
parser.add_argument('--finetune', action='store_true',
                    help='Finetune - only use to test baseline')
# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

parser.add_argument('--few_shot_reg', default=None, type=float,
                    help='image size')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node, args.distributed )
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
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
            model = hyper_resnet.resnet50(num_classes  = dataset_info[args.test_dataset]['num_classes'])
            linear_keyword = 'fc'

    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            print("####################### LAST EPOCH ######################", checkpoint['epoch'])
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

            # args.start_epoch = 0
            print(state_dict.keys())
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg.missing_keys)
            # # if args.arch == 'vit_base':
            # #     assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.train()
    # freeze all layers but the last fc
    if not args.finetune:
            for name, param in model.named_parameters():
                if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword] and 'bn' not in name:
                    param.requires_grad = False
                print(name, param.requires_grad)

    # infer learning rate before changing batch size, not done in hyoer-models
    init_lr = args.lr 

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()
    print(model)

    # print(model)
    # define loss function (criterion) and optimizer
    # Loss function used for classification is CE and for rgeression we use L1Loss
    if dataset_info[args.test_dataset]['mode'] == 'classification':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.L1Loss().cuda(args.gpu)

    # optimize only the linear classifier
    if args.baseline:
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        learn_inv= None
        optimizer = torch.optim.SGD(parameters, args.lr,
                                 momentum=args.momentum,
                                weight_decay = 0.)
    else:
        learn_inv = Variable(torch.tensor([0.9504, 0.7267])).cuda()
        # learn_inv.requires_grad_()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer_model = torch.optim.SGD(parameters, lr=init_lr, weight_decay = args.weight_decay)
        optimizer_inv = torch.optim.Adam([learn_inv], lr = 0.1)
        optimizer = [optimizer_model, optimizer_inv]

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.test_dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        train_dataset, test_dataset = prepare_data(args, norm=imagenet_mean_std, shots = args.few_shot_reg)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,  shuffle=(train_sampler is None),
                                                    num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print("LENGTH OF DATASETS", len(train_dataset), len(test_dataset))
 
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.baseline:
        if not args.finetune:
            fname = "manyshot_"+ args.test_dataset + "_" + args.arch + "_" + "baseline.json"
        else:
            fname = "manyshot_" + args.test_dataset + "_" + args.arch + "_" + "baseline_finetune.json"
    else:
        fname = "manyshot_" + args.test_dataset + "_" + args.arch + "_hyper.json"
        
    log_json = open(os.path.join("results/", fname), "w")

    results_dict = []
 
    for epoch in range(args.start_epoch, args.epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        epoch_results = {}
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.baseline:
            adjust_learning_rate(optimizer, init_lr, epoch, args)
        else:
            adjust_learning_rate(optimizer[0], init_lr, epoch, args)
            adjust_learning_rate(optimizer[1], init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, learn_inv)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print("Time for training", start.elapsed_time(end))

        # evaluate on validation set
        if args.discretize:
            discretized_inv = torch.tensor([0.0, 0.0]).to(args.gpu)
            # print(learn_inv)
            if learn_inv[0] >=0.5:
                discretized_inv[0] = 1.0
            if learn_inv[1] >=0.5:
                discretized_inv[1] = 1.0
            
            acc1 = validate(val_loader, model, criterion, args, discretized_inv)
        else:
            acc1 = validate(val_loader, model, criterion, args, learn_inv)
        if not args.baseline:
            print("Invariance hyper-parameters", learn_inv)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        epoch_results["accuracy"] = acc1
        epoch_results["time"] = time
        results_dict.append(epoch_results)
        print(epoch_results)
    
    save_dict = {}
    save_dict["Results"] = results_dict
    json.dump(results_dict, log_json)

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
        # print(learn_inv)

        # compute output
        if args.baseline:
            output = model(images)
        else:

            output = model(images, learn_inv) 
            
        # Normalize outputs or targets for regression datasets for better comparison
        if args.test_dataset == 'leeds_sports_pose':
            output = nn.functional.normalize(output, dim=1)
            target = nn.functional.normalize(target, dim=1)
        if args.test_dataset == 'celeba':
            target = nn.functional.normalize(target, dim=1)
        loss = criterion(output, target) 

        # measure accuracy and record loss
        if dataset_info[args.test_dataset]['mode'] == 'classification':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1 = [r2_score(target.flatten().detach().cpu().numpy(), output.flatten().detach().cpu().numpy())]
            acc5 = [criterion(output, target).item()]

        avg_acc += acc1[0]
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
    avg_acc = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.baseline:
                output = model(images)
            else:
                output = model(images, learn_inv) # mode='downstream')
            if args.test_dataset == 'leeds_sports_pose':
                output = nn.functional.normalize(output, dim=1)
                target = nn.functional.normalize(target, dim=1)
            if args.test_dataset == 'celeba':
                target = nn.functional.normalize(target, dim=1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if dataset_info[args.test_dataset]['mode'] == 'classification':
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            else:
                acc1 = [r2_score(target.flatten().detach().cpu().numpy(), output.flatten().detach().cpu().numpy())]
                acc5 = [criterion(output, target).item()]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            avg_acc += acc1[0]
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg


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


if __name__ == '__main__':
    main()
