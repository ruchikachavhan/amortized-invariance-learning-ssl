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
from torch.autograd import Variable
import wandb
from downstream_utils import *
import json


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Downstream training')

# Basic Training arguments
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--results-dir', default='results/', type = str)
parser.add_argument('--save_fc_model', action='store_true', help="If true, save the classifier state dict")

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

# DDP args
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

# args for Amortised Invariances
parser.add_argument('--baseline', action='store_true', help="If true, loads the baseline model")
parser.add_argument('--discretize', action='store_true', help="Discretize invariances before testing")

# additional configs:
parser.add_argument('--test_dataset', default='', type=str)
parser.add_argument('--data_root', default='../TestDatasets/', type = str)
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--few_shot_reg', default=None, type=float,
                    help='Performs few shot regression with given split size')


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

    model, linear_keyword = load_backbone(args)
    model.train()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

    # infer learning rate before changing batch size, not done for hyper-models
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
        optimizer = torch.optim.Adam(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay = 0.)
    else:
        learn_inv = Variable(torch.tensor([0.5, 0.5])).cuda()
        learn_inv.requires_grad_()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer_model = torch.optim.Adam(parameters, lr=init_lr, weight_decay = args.weight_decay)
        optimizer_inv = torch.optim.Adam([learn_inv], lr = 0.01) # Fixed lr for invariances with no weight decay
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
    if args.test_dataset == 'imagenet': # imagenet
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
        print("Size of train and test datasets", len(train_dataset), len(test_dataset))
 

    # If args.results does not exist
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    results_file = open(os.path.join(args.results_dir, args.test_dataset + "_" + args.arch + "_hyper.json"), "w")
    results_dict = {}
    best_results = {}
    best_acc1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        epoch_results = {}
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.baseline:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            adjust_learning_rate(optimizer[0], epoch, args)
            adjust_learning_rate(optimizer[1], epoch, args)

        # train for one epoch
        train_acc1, _ = train(train_loader, model, criterion, optimizer, epoch, args, learn_inv)

        # evaluate on validation set
        if args.discretize:
            discretized_inv = torch.tensor([0.0, 0.0]).to(args.gpu)
            if learn_inv[0] >=0.5:
                discretized_inv[0] = 1.0
            if learn_inv[1] >=0.5:
                discretized_inv[1] = 1.0
            
            val_acc1, _ = validate(val_loader, model, criterion, args, discretized_inv)
        else:
            val_acc1, _ = validate(val_loader, model, criterion, args, learn_inv)
        
        if val_acc1 > best_acc1:
            best_results['Epoch'] = epoch
            best_results['training accuracy'] = train_acc1
            best_results["val accuracy"] = val_acc1
            best_results['Invariances'] = learn_inv.cpu().detach().numpy().tolist()
            best_acc1 = val_acc1
            is_best = True
        else:
            is_best = False

        if not args.baseline:
            print("Invariance hyper-parameters", learn_inv)
            epoch_results['Invariances'] = learn_inv.cpu().detach().numpy().tolist()
    
        epoch_results['Epoch'] = epoch
        epoch_results['training accuracy'] = train_acc1
        epoch_results["val accuracy"] = val_acc1
        results_dict[epoch] = epoch_results
    
    # Saves results in args.results
    results_dict['best'] = best_results
    json.dump(results_dict, results_file)

    # Save checkpoint
    if args.save_fc_model:
        save_checkpoint({
                    'epoch': args.epochs,
                    'arch': args.arch,
                    'state_dict': model.fc.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)


if __name__ == '__main__':
    main()
