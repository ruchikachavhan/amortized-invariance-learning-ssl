import itertools
import torch
import os

def get_invariances(k_augs, baseline):
    if baseline:
        invariances = None
    elif k_augs == 5:
        inv_list = [list(i) for i in itertools.product([0.0, 1.0], repeat=5)]
        new_list = []
        for i in inv_list:
            if sum(i)> 1.0:
                new_list.append(i)

        invariances = [torch.tensor(new_list[k]) for k in range(0, len(new_list))]
    elif k_augs == 2:
        invariances = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])]
    return invariances

def get_file_name(args):
    if args.baseline:
        fname = "imagenet-moco_" + args.arch+"_"+args.base_augs+'_checkpoint_%04d.pth.tar'
    else:
        if args.arch == 'resnet50':
            fname = "hyper_{}_imagenet100-moco_{}".format(args.arch, args.k_augs) + '_checkpoint_%04d.pth.tar'
        elif args.arch == 'vit_base':
            fname = "prompt_{}_imagenet1k-moco_{}".format(args.arch, args.k_augs) + '_checkpoint_%04d.pth.tar'
    return fname

def check_expt_configs(args):
    if args.k_augs:
        assert args.k_augs == args.simclr_train
    if args.auto_augment:
        assert args.auto_augment == args.simclr_train
    if args.simclr_train:
        if not os.path.isdir(os.path.join(args.output_dir, 'simclr')):
            os.mkdir(os.path.join(args.output_dir, 'simclr'))
        args.output_dir = os.path.join(args.output_dir, 'simclr')
    else: # Moco training
        if not os.path.isdir(os.path.join(args.output_dir, 'moco')):
            os.mkdir(os.path.join(args.output_dir, 'moco'))
        args.output_dir = os.path.join(args.output_dir, 'moco') 
    if args.simclr_train:
        args.moco_m = 0.0
    return args