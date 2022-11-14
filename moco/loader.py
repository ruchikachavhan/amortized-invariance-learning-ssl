# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torch
import torchvision
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import itertools
import numpy as np
import torchvision.datasets as datasets


# For baseline model
class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class TwoCropsTransform_Hyper:
    """Returns two random crops of one image for each type of augmentation"""

    def __init__(self, dorsal_transform, ventral_transform, default_transform):
        self.dorsal_transform = dorsal_transform
        self.ventral_transform = ventral_transform
        self.def_transform = default_transform

    def __call__(self, x):
        d_im1 = self.dorsal_transform(x)
        d_im2 = self.dorsal_transform(x)
        v_im1 = self.ventral_transform(x)
        v_im2 = self.ventral_transform(x)
        def_im1 = self.def_transform(x)
        def_im2 = self.def_transform(x)
        return [d_im1, d_im2, v_im1, v_im2, def_im1, def_im2]


class FiveCropsTransform_Hyper:
    """Returns two random crops of one image for each comination of valid augmentations
    In this work, we consider 5 augmentations. There are 32 possible combinations. However, we do not consider 
    the combinations where less than two augmentations are applied (following SimCLR protocol)
    """

    def __init__(self, k_aug_list, size, k=5):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.convert_tensors =  [transforms.Resize(size),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor(),
                                    normalize ]
        self.k_aug_list = k_aug_list
        self.indices_list = [list(i) for i in itertools.product([0, 1], repeat=k)] # 2^k Binary vectors, k = 5 here, TODO: remove this hard coding 
        new_list = []
        if k == 5:
            for i in self.indices_list:
                if sum(i) > 1.0: # Exclude augmentations where less than two augmentations are applied 
                    new_list.append(i)
            self.indices_list = new_list
        else:
            for i in self.indices_list:
                if sum(i) == 2.0: # Exclude augmentations where less than two augmentations are applied 
                    new_list.append(i)
            self.indices_list = new_list
        print(self.k_aug_list, self.indices_list, len(self.indices_list))
    
    def __call__(self, x):
    
        outputs = []
       
        for p in range(0, len(self.indices_list)):
            indices = np.where(np.array(self.indices_list[p])==1)[0]
            aug_list = [self.k_aug_list[indices[k]] for k in range(indices.shape[0])]
            augmentation = torchvision.transforms.Compose(aug_list + self.convert_tensors)
            outputs.append(augmentation(x))
            outputs.append(augmentation(x))
        return outputs

class  Measure_inv:
    """Take two random crops of one image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        d_im1 = self.transform


class MeasureInvarianceDataset:
    """Take two random crops of one image"""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.base_transform =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

    def __call__(self, x):
        augmentated_samples = []
        augmentated_samples.append(self.base_transform(x))
        # for i in range(0, 1):
        #     augmentated_samples.append(self.transform1(x))
        # for i in range(0, 1):
        augmentated_samples.append(self.transform2(x))
        return torch.cat(augmentated_samples, dim=0)

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)

		
	