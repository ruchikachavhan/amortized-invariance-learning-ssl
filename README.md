# amortized-invariance-learning-ssl

This is the official implementation of the ICLR 2023 paper - [Amortised Invariance Learning for Contrastive Self-Supervision](https://arxiv.org/abs/2302.12712). 

## Requirements
This code base has been tested with the following package versions:

```
python=3.8.13
torch=1.13.0
torchvision=0.14.0
PIL=7.1.2
numpy=1.22.3
scipy=1.7.3
tqdm=4.31.1
sklearn=1.2.1
wandb=0.13.4
tllib=0.4
```

For pretraining download [ImageNet](https://www.image-net.org) and generate ImageNet-100 using this [repository](https://github.com/danielchyeh/ImageNet-100-Pytorch). 

Make a folder named ```TestDatasets``` to download and process downstream datasets. Below is the outline of expected file structure. 

```
imagenet1k/
imagenet-100/
amortized-invariance-learning-ssl/
    saved_models/
    ...
TestDatasets/
    CIFAR10/
    CIFAR100/
    300w/
    ...
```

## Pre-training

In our paper, we perform pre-training experiments with ResNet50 and ViTs with MoCo-v2 and MoCo-v3 respectively. This pre-training codebase is heavily based on both official implementations of [moco-v2](https://github.com/facebookresearch/moco) and [moco-v3](https://github.com/facebookresearch/moco-v3). 


### ResNet50

We parameterise the ResNet50 backbone in the form of a hypernetwork. To pre-train the hypernetwork on ImageNet-100 with 4-GPUs:

```
python main_moco.py -a resnet50 --lr 0.0005 --weight-decay 2e-5 --moco-t 0.1 --moco-mlp-dim 2048 --moco-dim 128 --warmup-epochs 0 --batch-size 128  --optimizer adamw --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data ../image-net100 
```

Models will be stored in ```saved_models/```

### ViT (base)

We implemented Amortised ViTs using invariances as prompt tokens. First, download the moco-v3 model for initialisation of Prompt-ViT from [this link](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar) . To pre-train the Prompt-ViT on ImageNet with 8-GPUs:
```
 python main_moco.py -a vit_base --lr 1.5e-4 --weight-decay 0.1 --stop-grad-conv1 --moco-t 0.2 --moco-m-cos --moco-mlp-dim 4096 --moco-dim 256  --batch-size 1024  --warmup-epochs=40 --epochs 300 --dist-url 'tcp://localhost:8008' --multiprocessing-distributed --world-size 1 --rank 0 --data ../../imagenet1k
```
Download pretrained models from [here](https://www.dropbox.com/s/aekxafpwpe5kgb3/hyper_resnet50_imagenet100-moco_2_checkpoint_0300.pth.tar?dl=0). 
## Downstream training

We evaluate on several downstream datasets including [CIFAR10](https://pytorch.org/vision/stable/datasets.html), [CIFAR100](https://pytorch.org/vision/stable/datasets.html), [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Oxford-Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), [Oxford-Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [300w](https://ibug.doc.ic.ac.uk/resources/300-W/), [Leeds Sports Pose](https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose_extended.html), and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). We download these datasets in ```../TestDatasets/```. Training and test splits have been adopted from the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/tree/master)

To run downstream experiments for amortised models, for example on CIFAR10 on GPU 0, run 
```
python main_lincls.py -a <arch> --test_dataset cifar10 --gpu 0 --pretrained saved_models/<name of checkpoint> 
```
To run baseline models, run
```
python main_lincls.py -a <arch> --test_dataset cifar10 --gpu 0 --baseline --pretrained saved_models/<name of baseline checkpoint> 
```
Results will be stored in ```results/```


If you find our work helpful, please cite our paper
```
@inproceedings{
chavhan2023amortised,
title={Amortised Invariance Learning for Contrastive Self-Supervision},
author={Ruchika Chavhan and Henry Gouk and Jan Stuehmer and Calum Heggan and Mehrdad Yaghoobi and Timothy Hospedales},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=nXOhmfFu5n}
}
```




