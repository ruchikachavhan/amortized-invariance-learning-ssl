# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, baseline, arch, dim=256, K=65536, m=0.999, T=0.07, mlp_dim=2048):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.K = K
        self.m = m

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.baseline = baseline
        self.arch = arch

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        if self.baseline:
            self.register_buffer("queue", torch.randn(dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        else:
            self.register_buffer("queue_dorsal", torch.randn(dim, K))
            self.queue_dorsal = nn.functional.normalize(self.queue_dorsal, dim=0)
            self.register_buffer("queue_ptr_dorsal", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_ventral", torch.randn(dim, K))
            self.queue_ventral = nn.functional.normalize(self.queue_ventral, dim=0)
            self.register_buffer("queue_ptr_ventral", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_default", torch.randn(dim, K))
            self.queue_default = nn.functional.normalize(self.queue_default, dim=0)
            self.register_buffer("queue_ptr_default", torch.zeros(1, dtype=torch.long))

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=True))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    # Copied from simclr code
    def contrastive_loss(self, q, k):
        # normalize
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long).cuda() + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T), logits, labels


    @torch.no_grad()
    def _dequeue_and_enqueue_dorsal(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_dorsal)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_dorsal[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_dorsal[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_ventral(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_ventral)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_ventral[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_ventral[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_default(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_default)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_default[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_default[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_baseline(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, x1, x2, m, simclr_train=False, inv=None, inv_index=None):
        """
        Input:
            x1: first views of images : Query
            x2: second views of images : key
            m: moco momentum
        Output:
            loss
        """

        # compute features
        if self.baseline:
            q = self.base_encoder(x1)
            q = nn.functional.normalize(q, dim=1)
            # q2 = self.predictor(self.base_encoder(x2, inv))

            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder
                if not simclr_train:
                    x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

                # compute momentum features as targets
                k = self.momentum_encoder(x2)
                k = nn.functional.normalize(k, dim=1)
                if not simclr_train:
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        else:
            # This is q1 in original moco paper
            q = self.predictor(self.base_encoder(x1, inv))
            q = nn.functional.normalize(q, dim=1)

            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder
                if not simclr_train:
                    x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

                # compute momentum features as targets
                # This is k2 in original moco paper
                k = self.momentum_encoder(x2, inv)
                k = nn.functional.normalize(k, dim=1)
                if not simclr_train: # Only shuffle batches is training for moco
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        if simclr_train: # Contrastive loss for simclr, Simclr not supported for ViT
            loss, logits, labels = self.contrastive_loss(q, k)
            return loss, logits, labels
        else: # Contrastive loss for queue for moco
            if self.arch == 'vit_base':
                # Computing q2 and k1 according to moco-v3
                # k (here) corresponds to k2 and k2 (here) corresponds to k1 from moco-v3: Change this later for less confusion
                q2 = self.predictor(self.base_encoder(x2, inv))
                k2 = self.momentum_encoder(x1, inv)
                # contrastive loss for (q1, k2) + (q2, k1)
                return self.contrastive_loss(q, k) + self.contrastive_loss(q2, k2)
            else:
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                
                # negative logits: NxK
                if self.baseline:
                    l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                else: # Queue supported only for dorsal, ventral and default
                    if inv_index == 0:
                        l_neg = torch.einsum('nc,ck->nk', [q, self.queue_dorsal.clone().detach()])
                    elif inv_index == 1:
                        l_neg = torch.einsum('nc,ck->nk', [q, self.queue_ventral.clone().detach()])
                    elif inv_index == 2:
                        l_neg = torch.einsum('nc,ck->nk', [q, self.queue_default.clone().detach()])

                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                # dequeue and enqueue
                if self.baseline:
                    self._dequeue_and_enqueue_baseline(k)
                else: # Different queue for different invariances
                    if inv_index==0:
                        self._dequeue_and_enqueue_dorsal(k)
                    elif inv_index == 1:
                        self._dequeue_and_enqueue_ventral(k)
                    elif inv_index ==2:
                        self._dequeue_and_enqueue_default(k)
                loss =  nn.CrossEntropyLoss()(logits, labels)
            
                return loss, logits, labels

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer
        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        # self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        self.predictor =  nn.Identity()


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
