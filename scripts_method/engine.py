#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:12:43 2024

@author: jianyu
"""

# %% import modules
import torch
import torch.nn.functional as F
from torchvision import transforms
from imageio import imread
from torch.utils.data import Dataset
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# %% merge images and labels
class DenseDataset(Dataset):
    def __init__(self, data, recon, labels, num_classes):
        self.input = data
        self.target = recon
        self.labels = labels
        self.Trans = transforms.ToTensor()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index: int):
        input_ID = self.input[index]
        target_ID = self.target[index]
        label = self.labels[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)
        x, y = self.Trans(x), self.Trans(y)
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return x, y, label.float()


# %% loss for training the autoencoders
def loss(recon_x, gt_recon, mu, logvar):
    recon_loss = F.mse_loss(recon_x.view(-1, 128*128),
                            gt_recon.view(-1, 128*128), reduction='none')
    l2_val, _ = torch.topk(recon_loss, k=recon_loss.shape[1]//4)
    L2loss = torch.sum(l2_val)
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L2loss + 0.1 * kldivergence


# %% generate the latent variables
def latent_var(model, dataloader, device):
    model.eval()
    mean = []
    std = []
    label = []
    for input_image, _, labels in dataloader:
        with torch.no_grad():
            input_image = input_image.to(device)
            labels = labels.to(device)
            _, latent_mu, logvar = model(input_image, labels)
            mean.append(latent_mu)
            std.append(logvar.exp())
            label.append(labels)

    mean = torch.cat(mean).cpu().numpy()
    std = torch.cat(std).cpu().numpy()
    label = torch.cat(label).cpu().numpy()
    return mean, std, label


# %% transform between 6d and matrix
"""

The transformation code comes from pytorch3D
https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html

BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)