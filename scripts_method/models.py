#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:17:25 2024

@author: jianyu
"""

# %% module
import torch
import torch.nn as nn
from torch.nn import functional as F


# %% resnet encoder
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=5, padding=2, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=5, padding=2)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = self.bn1(F.silu(self.conv1(X)))
        Y = self.bn2(F.silu(self.conv2(Y)))
        if self.conv3:
            X = F.silu(self.conv3(X))
        Y += X
        return Y


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class fb_residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=5, padding=2, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = self.bn1(F.silu(self.conv1(X)))
        Y = self.bn2(F.silu(self.conv2(Y)))
        recover_x = torch.split(X, [128, 8], dim=1)[0]
        Y += recover_x
        return Y


def first_block(input_channels, num_channels, num_residuals):
    blk = []
    blk.append(fb_residual(input_channels, num_channels))
    blk.append(Residual(num_channels, num_channels))
    return blk


class resnet_encoder(nn.Module):
    def __init__(self, num_labels):
        super(resnet_encoder, self).__init__()
        self.num_labels = num_labels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + num_labels, 128, kernel_size=5, stride=2, padding=2),
            nn.SiLU(), nn.BatchNorm2d(128))

        self.conv2 = nn.Sequential(*first_block(128 + num_labels, 128, 2))
        self.conv3 = nn.Sequential(*resnet_block(128 + num_labels, 256, 2))
        self.conv4 = nn.Sequential(*resnet_block(256 + num_labels, 256, 2))
        self.conv5 = nn.Sequential(*resnet_block(256 + num_labels, 512, 2))
        self.conv6 = nn.Sequential(*resnet_block(512 + num_labels, 512, 2),
                                   nn.Flatten())

        self.inter = nn.Sequential(nn.Linear(512*4*4 + num_labels, 1024), nn.SiLU())
        self.fc_mu = nn.Linear(1024 + num_labels, 256)
        self.fc_logvar = nn.Linear(1024 + num_labels, 256)

    def forward(self, x, labels):
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 128, 128)), dim=1)
        x = self.conv1(x)
        
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)), dim=1)
        x = self.conv2(x)
        
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)), dim=1)
        x = self.conv3(x)
        
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 32, 32)), dim=1)
        x = self.conv4(x)
        
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 16, 16)), dim=1)
        x = self.conv5(x)
        
        x = torch.cat((x, labels.unsqueeze(2).unsqueeze(3).expand(-1, -1, 8, 8)), dim=1)
        x = self.conv6(x)
        
        x = torch.cat((x, labels), 1)
        x = self.inter(x)
        
        x = torch.cat((x, labels), 1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


# # Example usage
# network = resnet_encoder(num_labels=8)
# images = torch.randn(32, 3, 128, 128)
# labels = torch.zeros(32, 8).scatter_(1, torch.randint(0, 8, (32, 1)), 1)
# mu, logvar = network(images, labels)


# %% decoder

class Decoder(nn.Module):
    def __init__(self, num_labels):
        super(Decoder, self).__init__()
        self.dense = nn.Sequential(nn.Linear(256 + num_labels, 1024), nn.SiLU())
        
        self.inter = nn.Sequential(
            nn.Linear(1024 + num_labels, 512*4*4), nn.SiLU(),
            nn.Unflatten(1, (512, 4, 4)), nn.BatchNorm2d(512))

        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + num_labels, 512, kernel_size=5, stride=1, padding=2),
            nn.SiLU(), nn.BatchNorm2d(512))

        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + num_labels, 256, kernel_size=5, stride=1, padding=2),
            nn.SiLU(), nn.BatchNorm2d(256))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + num_labels, 256, kernel_size=5, stride=1, padding=2),
            nn.SiLU(), nn.BatchNorm2d(256))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256 + num_labels, 128, kernel_size=5, stride=1, padding=2),
            nn.SiLU(), nn.BatchNorm2d(128))

        self.conv5 = nn.Sequential(
            nn.Conv2d(128 + num_labels, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid())

    def forward(self, x, labels):
        x = torch.cat((x, labels), 1)
        x = self.dense(x)
        
        x = torch.cat((x, labels), 1)
        x = self.inter(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        x = torch.cat((x,labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,8,8)),dim=1)
        x = self.conv1(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        x = torch.cat((x,labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,16,16)),dim=1)
        x = self.conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        x = torch.cat((x,labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,32,32)),dim=1)
        x = self.conv3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        x = torch.cat((x,labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,64,64)),dim=1)
        x = self.conv4(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        x = torch.cat((x,labels.unsqueeze(2).unsqueeze(3).expand(-1,-1,128,128)),dim=1)
        x = self.conv5(x)
        return x

# # Example usage
# network = Decoder(num_labels=8)
# latent_vectors = torch.randn(32, 128)
# labels = torch.zeros(32, 8).scatter_(1, torch.randint(0, 8, (32, 1)), 1)
# output = network(latent_vectors, labels)


# %% resnet autoencoder
class resnet_cvae(nn.Module):
    def __init__(self, num_labels):
        super(resnet_cvae, self).__init__()
        self.encoder = resnet_encoder(num_labels)
        self.decoder = Decoder(num_labels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, labels)
        return x_recon, mu, logvar

# # Example usage
# network = resnet_cvae(num_labels=8)
# images = torch.randn(32, 3, 128, 128)
# labels = torch.zeros(32, 8).scatter_(1, torch.randint(0, 8, (32, 1)), 1)
# recon, mu, logvar = network(images, labels)


# %% MLP

class rotation_mlp(nn.Module):
    def __init__(self, num_labels):
        super(rotation_mlp, self).__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Sequential(nn.Linear(256 + num_labels, 128), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(128 + num_labels, 64), nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(64 + num_labels, 32), nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(32 + num_labels, 16), nn.SiLU())
        self.fc5 = nn.Linear(16 + num_labels, 6)

    def forward(self, x, labels):
        x = torch.cat((x, labels), 1)
        x = self.fc1(x)
        
        x = torch.cat((x, labels), 1)
        x = self.fc2(x)
        
        x = torch.cat((x, labels), 1)
        x = self.fc3(x)
        
        x = torch.cat((x, labels), 1)
        x = self.fc4(x)

        x = torch.cat((x, labels), 1)
        x = self.fc5(x)
        return x

# # Example usage
# network = rotation_mlp(num_labels=8)
# images = torch.randn(32, 128)
# labels = torch.zeros(32, 8).scatter_(1, torch.randint(0, 8, (32, 1)), 1)
# x = network(images, labels)


class centre_mlp(nn.Module):
    def __init__(self, geo_info, num_labels):
        super(centre_mlp, self).__init__()
        self.geo_info = geo_info
        self.num_labels = num_labels
        self.fc1 = nn.Sequential(nn.Linear(256 + geo_info + num_labels, 128), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(128 + geo_info + num_labels, 64), nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(64 + geo_info + num_labels, 32), nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(32 + geo_info + num_labels, 16), nn.SiLU())
        self.fc5 = nn.Sequential(nn.Linear(16 + geo_info + num_labels, 8), nn.SiLU())
        self.fc6 = nn.Sequential(nn.Linear(8 + geo_info + num_labels, 4), nn.SiLU())
        self.fc7 = nn.Linear(4 + geo_info + num_labels, 2)

    def forward(self, x, geo_info, labels):
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc1(x)

        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc2(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc3(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc4(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc5(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc6(x)

        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc7(x)
        return x


class depth_mlp(nn.Module):
    def __init__(self, geo_info, num_labels):
        super(depth_mlp, self).__init__()
        self.num_labels = num_labels
        self.geo_info = geo_info
        self.fc1 = nn.Sequential(nn.Linear(256 + geo_info + num_labels, 128), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(128 + geo_info + num_labels, 64), nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(64 + geo_info + num_labels, 32), nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(32 + geo_info + num_labels, 16), nn.SiLU())
        self.fc5 = nn.Sequential(nn.Linear(16 + geo_info + num_labels, 8), nn.SiLU())
        self.fc6 = nn.Sequential(nn.Linear(8 + geo_info + num_labels, 4), nn.SiLU())
        self.fc7 = nn.Linear(4 + geo_info + num_labels, 1)

    def forward(self, x, geo_info, labels):
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc1(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc2(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc3(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc4(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc5(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc6(x)
        
        x = torch.cat((x, geo_info, labels), 1)
        x = self.fc7(x)
        return x





