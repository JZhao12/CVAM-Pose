#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:11:02 2024

@author: jianyu
"""

# %% import modules
import os
import torch
import glob
import multiprocessing
import random
import numpy as np
import torch.nn.functional as F
from models import resnet_cvae
from torch.utils.data import DataLoader, random_split
from engine import DenseDataset, loss

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# %% path, data information

device = torch.device("cuda:0")
parent_dir = os.getcwd()
model_path = parent_dir + '/results/lmo/cvae/'
if os.path.exists(model_path) is False:
    os.makedirs(model_path)

model_name = 'cvae.pth'

# %% train CVAE


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


object_number = [1, 5, 6, 8, 9, 10, 11, 12]
class_number = 0

labels=[]

pbr=[]
pbr_recon=[]

for number in object_number:

    ob_id = '%06i' % (number)
    
    pbr_path = parent_dir + '/processed data/lmo/pbr/' + ob_id
    recon_path = parent_dir + '/processed data/lmo/pbr_recon/' + ob_id

    new_pbr = sorted(glob.glob(pbr_path + "/**/*.png", recursive=True))

    pbr += new_pbr
    pbr_recon += sorted(glob.glob(recon_path + "/**/*.png", recursive=True))

    labels += [class_number for _ in range(len(new_pbr))]
    class_number += 1


whole_data = DenseDataset(pbr, pbr_recon, labels, len(object_number))

a = int(len(pbr)*0.9)
b = len(pbr) - a
lengths = [a, b]
train, valid = random_split(whole_data, lengths, generator=g)


train_dataloader = DataLoader(train, batch_size=128,
                              num_workers=multiprocessing.Pool()._processes,
                              worker_init_fn=seed_worker,
                              generator=g, pin_memory=True, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size=128,
                              num_workers=multiprocessing.Pool()._processes,
                              worker_init_fn=seed_worker,
                              generator=g, pin_memory=True, shuffle=False)

network = resnet_cvae(num_labels=8)

num_epochs = 1000
optimizer = torch.optim.AdamW(params=network.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=50, threshold=0.0001,
    threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)

min_valid_loss = np.inf
network = network.to(device)

for epoch in range(num_epochs):
    train_loss = 0.0
    network.train()
    for input_image, gt_recon, labels in train_dataloader:
        optimizer.zero_grad()
        input_image = input_image.to(device)
        gt_recon = gt_recon.to(device)
        labels = labels.to(device)

        x_recon, latent_mu, latent_logvar = network(input_image, labels)
        t_loss = loss(x_recon, gt_recon, latent_mu, latent_logvar)
        t_loss.backward()
        optimizer.step()
        train_loss += t_loss.item()

    valid_loss = 0.0
    network.eval()
    with torch.no_grad():
        for input_image, gt_recon, labels in valid_dataloader:

            input_image = input_image.to(device)
            gt_recon = gt_recon.to(device)
            labels = labels.to(device)

            x_recon, _, _ = network(input_image, labels)
            v_loss = F.mse_loss(x_recon.view(-1, 128*128), gt_recon.view(-1, 128*128), reduction='sum')
            valid_loss += v_loss.item()

    scheduler.step(valid_loss)
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(valid_dataloader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        epochs_no_improve = 0
        torch.save(network.state_dict(), model_path + model_name)
    elif (min_valid_loss < valid_loss and
          optimizer.param_groups[0]['lr'] == 1e-6):
        epochs_no_improve += 1

    if epochs_no_improve == 50:
        print('Finish Training')
        break
