#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:24:41 2024

@author: jianyu
"""

# %% import modules
import multiprocessing
import torch
import os
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models import depth_mlp
from sklearn.metrics import mean_absolute_error

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, argparse, data information

device = torch.device("cuda:0")

parent_dir = os.getcwd()
latent_path = parent_dir + '/results/lmo/latent/'
mlp_path = parent_dir + '/results/lmo/mlp/'
if os.path.exists(mlp_path) is False:
    os.makedirs(mlp_path)

mlp_name = 'depth.pth'

# %% data
train_save = latent_path + 'train.npz'
valid_save = latent_path + 'valid.npz'

with np.load(train_save) as X:
    train_mean = X['mean']
    train_label = X['label']
    train_t = X['t']
    train_bbox = X['bbox']

with np.load(valid_save) as X:
    valid_mean = X['mean']
    valid_label = X['label']
    valid_t = X['t']
    valid_bbox = X['bbox']

train_geo = train_bbox[:, 2:4]
train_depth = np.expand_dims(train_t[:, 2], axis=1)

valid_geo = valid_bbox[:, 2:4]
valid_depth = np.expand_dims(valid_t[:, 2], axis=1)

train_mean = torch.tensor(train_mean)
train_geo = torch.tensor(train_geo)
train_label = torch.tensor(train_label)
train_depth = torch.tensor(train_depth)

valid_mean = torch.tensor(valid_mean)
valid_geo = torch.tensor(valid_geo)
valid_label = torch.tensor(valid_label)
valid_depth = torch.tensor(valid_depth)

mlp_train_data = []
for i in range(len(train_mean)):
    mlp_train_data.append([train_mean[i], train_geo[i], train_label[i], train_depth[i]])

mlp_valid_data = []
for i in range(len(valid_mean)):
    mlp_valid_data.append([valid_mean[i], valid_geo[i], valid_label[i], valid_depth[i]])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

mlp_train_data = DataLoader(mlp_train_data, batch_size=len(mlp_train_data),
                            num_workers=multiprocessing.Pool()._processes,
                            worker_init_fn=seed_worker,
                            generator=g, pin_memory=True, shuffle=True)

mlp_valid_data = DataLoader(mlp_valid_data, batch_size=len(mlp_valid_data),
                            num_workers=multiprocessing.Pool()._processes,
                            worker_init_fn=seed_worker,
                            generator=g, pin_memory=True, shuffle=False)

device = torch.device("cuda:0")
mlp = depth_mlp(geo_info=2, num_labels=8)
mlp = mlp.to(device)

num_epochs = 100000
# num_epochs = 1

loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(params=mlp.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=500, threshold=0.0001,
    threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)
min_valid_loss = np.inf

train_mode = True
# train_mode = False
if train_mode:

    for epoch in range(num_epochs):
    
        train_loss = 0.0
        mlp.train()
    
        for inputs, bb, labels, targets in mlp_train_data:
    
            optimizer.zero_grad()
            inputs = inputs.to(device)
            bb = bb.to(device)
            labels = labels.to(device)
            targets = targets.to(device)
    
            output = mlp(inputs, bb, labels)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        valid_loss = 0.0
        mlp.eval()
        with torch.no_grad():
    
            for inputs, bb, labels, targets in mlp_valid_data:
    
                inputs = inputs.to(device)
                bb = bb.to(device)
                labels = labels.to(device)
                targets = targets.to(device)
    
                output = mlp(inputs, bb, labels)
                loss = loss_func(output, targets)
                valid_loss += loss.item()
    
        scheduler.step(valid_loss)
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(mlp_train_data)} \t\t Validation Loss: {valid_loss / len(mlp_valid_data)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(mlp.state_dict(), mlp_path + mlp_name)
        elif (min_valid_loss < valid_loss and
              optimizer.param_groups[0]['lr'] == 1e-6):
            epochs_no_improve += 1
    
        if epochs_no_improve == 1000:
            print('Finish Training')
            break

# %% for test
result_path = parent_dir + '/results/lmo/pose/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

mlp.load_state_dict(torch.load(mlp_path + mlp_name))

object_number = [1, 5, 6, 8, 9, 10, 11, 12]

for number in object_number:
    ob_id = '%06i' % (number)

    results_save = result_path + ob_id + '_depth'
    test_save = latent_path + 'test_' + ob_id + '.npz'
    with np.load(test_save) as X:
        test_mean = X['mean']
        test_bbox = X['bbox']
        test_label = X['label']

    test_geo = test_bbox[:, 2:4]

    test_mean = torch.tensor(test_mean)
    test_geo = torch.tensor(test_geo)
    test_label = torch.tensor(test_label)

    mlp_test_data = []
    for i in range(len(test_mean)):
        mlp_test_data.append([test_mean[i], test_geo[i], test_label[i]])

    x_test = DataLoader(mlp_test_data,
                        batch_size=len(mlp_test_data), shuffle=False)

    prediction = []
    mlp.eval()
    for inputs, bb, labels in x_test:
        with torch.no_grad():
            inputs = inputs.to(device)
            bb = bb.to(device)
            labels = labels.to(device)

            output = mlp(inputs, bb, labels)
            prediction.append(output)

    recover_tz = torch.cat(prediction).cpu().numpy()

    # recover_tz = np.expand_dims(np.concatenate(recover_tz, axis=0), axis=1)
    np.savez(results_save, depth=recover_tz)
