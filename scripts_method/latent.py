#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:13:53 2024

@author: jianyu
"""

# %% import modules
import torch
import os
import numpy as np
import json
import glob
import random
import multiprocessing
from models import resnet_cvae
from torch.utils.data import DataLoader, random_split
from engine import latent_var, DenseDataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, argparse, data information

parent_dir = os.getcwd()
model_path = parent_dir + '/results/lmo/cvae/'
latent_path = parent_dir + '/results/lmo/latent/'
if os.path.exists(latent_path) is False:
    os.makedirs(latent_path)

model_name = 'cvae.pth'
device = torch.device("cuda:0")

# %% generate latent variables

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
pbr_info = []
for number in object_number:

    ob_id = '%06i' % (number)
    
    pbr_path = parent_dir + '/processed data/lmo/pbr/' + ob_id
    recon_path = parent_dir + '/processed data/lmo/pbr_recon/' + ob_id
    
    new_pbr=sorted(glob.glob(pbr_path + "/**/*.png", recursive=True))
    
    single_info = json.load(open(pbr_path + '_all.json'))
    pbr_info += single_info

    pbr += new_pbr
    pbr_recon += sorted(glob.glob(recon_path + "/**/*.png", recursive=True))

    labels+=[class_number for _ in range(len(new_pbr))]
    class_number+=1


whole_data = DenseDataset(pbr, pbr_recon, labels, len(object_number))

a = int(len(pbr)*0.9)
b = len(pbr) - a
lengths = [a, b]
train, valid = random_split(whole_data, lengths, generator=g)


pbr_loader = DataLoader(whole_data, batch_size=512,
                        num_workers=multiprocessing.Pool()._processes,
                        worker_init_fn=seed_worker,
                        generator=g, pin_memory=True, shuffle=False)

network = resnet_cvae(num_labels=8)

network.load_state_dict(torch.load(model_path + model_name, map_location='cuda:0'))
network = network.to(device)

pbr_mean, pbr_std, pbr_label = latent_var(network, pbr_loader, device)

rotation = [d['r'] for d in pbr_info]
translation = [d['t'] for d in pbr_info]
bbox = [d['bbox'] for d in pbr_info]
obj_cx = [d['cen_x'] for d in pbr_info]
obj_cy = [d['cen_y'] for d in pbr_info]

rotation = np.array(rotation, dtype='float32')
translation = np.array(translation, dtype='float32')
bbox = np.array(bbox, dtype='float32')
obj_cx = np.expand_dims(np.array(obj_cx, dtype='float32'), axis=1)
obj_cy = np.expand_dims(np.array(obj_cy, dtype='float32'), axis=1)


train_idx = train.indices
train_mean = pbr_mean[train_idx, :]
train_std = pbr_std[train_idx, :]
train_label = pbr_label[train_idx, :]

train_r = rotation[train_idx, :]
train_t = translation[train_idx, :]
train_bbox = bbox[train_idx, :]
train_objcx = obj_cx[train_idx, :]
train_objcy = obj_cy[train_idx, :]

train_save = latent_path + 'train'
np.savez(train_save, mean=train_mean, std=train_std, label=train_label,
         r=train_r, t=train_t, bbox=train_bbox,
         cen_x=train_objcx, cen_y=train_objcy)

valid_idx = valid.indices
valid_mean = pbr_mean[valid_idx, :]
valid_std = pbr_std[valid_idx, :]
valid_label = pbr_label[valid_idx, :]

valid_r = rotation[valid_idx, :]
valid_t = translation[valid_idx, :]
valid_bbox = bbox[valid_idx, :]
valid_objcx = obj_cx[valid_idx, :]
valid_objcy = obj_cy[valid_idx, :]

valid_save = latent_path + 'valid'
np.savez(valid_save, mean=valid_mean, std=valid_std, label=valid_label,
         r=valid_r, t=valid_t, bbox=valid_bbox,
         cen_x=valid_objcx, cen_y=valid_objcy)

# %% test latent

class_number = 0
for number in object_number:

    ob_id = '%06i' % (number)

    test_path = parent_dir + '/processed data/lmo/test_bop/' + ob_id
    test_save = latent_path + 'test_' + ob_id

    test_fold = sorted(glob.glob(test_path + "/**/*.png", recursive=True))
    labels = [class_number for _ in range(len(test_fold))]

    test_set = DenseDataset(test_fold, test_fold, labels, len(object_number))

    test_loader = DataLoader(test_set, batch_size=128,
                             num_workers=multiprocessing.Pool()._processes,
                             worker_init_fn=seed_worker,
                             generator=g, pin_memory=True, shuffle=False)

    test_mean, test_std, test_label = latent_var(network, test_loader, device)
    
    test_json = json.load(open(test_path + '_all.json'))
    
    test_r = [d['r'] for d in test_json]
    test_t = [d['t'] for d in test_json]
    test_bbox = [d['bbox'] for d in test_json]
    
    test_r = np.array(test_r, dtype='float32')
    test_t = np.array(test_t, dtype='float32')
    test_bbox = np.array(test_bbox, dtype='float32')

    np.savez(test_save, mean=test_mean, std=test_std, label=test_label,
             r=test_r, t=test_t, bbox=test_bbox)
    
    class_number+=1
