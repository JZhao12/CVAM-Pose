#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:25:07 2024

@author: jianyu
"""

# %% import modules
import torch
import os
import random
import json
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, argparse, data information
device = torch.device("cuda:0")

parent_dir = os.getcwd()
result_path = parent_dir + '/results/lmo/pose/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

# %% Camera
test_camera = json.load(open(parent_dir + '/original data/lmo/camera.json'))
cx, cy, fx, fy = (test_camera['cx'], test_camera['cy'],
                  test_camera['fx'], test_camera['fy'])

# %% cal

object_number = [1, 5, 6, 8, 9, 10, 11, 12]

for number in object_number:

    ob_id = '%06i' % (number)

    results_save = result_path + ob_id + '_translation'

    centre_results = result_path + ob_id + '_centre.npz'
    with np.load(centre_results) as X:
        pre_objcx = X['ocx']
        pre_objcy = X['ocy']

    depth_results = result_path + ob_id + '_depth.npz'
    with np.load(depth_results) as X:
        recover_tz = X['depth']

    recover_tx = []
    recover_ty = []
    for i in range(len(recover_tz)):
        Tx = (pre_objcx[i]-cx)*recover_tz[i]/fx
        Ty = (pre_objcy[i]-cy)*recover_tz[i]/fy
        recover_tx.append(Tx)
        recover_ty.append(Ty)
    recover_tx = np.expand_dims(np.concatenate(recover_tx, axis=0), axis=1)
    recover_ty = np.expand_dims(np.concatenate(recover_ty, axis=0), axis=1)

    pre_t = np.concatenate((recover_tx, recover_ty, recover_tz), axis=1)

    np.savez(results_save, t=pre_t)
