#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:18:44 2022

@author: bogdan
"""

# %% import modules
import imageio
import numpy as np
import pandas as pd
import json
import glob
import cv2
import os
import math
from torchvision import datasets, transforms

# %% PBR_train path
parent_dir = os.getcwd()
ori_path = parent_dir + '/original data/lmo/'
pbr_save_dir = parent_dir + '/processed data/lmo/pbr/'

obj_range = [1, 5, 6, 8, 9, 10, 11, 12]

pbr_camera = json.load(open(ori_path + '/camera.json'))

cx, cy, fx, fy = (pbr_camera['cx'], pbr_camera['cy'],
                  pbr_camera['fx'], pbr_camera['fy'])

# %% extract pbr and save into new folders
for pbr_scene in range(50):

    ori_pbr = (ori_path + '/train_pbr/' + str(pbr_scene).zfill(6))

    scene_gt_info_path = ori_pbr + '/scene_gt_info.json'
    scene_gt_path = ori_pbr + '/scene_gt.json'
    scene_gt_info = json.load(open(scene_gt_info_path))
    scene_gt = json.load(open(scene_gt_path))

    name = []
    ob_id = []
    rotation = []
    translation = []
    bbox = []
    obj_cx = []
    obj_cy = []

    for imageid in scene_gt_info:

        object_gt = scene_gt[imageid]
        object_gt_info = scene_gt_info[imageid]

        rgb = cv2.cvtColor(
            cv2.imread(ori_pbr + '/rgb/' + imageid.zfill(6) + '.jpg'),
            cv2.COLOR_BGR2RGB)

        for column in range(len(object_gt_info)):
            obj_id = object_gt[column]["obj_id"]

            if obj_id in obj_range:
                visib_fract = object_gt_info[column]["visib_fract"]

                if visib_fract >= 0.1:
                    cam_R_m2c = object_gt[column]["cam_R_m2c"]
                    cam_t_m2c = object_gt[column]["cam_t_m2c"]
                    object_cx = cam_t_m2c[0]*fx/cam_t_m2c[2]+cx
                    object_cy = cam_t_m2c[1]*fy/cam_t_m2c[2]+cy

                    bbox_obj = object_gt_info[column]["bbox_visib"]
                    x, y, w, h = bbox_obj

                    if w > h:
                        new_x = x
                        new_y = math.floor(y-(w-h)/2)
                        new_w = w
                        new_h = w
                    else:
                        new_x = math.floor(x-(h-w)/2)
                        new_y = y
                        new_w = h
                        new_h = h

                    left_trunc = np.maximum(new_x, 0)
                    right_trunc = np.minimum(new_x + new_w, rgb.shape[1])
                    top_trunc = np.maximum(new_y, 0)
                    bottom_trunc = np.minimum(new_y + new_h, rgb.shape[0])

                    rgb_ROI = rgb[top_trunc:bottom_trunc,
                                  left_trunc:right_trunc]

                    rgb_resized = cv2.resize(rgb_ROI, (128, 128), interpolation=cv2.INTER_CUBIC)

                    save_path = os.path.join(pbr_save_dir, str(obj_id).zfill(6), str(pbr_scene).zfill(6))
                    if os.path.exists(save_path) is False:
                        os.makedirs(save_path)

                    rgb_name = '%06i_%06i_%s.png' % (obj_id, pbr_scene, imageid.zfill(6))
                    rgb_dir = os.path.join(save_path, rgb_name)
                    imageio.imwrite(rgb_dir, rgb_resized)
                    print(rgb_dir)

                    name.append(rgb_name)
                    rotation.append(cam_R_m2c)
                    translation.append(cam_t_m2c)
                    ob_id.append(obj_id)
                    bbox.append(bbox_obj)
                    obj_cx.append(object_cx)
                    obj_cy.append(object_cy)

    csvfile = {'filename': name, 'r': rotation, 't': translation, 'bbox': bbox,
               'id': ob_id, 'cen_x': obj_cx, 'cen_y': obj_cy}

    df = pd.DataFrame(csvfile)
    df.set_index("id", inplace=True)
    df.head()

    for obidx in obj_range:
        subset = df.loc[obidx]
        subset = subset.to_dict(orient='records')

        json_name = '%06i.json' % (pbr_scene)
        IDobject = str(obidx).zfill(6)
        json_path = os.path.join(pbr_save_dir, IDobject, json_name)

        json_file = open(json_path, "w")
        json.dump(subset, json_file)

# %% Check if there are any errors
for i in obj_range:
    obidx = str(i).zfill(6)
    GT_Train_path = pbr_save_dir + obidx + '/'
    GT_Train_jsons = sorted(glob.glob(GT_Train_path + "*.json"))

    json_train = []
    for sceneid in range(50):
        temp = json.load(open(GT_Train_jsons[sceneid]))
        json_train.extend(temp)

    train_dataset = datasets.ImageFolder(pbr_save_dir + obidx,
                                         transform=transforms.ToTensor())

    if len(train_dataset) == len(json_train):
        json_name = '%s_all.json' % (obidx)
        json_path = os.path.join(pbr_save_dir, json_name)
        json_file = open(json_path, "w")
        json.dump(json_train, json_file)
    else:
        print('error')
