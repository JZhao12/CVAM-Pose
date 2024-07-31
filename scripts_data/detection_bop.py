#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:06:27 2022

@author: bogdan
"""

# %% import modules
import imageio
import json
import cv2
import os
import math
import numpy as np
import pandas as pd
# %% test path

parent_dir = os.getcwd()

ori_path = parent_dir + '/original data/lmo/'

test_targets = json.load(open(ori_path + '/test_targets_bop19.json'))
test_path = ori_path + '/test/'
test_save_dir = parent_dir + '/processed data/lmo/test_bop/'

detection_dir = parent_dir + '/original data/bop22_default_detections_and_segmentations/cosypose_maskrcnn_pbr/'
detection_name = 'challenge2022-524061_lmo-test.json'
obj_range = [1, 5, 6, 8, 9, 10, 11, 12]

detection_results = json.load(open(detection_dir + detection_name))

# %% extract test results and save into new folders

name = []
ob_id = []
rotation = []
translation = []
bbox = []
vis_fra = []

for test_instance in test_targets:
    scene_id = test_instance['scene_id']
    im_id = test_instance['im_id']
    obj_id = test_instance['obj_id']
    
    candidates = []
    
    for detected_instance in detection_results:
        
        sce_id = detected_instance['scene_id']
        image_id = detected_instance['image_id']
        category_id = detected_instance['category_id']
        if sce_id == scene_id and image_id == im_id and category_id == obj_id and detected_instance['score'] > 0.9:
            candidates.append(detected_instance)

    if len(candidates) > 0:
        confirmed_instance = candidates[0]
        rgb = cv2.cvtColor(cv2.imread(test_path + str(scene_id).zfill(6) + '/rgb/' + str(im_id).zfill(6) + '.png'), cv2.COLOR_BGR2RGB)
        scene_gt = json.load(open(test_path + str(scene_id).zfill(6) + '/scene_gt.json'))
        scene_gt_info = json.load(open(test_path + str(scene_id).zfill(6) + '/scene_gt_info.json'))
        
        single_scene = scene_gt[str(im_id)]
        single_scene_info = scene_gt_info[str(im_id)]
        for i in range(len(single_scene)):
            if obj_id == single_scene[i]['obj_id']:
                cam_R_m2c = single_scene[i]["cam_R_m2c"]
                cam_t_m2c = single_scene[i]["cam_t_m2c"]
                vis = single_scene_info[i]["visib_fract"]

                bb = confirmed_instance['bbox']
                bbox_obj = [math.floor(bb[0]), math.floor(bb[1]),
                            math.ceil(bb[2]), math.ceil(bb[3])]
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
                roi = rgb[top_trunc:bottom_trunc, left_trunc:right_trunc]
                resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_CUBIC)

                path = os.path.join(test_save_dir, str(obj_id).zfill(6),
                                    str(scene_id).zfill(6))
                if os.path.exists(path) is False:
                    os.makedirs(path)

                filename = '%06i_%06i_%06i.png' % (obj_id, scene_id, im_id)
                img_dir = os.path.join(test_save_dir, str(obj_id).zfill(6),
                                       str(scene_id).zfill(6), filename)

                imageio.imwrite(img_dir, resized)
                print(img_dir)

                name.append(filename)
                rotation.append(cam_R_m2c)
                translation.append(cam_t_m2c)
                ob_id.append(obj_id)
                bbox.append(bbox_obj)
                vis_fra.append(vis)

myset = set(name)

if len(name) != len(myset):
    print("duplicates found in the list")
else:
    print("No duplicates found in the list")
    csvfile = {'filename': name, 'r': rotation, 't': translation,
                'bbox': bbox, 'id': ob_id, 'vis': vis_fra }
    df = pd.DataFrame(csvfile)
    df.set_index("id", inplace=True)
    df.head()
    
    for obidx in obj_range:
        if obidx in ob_id:
            subset = df.loc[obidx]
            subset = subset.to_dict(orient='records')
            json_name = '%06i_all.json' % (obidx)
            json_path = os.path.join(test_save_dir, json_name)
            json_file = open(json_path, "w")
            json.dump(subset, json_file)
