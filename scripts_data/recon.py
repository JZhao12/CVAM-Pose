#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:04:18 2022

@author: bogdan
"""

# %% import modules
import os
import argparse
import numpy as np
import trimesh
import pyrender
import json
import imageio
import cv2
import math
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# %% path, argparse, model information
parser = argparse.ArgumentParser(description='pbr reconstruction images')
parser.add_argument('--object',
                    type=int,
                    help='object to reconstruct')
args = parser.parse_args()

parent_dir = os.getcwd()
ori_path = parent_dir + '/original data/lmo/'
model_path = ori_path + '/models/'
light_inten = 5

pbr_save_dir = parent_dir + '/processed data/lmo/pbr/'
recon_save = parent_dir + '/processed data/pbr/pbr_recon/'

# %% Configurations of the rendering

"""
The off-screen rendering is based on this example:
https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

"""

view_width = 3200
view_height = 2400
camera = pyrender.PerspectiveCamera(yfov=2)

light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]),
                                  intensity=light_inten)
camera_pose = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
Ex_parameter = np.array([[0.0, 0.0, 0.0, 1.0]])

# %% start render each object, and save images

object_number = args.object
ob_id = '%06i' % (object_number)

gt_info = json.load(open(pbr_save_dir + ob_id + '_all.json'))
fuze_trimesh = trimesh.load(model_path + 'obj_' + ob_id + '.ply')

for column in range(len(gt_info)):

    filename = gt_info[column]['filename']

    r = gt_info[column]['r']
    r = np.reshape(r, (3, 3))

    tv = gt_info[column]['t']
    t = np.array([[tv[0]], [tv[1]], [tv[2]]])

    inter_pose = np.concatenate((r, t), axis=1)
    objectpose = np.concatenate((inter_pose, Ex_parameter), axis=0)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh,
                                      poses=objectpose, smooth=True)

    scene = pyrender.Scene(bg_color=([0, 0, 0]))
    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    ren = pyrender.OffscreenRenderer(viewport_width=3200,
                                     viewport_height=2400)
    rendered_image, _ = ren.render(scene)

    if rendered_image[0, :].any() or rendered_image[2399, :].any() or rendered_image[:, 0].any() or rendered_image[:, 3199].any():
        print('error')
        break
    for pixel in range(view_height):
        a = rendered_image[pixel, :]
        if a.any():
            y = pixel
            break

    for pixel in reversed(range(view_height)):
        a = rendered_image[pixel, :]
        if a.any():
            y2 = pixel
            break

    for pixel in range(view_width):
        a = rendered_image[:, pixel]
        if a.any():
            x = pixel
            break

    for pixel in reversed(range(view_width)):
        a = rendered_image[:, pixel]
        if a.any():
            x2 = pixel
            break

    w = x2-x
    h = y2-y

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
    right_trunc = np.minimum(new_x + new_w, rendered_image.shape[1])
    top_trunc = np.maximum(new_y, 0)
    bottom_trunc = np.minimum(new_y + new_h, rendered_image.shape[0])

    ROI = rendered_image[top_trunc:bottom_trunc, left_trunc:right_trunc]
    resized = cv2.resize(ROI, (128, 128), interpolation=cv2.INTER_CUBIC)

    objectid = filename[0:6]
    folderid = filename[7:13]
    save_path = os.path.join(recon_save, objectid, folderid)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rgb_dir = os.path.join(recon_save, objectid, folderid, filename)
    imageio.imwrite(rgb_dir, resized)
    print(rgb_dir)
    ren.delete()
