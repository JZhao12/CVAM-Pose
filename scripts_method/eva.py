#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:36:11 2024

@author: jianyu
"""

# %% import modules

import sys
sys.path.append('/home/jianyu/CVAM-Pose/bop_toolkit/')

import os
import numpy as np
import json
import trimesh
import pyrender
import imageio.v2
import glob
from bop_toolkit_lib import pose_error, misc
from bop_toolkit_lib.inout import load_ply
from bop_toolkit_lib.misc import get_symmetry_transformations
from bop_toolkit_lib import visibility

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# %% path, argparse, data information
parent_dir = os.getcwd()

latent_path = parent_dir + '/results/lmo/latent/'
result_path = parent_dir + '/results/lmo/pose/'

lmo_obj_range = [1, 5, 6, 8, 9, 10, 11, 12]
MeshDir = parent_dir + '/original data/lmo/models_eval/'

"""
The VSD, MSSD, MSPD are based on:
https://github.com/thodan/bop_toolkit

"""

# %% VSD


def render_object(R, T, trimesh):
    TS = np.array([[T[0]],
                   [T[1]],
                   [T[2]]])
    Ex_parameter = np.array([[0.0, 0.0, 0.0, 1.0]])
    inter_pose = np.concatenate((R, TS), axis=1)
    objectpose = np.concatenate((inter_pose, Ex_parameter), axis=0)
    mesh = pyrender.Mesh.from_trimesh(trimesh, poses=objectpose,
                                      smooth=True)
    scene = pyrender.Scene(bg_color=([0, 0, 0]))
    scene.add(mesh)

    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=640,
                                   viewport_height=480)
    _, depth = r.render(scene)
    r.delete()
    return depth


print('first metric: VSD')

BOP_score = []

lmo_ori_depth = parent_dir + '/original data/lmo/test/000002/depth/'
lmo_test_dir = parent_dir + '/processed data/lmo/test_bop/'

test_camera = json.load(open(parent_dir + '/original data/lmo/test/000002/scene_camera.json'))
K = np.reshape(test_camera['3']['cam_K'], (3, 3))
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05,
                                    zfar=5000.0)
camera_pose = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=5)

Overall = []
for ob_id in lmo_obj_range:

    X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
    PredictedRM = X['r']

    X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
    PredictedTV = X['t']
    # PredictedTV = np.squeeze(PredictedTV, axis=2)

    X = np.load(latent_path + 'test_' + str(ob_id).zfill(6) + '.npz')
    GT_RM = X['r']
    GT_RM = GT_RM.reshape(-1, 3, 3)
    GT_TV = X['t']

    lmo_test = sorted(glob.glob(lmo_test_dir + str(ob_id).zfill(6) + '/000002/' + "/**/*.png", recursive=True))
    
    depth_list = []
    for t in lmo_test:
        t_name = t[-10:]
        depth_list.append(t_name)

    Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
    model_info = json.load(open(MeshDir + 'models_info.json'))
    fuze_trimesh = trimesh.load(Target_model)

    delta = 15
    normalized_by_diameter = True
    diameter = model_info[str(ob_id)]['diameter']
    taus = list(np.arange(0.05, 0.51, 0.05))
    cost_type = 'step'
    correct_th = np.arange(0.05, 0.51, 0.05)

    single_vsd = []
    for i in range(len(GT_TV)):
        R_est = PredictedRM[i, :]
        t_est = PredictedTV[i, :]
        depth_est = render_object(R_est, t_est, fuze_trimesh)

        R_gt = GT_RM[i, :]
        t_gt = GT_TV[i, :]
        
        depth_gt = render_object(R_gt, t_gt, fuze_trimesh)
        
        depth_test = imageio.v2.imread(lmo_ori_depth + depth_list[i])

        dist_test = misc.depth_im_to_dist_im_fast(depth_test, K)
        dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
        dist_est = misc.depth_im_to_dist_im_fast(depth_est, K)

        visib_gt = visibility.estimate_visib_mask_gt(
            dist_test, dist_gt, delta, visib_mode='bop19')
        visib_est = visibility.estimate_visib_mask_est(
            dist_test, dist_est, visib_gt, delta, visib_mode='bop19')

        visib_inter = np.logical_and(visib_gt, visib_est)
        visib_union = np.logical_or(visib_gt, visib_est)

        visib_union_count = visib_union.sum()
        visib_comp_count = visib_union_count - visib_inter.sum()

        dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])

        if normalized_by_diameter:
            dists /= diameter

        if visib_union_count == 0:
            errors = [1.0] * len(taus)
        else:
            errors = []
        for tau in taus:

            # Pixel-wise matching cost.
            if cost_type == 'step':
                costs = dists >= tau
            elif cost_type == 'tlinear':
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError('Unknown pixel matching cost.')

            e = ((np.sum(costs) + visib_comp_count) / float(visib_union_count))
            errors.append(e)

        good_prediction = 0
        for th_idx in range(10):
            th = correct_th[th_idx]
            er_vsd = errors[th_idx]
            if er_vsd < th:
                good_prediction += 1
            single_acc = good_prediction/10 * 100

        single_vsd.append(single_acc)

    e_vsd = np.mean(single_vsd)
    Overall.append(e_vsd)
print('AR(VSD): ', np.mean(Overall))
BOP_score.append(np.mean(Overall))


# MSSD
print('second metric: MSSD')

Overall = []

for ob_id in lmo_obj_range:

    X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
    PredictedRM = X['r']

    X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
    PredictedTV = X['t']

    X = np.load(latent_path + 'test_' + str(ob_id).zfill(6) + '.npz')
    GT_RM = X['r']
    GT_RM = GT_RM.reshape(-1, 3, 3)
    GT_TV = X['t']

    Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
    Modelpoint = load_ply(Target_model)['pts']
    model_info = json.load(open(MeshDir + 'models_info.json'))
    s_trans = get_symmetry_transformations(model_info[str(ob_id)], 0.01)
    diameter = model_info[str(ob_id)]['diameter']

    radio_range = np.arange(0.05, 0.51, 0.05)
    MSSD = []
    for radio_idx in range(len(radio_range)):
        radio = radio_range[radio_idx]

        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):
            R_est = PredictedRM[i]
            t_est = np.reshape(PredictedTV[i], (3, 1))
            R_gt = GT_RM[i]
            t_gt = np.reshape(GT_TV[i], (3, 1))

            distance = pose_error.mssd(R_est, t_est, R_gt, t_gt, Modelpoint, s_trans)

            All_distance.append(distance)
            if distance < diameter*radio:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        MSSD.append(Accuracy)

    e_MSSD = np.mean(MSSD)
    Overall.append(e_MSSD)
print("AR(MSSD): ", np.mean(Overall))
BOP_score.append(np.mean(Overall))

print('third metric: MSPD')

Overall = []

for ob_id in lmo_obj_range:

    X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
    PredictedRM = X['r']

    X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
    PredictedTV = X['t']

    X = np.load(latent_path + 'test_' + str(ob_id).zfill(6) + '.npz')
    GT_RM = X['r']
    GT_RM = GT_RM.reshape(-1, 3, 3)
    GT_TV = X['t']

    Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
    Modelpoint = load_ply(Target_model)['pts']
    model_info = json.load(open(MeshDir + 'models_info.json'))
    s_trans = get_symmetry_transformations(model_info[str(ob_id)], 0.01)

    radio_range = np.arange(5, 51, 5)
    w = 640
    r = w/640

    MSPD = []
    for radio_idx in range(len(radio_range)):
        radio = radio_range[radio_idx]

        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):
            R_est = PredictedRM[i]
            t_est = np.reshape(PredictedTV[i], (3, 1))
            R_gt = GT_RM[i]
            t_gt = np.reshape(GT_TV[i], (3, 1))

            distance = pose_error.mspd(R_est, t_est, R_gt, t_gt, K, Modelpoint, s_trans)

            All_distance.append(distance)
            if distance < r*radio:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        MSPD.append(Accuracy)

    e_MSPD = np.mean(MSPD)
    Overall.append(e_MSPD)
print("AR(MSPD): ", np.mean(Overall))

BOP_score.append(np.mean(Overall))
print("AR(score): ", np.mean(BOP_score))
