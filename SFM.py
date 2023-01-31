# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:10:21 2023

Structure from motion by Pengyu Zhang

@author: ASUS
"""

import os
import cv2
import sys
import math
import config
import collections
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.linalg import lstsq
from scipy.optimize import least_squares
import time

"""
Feature extraction and matching
"""
def feature_extraction(img, keypoint = 'SIFT'):
    dtype=object
    t0 = time.time()
    if keypoint == 'SIFT':
        kp = cv2.SIFT_create(0, 3, 0.04, 10)
    if keypoint == 'orb':
        kp = cv2.ORB_create(4000)
    kp_all = []
    des_all = []
    colors_all = []
    for image_name in img:
        image = cv2.imread(image_name,0)
        if image is None:
            continue
        key_points, descriptor = kp.detectAndCompute(image, None)
        if len(key_points) <= 10:
            continue       
        kp_all.append(key_points)
        des_all.append(descriptor)
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]         
        colors_all.append(colors)   
    t1 = time.time()
    print('Feature detection takes %d seconds' % (t1-t0))
    return np.array(kp_all,dtype=object), np.array(des_all,dtype=object), np.array(colors_all,dtype=object)

def match_features(des_all):
    matches_all = []
    flann = cv2.FlannBasedMatcher(dict(algorithm = 1),dict())
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for i in range(len(des_all) - 1):
        knn_matches = bf.knnMatch(des_all[i], des_all[i + 1], k = 2)
        # knn_matches = sorted(knn_matches,key=lambda x:x[0].distance)
        matches = []
        # filter repeat train point
        ind_train = []
        for m, n in knn_matches:
            if m.distance < config.MRT * n.distance and m.trainIdx not in ind_train:
                matches.append(m)             
                ind_train.append(m.trainIdx)
        matches_all.append(matches)   
    return np.array(matches_all,dtype=object)


"""
Calculate Rotatin and Translation
"""
def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts

def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])    
    return color_src_pts, color_dst_pts

def Triangulate(K, R1, T1, R2, T2, p1, p2):
    # project P = K*[R T]
    P1 = np.zeros((3, 4))
    P2 = np.zeros((3, 4))
    P1[0:3, 0:3] = np.float32(R1)
    P1[:, 3] = np.float32(T1.T)
    P2[0:3, 0:3] = np.float32(R2)
    P2[:, 3] = np.float32(T2.T)
    fk = np.float32(K)
    P1 = np.dot(K, P1)
    P2 = np.dot(K, P2)
    # Triangulate get 3d point
    point3d = cv2.triangulatePoints(P1, P2, p1.T, p2.T).T[:,:4]  
    # regular 3D points
    for i in range(len(point3d)):
        point3d[i] /= point3d[i][3]
    
    return np.array(point3d[:,:3])

def reproj_err(point3d,point2d,R,T,K):
    # calculate reproj err
    reproj_err = []
    for i in range(len(point3d)):
        reproj_2d,jac = cv2.projectPoints(point3d[i][:3],R,T,K, np.array([]))
        err = reproj_2d.reshape(2) - point2d[i]
        reproj_err.append(np.linalg.norm(err))
    return np.array(reproj_err,dtype=object)


def init(K, kp, color, matches):
    """ get points' coordinates"""
    p1, p2 = get_matched_points(kp[0],    kp[1],    matches)
    c1, c2 = get_matched_colors(color[0], color[1], matches)
    """ calculate Essential,Rotation,Translation"""
    
    # method 1
    # F, mask = cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,1,0.99999)
    # # filter the outliers 
    # p1 = p1[mask.ravel()==1]
    # p2 = p2[mask.ravel()==1]
    # # get Essential matrix from calibration matrix and fundamental matrix, E = K'FK
    # E = np.dot(np.dot(config.K.T,F),config.K) 
    
    # method 2
    E, mask = cv2.findEssentialMat(p1, p2, config.K) ## which one?
    p1 = p1[mask.ravel()==1]
    p2 = p2[mask.ravel()==1]
    matches = np.array(matches)[mask.ravel()==1]
    
    
    # set fisrt camera as R0
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    
    # get R and T
    count, R2, T2, _ = cv2.recoverPose(E, p1, p2, config.K, mask) 
    R = [R1, R2]
    T = [T1, T2]
    """ Traingulation to 3D"""
    init_3d = Triangulate(K, R1, T1, R2, T2, p1, p2)
    
    # filter outlier by distance
    origin = np.mean(init_3d,0)
    distance = np.linalg.norm(init_3d - origin,axis=1)
    inlier = distance < config.dist_threshold
    init_3d = init_3d[inlier]
    matches = matches[inlier]
    # filter outlier by reproj err
    err = reproj_err(init_3d,p2[inlier],R2,T2,config.K)
    repro_inlier = err < config.reproj_threshold
    init_3d = init_3d[repro_inlier]
    matches = matches[repro_inlier]

    # get points3d index
    index_3d = []
    for key_p in kp:
        index_3d.append(np.ones(len(key_p)) * -1)
    index_3d = np.array(index_3d,dtype=object)
    idx = 0
    for i, match in enumerate(matches):
        index_3d[0][int(match.queryIdx)] = idx
        index_3d[1][int(match.trainIdx)] = idx
        idx += 1
    return init_3d,index_3d, R, T, origin
    

    
"""
Register more cameras
"""
def PnPsolvepose(matches, points3d, index_3d, key_points, K):
    points_3d_scene = []
    points_2d_camera = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        points3d_idx = index_3d[query_idx]  
        if points3d_idx < 0: 
            continue
        points_3d_scene.append(points3d[int(points3d_idx)])
        points_2d_camera.append(key_points[train_idx].pt)
    points_3d_scene,points_2d_camera = np.array(points_3d_scene), np.array(points_2d_camera)
    _, r, Translate, _ = cv2.solvePnPRansac(points_3d_scene, points_2d_camera, K, np.array([]))
    Rotation, _ = cv2.Rodrigues(r)
    return Rotation,Translate


def fusion_points(matches, points_3d, incremental_3d, index_3d, incre_index):
    for i,match in enumerate(matches):
        queryIdx = match.queryIdx
        trainIdx = match.trainIdx
        index_ = index_3d[queryIdx]
        if index_ >= 0: # if index_ < 0 means the 3d point is not repeat point.
            incre_index[trainIdx] = index_ 
            continue
        points_3d = np.append(points_3d, [incremental_3d[i]], axis = 0)
        index_3d[queryIdx] = incre_index[trainIdx] = len(points_3d) - 1
    return points_3d, index_3d, incre_index

"""
Bundle Adjustment
"""
# least square to get new point
def ls_solve(point3d, point2d, r, t, K):
    # define residual function
    def fun(point3d):
        reproj_2d,jac = cv2.projectPoints(point3d.reshape(1, 1, 3),r,t,K, np.array([]))
        err = reproj_2d.reshape(2) - point2d
        return err
    res = least_squares(fun, point3d)
    return res.x

def bundle_adjustment(R, T, K, keypoints, points3d, index_3d):
    for i in range(len(R)):
        r, _ = cv2.Rodrigues(R[i])
        R[i] = r
    for i in range(len(index_3d)):
        indexs = index_3d[i]
        key_point = keypoints[i]
        r = R[i]
        t = T[i]      
        for j in range(len(indexs)):
            index = int(indexs[j])
            if index < 0:
                continue
            points3d[index] = ls_solve(points3d[index], key_point[j].pt, r, t, K)           
    return points3d





if __name__ == '__main__':
    img_dir = config.image_dir
    img = []
    for root, dirs, files in os.walk(img_dir):
       for fileObj in files:
           path = os.path.join(root, fileObj).replace('\\','/')
           img.append(path)
    """
    extract feature and matching
    """
    kp, des, color = feature_extraction(img)
    matches = match_features(des)
    # # display registration result
    # draw_params = dict(matchColor = (0,255,0), flags = 2)
    # for i in range(len(matches)):
    #     result = cv2.drawMatches(cv2.imread(img[i]), kp[i], cv2.imread(img[i+1]), kp[i+1], matches[i], None, **draw_params)
    #     cv2.imshow("match", result)
    """
    init stucture
    """
    init_index = 0
    points_3d, index_3d, R, T, origin = init(config.K, kp, color, matches[init_index])
    print('Camera %d and Camera %d are used to initialize, %d pairs used to reconstructed ' 
        % (init_index, init_index+1, len(matches)))
    """ 
    Add other views points
    """
    for i in range(1, len(matches)):
        print('Register %dth camera' % (i+1))
        # get R and T by PnP
        r, t = PnPsolvepose(matches[i], points_3d, index_3d[i], kp[i+1], config.K)
        R.append(r)
        T.append(t)
        # incremental
        # get points' coordinates 
        p1, p2 = get_matched_points(kp[i],    kp[i+1],    matches[i])
        c1, c2 = get_matched_colors(color[i], color[i+1], matches[i])
        incremental_3d = Triangulate(config.K, R[i], T[i], r, t, p1, p2) 
        
        # filter before fusion
        # by origin
        distance = np.linalg.norm(incremental_3d - origin,axis=1)
        inlier = distance < config.dist_threshold
        incremental_3d = incremental_3d[inlier]
        matches[i] = np.array(matches[i])[inlier]
        # by reproj err
        err = reproj_err(incremental_3d, p2[inlier],r,t,config.K)
        repro_inlier = err < config.reproj_threshold
        incremental_3d = incremental_3d[repro_inlier]
        matches[i] = np.array(matches[i])[repro_inlier]
        
        # fusion
        points_3d, index_3d[i], index_3d[i+1] = \
        fusion_points(matches[i], points_3d, incremental_3d, index_3d[i], index_3d[i+1])
    # bundle adjustment
    points3d = bundle_adjustment(R, T, config.K, kp, points_3d, index_3d)
        
    """
    Visualization
    """
    mlab.points3d(points_3d[:,0], points_3d[:,1], points_3d[:,2],mode = 'point')

    