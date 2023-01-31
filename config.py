# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:20:58 2023

config

@author: ASUS
"""

import os
import numpy as np

image_dir = "D:/Study/Code/Image_Processing/Reconstruction/data//part/"
MRT = 0.8 # match ratio
dist_threshold = 70 # distance_threshold
reproj_threshold = 15# reproj_err_thre
kp = 'SIFT'
K = np.array([
        [901.2743,      0,              952.9357],
        [0,             901.6915,       536.2312],
        [0,      0,       1]])

#选择性删除所选点的范围。
x = 0.5
y = 1