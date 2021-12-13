import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from os import listdir
from os.path import join
import os

import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
import cv2
from tqdm import tqdm


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


def median_imgs(imgs_dir):

    for i in range(3,12):
        img_dirs = sorted(glob.glob(f"{imgs_dir}/*.jpg"))[:i]
        print(img_dirs)
        imgs_r = []
        imgs_g = []
        imgs_b = []

        for img_dir in tqdm(img_dirs):
            img = cv2.imread(img_dir)
            imgs_b.append(img[:, :, 0])
            imgs_g.append(img[:, :, 1])
            imgs_r.append(img[:, :, 2])

        imgs_r = np.median(np.stack(imgs_r, axis=0), axis=0)
        imgs_g = np.median(np.stack(imgs_g, axis=0), axis=0)
        imgs_b = np.median(np.stack(imgs_b, axis=0), axis=0)

        imgs = np.stack([imgs_b, imgs_g, imgs_r], axis=2)

        cv2.imwrite(f'{str(i).zfill(2)}.png', imgs)




my_dir = 'rain_DB/level4'
# img_dirs = [os.path.join(my_dir, x) for x in sorted(os.listdir(my_dir))]
# print(img_dirs)

img3 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0017.jpg')


img0 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0014.jpg')
print(numpyPSNR(img3, img0))
img1 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0015.jpg')
print(numpyPSNR(img3, img1))
img2 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0016.jpg')
print(numpyPSNR(img3, img2))


img4 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0018.jpg')
print(numpyPSNR(img3, img4))
img5 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0019.jpg')
print(numpyPSNR(img3, img5))
img6 = cv2.imread('/home/lab/works/projects/human_and_forest/noisy_style_transformer_train/temp/rain_DB/level4/D-210801_O1012R04_002_0020.jpg')
print(numpyPSNR(img3, img6))

img7 = cv2.imread('03.png')
print(numpyPSNR(img3, img7))


# median_imgs(my_dir)
#cv2.imwrite(f'median_img.png', median_img)

#
#
# # img_sample = cv2.imread('/hdd1/works/projects/human_and_forest/MPRNet/noisy_style_transformer_train/sample_test_imgs/D-210814_O9112R03_006_0033.jpg')
#
#
# img_sample = cv2.imread('D-210801_O1012R04_002_0001.png')
#
# scale_factor = 0.25
# #img_sample = cv2.resize(img_sample, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
# ksize =25
# img_sample = cv2.medianBlur(img_sample, ksize)
#
# #img_sample = cv2.resize(img_sample, (0,0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_AREA)
#
# cv2.imwrite('only_mesian.png', img_sample)

