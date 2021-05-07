import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
import cv2
import visualization
import unet_model

LOAD_DATA = True
root = "./data/NormalizedImg/"
path_save_img = "./data/model_imgs/"
path_save_mask = "./data/model_masks/"
dim_img = 256 * 2


if LOAD_DATA:
    images = os.listdir(root)
    images.sort()
    im_array = []
    for i in images:
        read_image = cv2.imread(root + i)
        image_grayscale = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(image_grayscale, (dim_img, dim_img))[:, :]
        im_array.append(im)

unet_pretrained = unet_model.unet(input_size=(512, 512, 1))
unet_pretrained.summary()
unet_pretrained.load_weights('./pretrained_models/cxr_reg_weights.best.hdf5')

# Predict on everything
all_imgs = np.array(im_array).reshape(len(im_array), dim_img, dim_img, 1)
preds = unet_pretrained.predict(all_imgs)

# Plot prediction and image
i_test = 0
img_i = np.squeeze(all_imgs[i_test])
pred_i = np.squeeze(preds[i_test])
visualization.plot_img_pred(img_i, pred_i)
