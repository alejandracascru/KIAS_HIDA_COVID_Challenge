import os
import cv2

LOAD_DATA = True
root = "./data/test_set/normalizedImg/"
path_save = "./data/test_set/threshold_otsu_test/"
path_save_mask = "./data/test_set/threshold_otsu_test_mask/"


if LOAD_DATA:
    images = os.listdir(root)
    images.sort()
    im_array = []
    for i in images:
        read_image = cv2.imread(root + i)
        image_grayscale = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_grayscale, (25, 25), 0)
        new_img = cv2.convertScaleAbs(image_grayscale, alpha=2, beta=80)
        ret2, th2 = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_final = cv2.bitwise_not(th2)
        mask_applied = cv2.bitwise_and(image_grayscale, image_grayscale, mask=mask_final)
        image_small = cv2.resize(mask_applied, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path_save + i, image_small)
        cv2.imwrite(path_save_mask + i, mask_final)
