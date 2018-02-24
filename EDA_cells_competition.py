import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import feature
from skimage.filters import threshold_otsu


def get_masks(image_folder):
    # Get all masks from a directory per one image
    masks_path = os.path.join(image_folder, 'masks')
    masks_dict = dict()
    for i in range(len(os.listdir(masks_path))):
        masks_dict[i] = scipy.ndimage.imread(os.path.join(masks_path, os.listdir(masks_path)[i]))

    return masks_dict


def threshold_mask(mask_img):
    # Threshold masks to binary based on Otsu thresholding
    thresh_val = threshold_otsu(mask_img)
    mask = np.where(mask_img > thresh_val, 1, 0)
    return mask


def add_all_masks(masks):
    # Add all masks to one image
    ans = np.zeros([len(masks[0]), len(masks[0][0])])
    for i in range(len(masks)):
        ans += threshold_mask((masks[i]))

    return ans


def overlay_image_and_masks(img, masks):
    # Overlay original image and masks for better visualization , and plot image, masks and overlay
    added_mask = add_all_masks(masks)
    edge_mask = feature.canny(added_mask)
    temp_img = img.copy()
    temp_img[edge_mask] = 0

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title('Masks')
    plt.imshow(added_mask)
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(temp_img)
    plt.show()


def read_img_and_masks(img_number, images_folders):
    # call overlay_image_and_masks for a particular image
    img_path = os.path.join(images_folders[img_number], 'images')
    img = scipy.ndimage.imread(os.path.join(img_path, os.listdir(img_path)[0]))
    masks = get_masks(images_folders[img_number])
    overlay_image_and_masks(img, masks)


if __name__ == '__main__':
    # The path to the data folder in your computer
    path = r'C:\Users\omri\Personal\Kaggle\Project'

    # Get Paths to all images and masks
    images_path = os.path.join(path, 'stage1_train')
    images_folders = [os.path.join(images_path, x) for x in os.listdir(images_path)]

    # Examples
    read_img_and_masks(1, images_folders)
    read_img_and_masks(2, images_folders)
    read_img_and_masks(3, images_folders)
