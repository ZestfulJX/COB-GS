import os
import numpy as np
from PIL import Image
import cv2
import sys

dataset_name = sys.argv[1]

dataset_name_real = dataset_name.split("_")[0]
# You can change gt_folder_path to your dataset xxx/llff_mask/masks. Of course, I'm ZJX.
gt_folder_path = os.path.join('/data_nvme/zjx/llff_mask/masks', dataset_name)

pred_folder_path = os.path.join('output', dataset_name_real, 'test/ours_22x/mask_renders', dataset_name)

# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt > 128).astype('uint8')
    gt = (gt > 128).astype('uint8')

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def load_mask(mask_path):
    """Load the mask from the given path."""
    # print(os.path.exists(mask_path))
    if os.path.exists(mask_path):
        return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
    return None


def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))


def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_accuracy(mask1, mask2):
    """Calculate accuracy between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    correct_predictions = np.sum(mask1_bool == mask2_bool)
    total_pixels = mask1.size
    accuracy = correct_predictions / total_pixels
    return accuracy

iou_scores = {}  # Store IoU scores for each class
acc_scores = {}
class_counts = {}  # Count the number of times each class appears
# Iterate over each image and category in the GT dataset
idx = 0
png_files = [f for f in os.listdir(gt_folder_path) if f.endswith('.png')]
print(gt_folder_path)
print(len(png_files))
if len(png_files) == 1:
    gt_mask_path = os.path.join(gt_folder_path, png_files[0])
else:
    raise ValueError("no only png")
pred_mask_path = os.path.join(pred_folder_path, f'00000.png')
gt_mask = load_mask(gt_mask_path)
pred_mask = load_mask(pred_mask_path)
print("GT:  ", gt_mask_path)
print("Pred:  ", pred_mask_path)
if gt_mask is not None and pred_mask is not None:
    # Resize prediction mask to match GT mask shape if they are different
    if pred_mask.shape != gt_mask.shape:
        pred_mask = resize_mask(pred_mask, gt_mask.shape)

    iou = calculate_iou(gt_mask, pred_mask)
    acc = calculate_accuracy(gt_mask, pred_mask)
    biou = boundary_iou(gt_mask, pred_mask)

    print("IoU:", iou)
    print("Boundary IoU:", biou)
    print("Acc:", acc)

    with open(f'{pred_folder_path}/results.txt', 'w') as f:
        f.write(f'IoU: {iou}\n')
        f.write(f'Boundary IoU: {biou}\n')
        f.write(f'"Acc : {acc}\n')
