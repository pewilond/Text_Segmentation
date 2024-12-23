import cv2
import json
import os
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

class RandomLines(ImageOnlyTransform):
    def __init__(self, num_lines=5, max_length=40, color=(0, 0, 0), thickness=1, always_apply=False, p=0.2):
        super(RandomLines, self).__init__(always_apply=always_apply, p=p)
        self.num_lines = num_lines
        self.max_length = max_length
        self.color = color
        self.thickness = thickness

    def apply(self, image, **params):
        h, w = image.shape[:2]
        for _ in range(self.num_lines):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(10, self.max_length)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            cv2.line(image, (x1, y1), (x2, y2), self.color, self.thickness)
        return image

    def get_transform_init_args_names(self):
        return ("num_lines", "max_length", "color", "thickness")
    
class SpoiltImage(ImageOnlyTransform):
    def __init__(self, max_spoils=5, max_size=20, color=(0, 0, 0), always_apply=False, p=0.1):
        super(SpoiltImage, self).__init__(always_apply=always_apply, p=p)
        self.max_spoils = max_spoils
        self.max_size = max_size
        self.color = color

    def apply(self, image, **params):
        h, w = image.shape[:2]
        num_spoils = random.randint(1, self.max_spoils)
        for _ in range(num_spoils):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(5, self.max_size)
            cv2.circle(image, (x, y), size, self.color, -1)
        return image

    def get_transform_init_args_names(self):
        return ("max_spoils", "max_size", "color")


def get_standard_transform():
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.PadIfNeeded(min_height=224, min_width=224, 
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=(255,255,255), p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),

    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    return transform

def get_standard_plus_specific_transform():
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.9, 1.1), shear=(-10, 10), p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        RandomLines(num_lines=5, max_length=40, color=(0, 0, 0), p=0.2),
        SpoiltImage(max_spoils=5, max_size=20, color=(0, 0, 0), p=0.1),
        A.PadIfNeeded(min_height=224, min_width=224, 
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=(255, 255, 255), p=1.0),
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    return transform


def draw_boxes_masks(img, boxes, masks):
    img_with_boxes = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1,y1), (x2,y2), (0,255,0), 2)

    if len(masks) > 0:
        combined_mask = np.zeros_like(img_with_boxes, dtype=np.uint8)
        for m in masks:
            mask_3d = np.zeros_like(img_with_boxes)
            mask_3d[m == 1] = (0,0,255)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, mask_3d, 0.5, 0)
        img_with_boxes = cv2.addWeighted(img_with_boxes, 1.0, combined_mask, 0.5, 0)
    return img_with_boxes

def visualize_before_after(original_image, original_boxes, original_masks, 
                           augmented_image, augmented_boxes, augmented_masks):
    orig_viz = draw_boxes_masks(original_image, original_boxes, original_masks)
    aug_viz = draw_boxes_masks(augmented_image, augmented_boxes, augmented_masks)

    fig, axs = plt.subplots(1, 4, figsize=(20,10))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(orig_viz)
    axs[1].set_title("Original with Boxes & Masks")

    axs[2].imshow(augmented_image)
    axs[2].set_title("Augmented Image")

    axs[3].imshow(aug_viz)
    axs[3].set_title("Augmented with Boxes & Masks")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
