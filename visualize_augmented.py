import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as maskUtils
import cv2

AUG_IMAGES_DIR = 'data_seg_augmented/images_train'
AUG_ANN_FILE = 'data_seg_augmented/annotations_train.json'

with open(AUG_ANN_FILE, 'r', encoding='utf-8') as f:
    augmented_dataset = json.load(f)

augmented_images = [img for img in augmented_dataset['images'] if img['file_name'].startswith('aug_')]

if not augmented_images:
    exit()

random_aug_image = random.choice(augmented_images)

image_id = random_aug_image['id']
image_file = random_aug_image['file_name']
image_path = os.path.join(AUG_IMAGES_DIR, image_file)

if not os.path.exists(image_path):
    print(f"empty file path {image_path} .")
    exit()

image = Image.open(image_path).convert('RGB')
image_np = np.array(image)

annotations = [ann for ann in augmented_dataset['annotations'] if ann['image_id'] == image_id]

fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(image_np)

colors = plt.cm.hsv(np.linspace(0, 1, 256)).tolist()

for ann in annotations:
    bbox = ann['bbox']
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    segmentation = ann['segmentation']
    if isinstance(segmentation, list):
        for seg in segmentation:
            if len(seg) < 6:
                continue
            poly = np.array(seg).reshape((-1, 2))
            poly = np.round(poly, 2)
            ax.plot(poly[:, 0], poly[:, 1], linewidth=1.5)
    elif isinstance(segmentation, dict):
        rle = segmentation
        mask = maskUtils.decode(rle)
        if mask.sum() == 0:
            print(f"mask RGB for ID {ann['id']} empty.")
            continue
   
        color = np.array(colors[ann['category_id'] % 256][:3]) * 255
        color = color.astype(np.uint8)
        mask_rgb = np.zeros_like(image_np, dtype=np.uint8)
        mask_rgb[mask == 1] = color
        
        if mask_rgb.sum() == 0:
            print(f"mask RGB for ID {ann['id']} empty.")
            continue
        
        alpha = 0.5
        ax.imshow(mask_rgb, alpha=alpha)
    else:
        print(f"mask problem {ann['id']}")

    category_id = ann['category_id']
    category = next((cat['name'] for cat in augmented_dataset['categories'] if cat['id'] == category_id), str(category_id))

plt.title(f"Augmented image: {image_file}")
plt.axis('off')
plt.show()
