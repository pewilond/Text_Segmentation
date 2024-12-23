import os
import random
import numpy as np
import json
from PIL import Image
from pycocotools.coco import COCO
import torch
import cv2
from train.train_mask_rcnn.transform import get_transform

from augmentation_pipelines import (
    get_standard_transform, 
    get_standard_plus_specific_transform,
    visualize_before_after
)


ORIG_IMAGES_DIR = 'data_seg/images_train'
ORIG_ANN_FILE = 'data_seg/annotations_train.json'

with open(ORIG_ANN_FILE, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

coco = COCO()
coco.dataset = dataset
coco.createIndex()

img_ids = coco.getImgIds()
random_img_id = random.choice(img_ids)
img_info = coco.loadImgs(random_img_id)[0]
img_path = os.path.join(ORIG_IMAGES_DIR, img_info['file_name'])
image = np.array(Image.open(img_path).convert('RGB'))

ann_ids = coco.getAnnIds(imgIds=[random_img_id])
anns = coco.loadAnns(ann_ids)

boxes = []
masks = []
labels = []
for ann in anns:
    x, y, w, h = ann['bbox']
    boxes.append([x, y, x + w, y + h])
    mask = coco.annToMask(ann)
    masks.append(mask)
    labels.append(ann['category_id'])

boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
masks = np.stack(masks, axis=0) if masks else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

category_ids = labels.tolist()

transform_pipeline = get_standard_plus_specific_transform()

pre_transform = get_transform(True)

tmp_target = {
    'boxes': torch.tensor(boxes, dtype=torch.float32),
    'masks': torch.tensor(masks, dtype=torch.uint8),
    'labels': torch.tensor(labels, dtype=torch.int64)
}

image_pre, target_pre = pre_transform(Image.fromarray(image), tmp_target)

aug_image_np = (image_pre.permute(1,2,0).numpy()*255).astype(np.uint8)
aug_boxes = target_pre['boxes'].numpy()
aug_masks = target_pre['masks'].numpy()
aug_labels = target_pre['labels'].numpy().tolist()

transformed = transform_pipeline(
    image=aug_image_np,
    bboxes=aug_boxes,
    masks=aug_masks,
    category_ids=aug_labels
)
aug_image = transformed['image']
aug_boxes_transformed = np.array(transformed['bboxes'], dtype=np.float32)
aug_masks_transformed = np.array(transformed['masks'], dtype=np.uint8)
aug_labels_transformed = np.array(transformed['category_ids'], dtype=np.int64)

visualize_before_after(
    image,
    boxes,
    masks,
    aug_image,
    aug_boxes_transformed,
    aug_masks_transformed
)