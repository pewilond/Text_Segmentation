import os
import json
import shutil
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import torch
import time
from tqdm import tqdm
import random

from augmentation_pipelines import (
    get_standard_transform, 
    get_standard_plus_specific_transform
)

from train.train_mask_rcnn.transform import get_transform

ORIG_IMAGES_DIR = 'data_seg/images_train'
ORIG_ANN_FILE = 'data_seg/annotations_train.json'

AUG_DATASET_DIR = 'data_seg_augmented'  
AUG_IMAGES_DIR = os.path.join(AUG_DATASET_DIR, 'images_train')
AUG_ANN_FILE = os.path.join(AUG_DATASET_DIR, 'annotations_train.json')

os.makedirs(AUG_DATASET_DIR, exist_ok=True)
os.makedirs(AUG_IMAGES_DIR, exist_ok=True)

with open(ORIG_ANN_FILE, 'r', encoding='utf-8') as f:
    orig_dataset = json.load(f)

coco = COCO()
coco.dataset = orig_dataset
coco.createIndex()

img_ids = coco.getImgIds()

if os.path.exists(AUG_ANN_FILE):
    with open(AUG_ANN_FILE, 'r', encoding='utf-8') as f:
        combined_dataset = json.load(f)
else:
    combined_dataset = {
        'images': orig_dataset['images'].copy(),
        'annotations': orig_dataset['annotations'].copy(),
        'categories': orig_dataset['categories'].copy()
    }
    with open(AUG_ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)

print("Копирование оригинальных изображений...")
for img_info in tqdm(orig_dataset['images'], desc="Copying images"):
    src_img_path = os.path.join(ORIG_IMAGES_DIR, img_info['file_name'])
    dst_img_path = os.path.join(AUG_IMAGES_DIR, img_info['file_name'])
    if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
        shutil.copy(src_img_path, dst_img_path)

print("Копирование завершено.")
def mask_to_polygon(mask, epsilon=2.0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_flat = approx.flatten().tolist()
        if len(approx_flat) > 4:
            segmentation.append(approx_flat)
    return segmentation

transform_pipeline = get_standard_plus_specific_transform()
pre_transform = get_transform(True)

if len(combined_dataset['annotations']) > 0:
    ann_id_start = max(ann['id'] for ann in combined_dataset['annotations']) + 1
else:
    ann_id_start = 1

if len(combined_dataset['images']) > 0:
    img_id_new = max(img['id'] for img in combined_dataset['images']) + 1
else:
    img_id_new = 1

processed_orig_ids = set([img['id'] for img in combined_dataset['images'] if not img['file_name'].startswith('aug_')])
all_orig_ids = set(img_ids)
remaining_ids = list(all_orig_ids - processed_orig_ids)
remaining_ids.sort()

if not remaining_ids:
    print("Все изображения уже обработаны.")
    exit()

print("Начинаем аугментацию...")
start_time_total = time.time()

ann_id = ann_id_start

for image_id in tqdm(remaining_ids, desc="Augmenting images"):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(ORIG_IMAGES_DIR, img_info['file_name'])

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Не удалось открыть изображение {img_path}: {e}")
        continue

    w, h = image.size

    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)

    boxes = []
    masks = []
    labels = []
    attributes_list = []
    MAX_MASKS = 200

    for ann in anns:
        x, y, w_box, h_box = ann['bbox']
        boxes.append([x, y, x + w_box, y + h_box])
        mask = coco.annToMask(ann).astype(bool)
        masks.append(mask)
        labels.append(ann['category_id'])
        attributes_list.append(ann.get('attributes', {}))

    if len(boxes) > MAX_MASKS:
        boxes = boxes[:MAX_MASKS]
        masks = masks[:MAX_MASKS]
        labels = labels[:MAX_MASKS]
        attributes_list = attributes_list[:MAX_MASKS]

    boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
    masks = np.stack(masks, axis=0) if masks else np.zeros((0, h, w), dtype=bool)
    labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

    tmp_target = {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'masks': torch.tensor(masks, dtype=torch.bool),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }

    try:
        image_pre, target_pre = pre_transform(image, tmp_target)
    except Exception as e:
        print(f"Ошибка при предобработке изображения ID {image_id}: {e}")
        continue

    aug_image_np = (image_pre.permute(1,2,0).numpy()*255).astype(np.uint8)
    aug_boxes = target_pre['boxes'].numpy()
    aug_masks = target_pre['masks'].numpy()
    aug_labels = target_pre['labels'].numpy().tolist()

    try:
        transformed = transform_pipeline(
            image=aug_image_np, 
            bboxes=aug_boxes,
            masks=aug_masks.astype(np.uint8),
            category_ids=aug_labels
        )
    except MemoryError as e:
        print(f"Ошибка памяти при обработке изображения ID {image_id}: {e}")
        continue
    except Exception as e:
        print(f"Неизвестная ошибка при обработке изображения ID {image_id}: {e}")
        continue

    final_image = transformed['image']
    final_boxes = np.array(transformed['bboxes'], dtype=np.float32)
    final_masks = np.array(transformed['masks'], dtype=bool)
    final_labels = np.array(transformed['category_ids'], dtype=np.int64)

    aug_img_name = f"aug_{os.path.splitext(img_info['file_name'])[0]}_{img_id_new}.jpg"
    cv2.imwrite(os.path.join(AUG_IMAGES_DIR, aug_img_name), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    combined_dataset['images'].append({
        'file_name': aug_img_name,
        'height': final_image.shape[0],
        'width': final_image.shape[1],
        'id': img_id_new,
        'date_captured': img_info.get('date_captured', 0),
        'flickr_url': img_info.get('flickr_url', ''),
        'coco_url': img_info.get('coco_url', ''),
        'license': img_info.get('license', 0)
    })

    if len(final_boxes) != len(final_masks):
        print(f"Внимание: количество коробок ({len(final_boxes)}) не совпадает с количеством масок ({len(final_masks)}) для ID {image_id}")
        min_len = min(len(final_boxes), len(final_masks))
        final_boxes = final_boxes[:min_len]
        final_masks = final_masks[:min_len]
        final_labels = final_labels[:min_len]
        attributes_list = attributes_list[:min_len]

    for box, mask, label, original_attributes in zip(final_boxes, final_masks, final_labels, attributes_list):
        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1

        segmentation = mask_to_polygon(mask, epsilon=2.0)
        if not segmentation:
            continue

        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        area = maskUtils.area(rle)

        attributes = original_attributes if original_attributes else {}
        ann = {
            'image_id': img_id_new,
            'bbox': [float(x1), float(y1), float(w_box), float(h_box)],
            'category_id': int(label),
            'iscrowd': 0,
            'id': ann_id,
            'segmentation': segmentation,
            'area': float(area),
            'attributes': attributes
        }
        ann_id += 1
        combined_dataset['annotations'].append(ann)

    img_id_new += 1

    with open(AUG_ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)

end_time_total = time.time()
print(f"Аугментация завершена за {end_time_total - start_time_total:.2f} секунд.")
print("Обновленный датасет сохранен в:", AUG_DATASET_DIR)
