import os
import json
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

MAX_MASKS = 200 

class CustomDataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(ann_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        self.coco = COCO()
        self.coco.dataset = dataset
        self.coco.createIndex()

        self.ids = list(sorted(self.coco.imgs.keys()))

        cat_ids = self.coco.getCatIds()
        categories = self.coco.loadCats(cat_ids)
        self.category_id_to_label = {cat['id']: idx for idx, cat in enumerate(categories, start=1)}
        self.num_classes = len(self.category_id_to_label) + 1
        
        print(f"Category ID to Label Mapping: {self.category_id_to_label}")
        print(f"Number of classes (including background): {self.num_classes}")

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []

        for ann in anns[:MAX_MASKS]:
            cat_id = ann['category_id']
            label = self.category_id_to_label.get(cat_id, 0)
            labels.append(label)

            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

            mask = self.coco.annToMask(ann)
            mask = torch.from_numpy(mask).bool()
            masks.append(mask)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.bool)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks, dim=0)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            image_width, image_height = img_info['width'], img_info['height']

            assert (area > 0).all(), f"Zero or negative area boxes found for image_id {image_id}"

            assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= image_width).all(), f"Box x-coordinates out of bounds for image_id {image_id}"
            assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= image_height).all(), f"Box y-coordinates out of bounds for image_id {image_id}"

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor(image_id, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)
