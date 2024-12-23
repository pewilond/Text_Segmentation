import os
import json
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from transform import get_transform
from dataset import CustomCocoDataset

def check_dataset_integrity(images_dir, ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    ann_img_ids = set(coco.getImgIds())
    dir_imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    missing_files = []
    for img_id in ann_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_files.append(img_info['file_name'])

    ann_files = set([coco.loadImgs(i)[0]['file_name'] for i in ann_img_ids])
    extra_files = [f for f in dir_imgs if f not in ann_files]

    integrity_passed = True

    if len(missing_files) == 0 and len(extra_files) == 0:
        print("Dataset integrity check passed: all images and annotations match.")
    else:
        integrity_passed = False
        if missing_files:
            print("Missing files:")
            for mf in missing_files:
                print(" ", mf)
        if extra_files:
            print("Extra files:")
            for ef in extra_files:
                print(" ", ef)
    return integrity_passed


def check_categories(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    categories = coco.loadCats(coco.getCatIds())
    valid_cat_ids = set([cat['id'] for cat in categories])

    annotations = coco.loadAnns(coco.getAnnIds())
    ann_cat_ids = set([ann['category_id'] for ann in annotations])

    undefined_cats = ann_cat_ids - valid_cat_ids

    if not undefined_cats:
        print("Category check passed: all category_ids are defined.")
        return True
    else:
        print("Undefined category_ids found in annotations:")
        for cid in undefined_cats:
            print(" ", cid)
        return False


def check_duplicate_image_ids(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    image_ids = [image['id'] for image in dataset.get('images', [])]
    unique_image_ids = set(image_ids)

    if len(image_ids) == len(unique_image_ids):
        print("Duplicate image_id check passed: No duplicates found.")
        return True
    else:
        duplicates = set([x for x in image_ids if image_ids.count(x) > 1])
        print("Duplicate image_ids found:")
        for dup in duplicates:
            print(" ", dup)
        return False


def check_annotation_fields(ann_file):
    required_fields = {'image_id', 'category_id', 'bbox', 'segmentation', 'iscrowd', 'id'}
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    annotations = dataset.get('annotations', [])
    missing_fields = set()

    for ann in annotations:
        missing = required_fields - ann.keys()
        if missing:
            missing_fields.update(missing)

    if not missing_fields:
        print("Annotation fields check passed: All annotations contain required fields.")
        return True
    else:
        print("Missing fields in annotations:")
        for field in missing_fields:
            print(" ", field)
        return False


def check_bbox_format(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    annotations = dataset.get('annotations', [])
    invalid_bboxes = []

    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            invalid_bboxes.append((ann['id'], bbox))
            continue
        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            invalid_bboxes.append((ann['id'], bbox))

    if not invalid_bboxes:
        print("Bounding boxes check passed: All bounding boxes have positive width and height.")
        return True
    else:
        print("Invalid bounding boxes found in annotations:")
        for ann_id, bbox in invalid_bboxes:
            print(f" Annotation ID {ann_id}: {bbox}")
        return False


def check_segmentation_format(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    annotations = dataset.get('annotations', [])
    invalid_segmentations = []

    for ann in annotations:
        seg = ann.get('segmentation', None)
        if not seg:
            invalid_segmentations.append(ann['id'])
            continue
        if isinstance(seg, list):
            if not all(isinstance(poly, list) for poly in seg):
                invalid_segmentations.append(ann['id'])
        elif isinstance(seg, dict):
            if 'counts' not in seg or 'size' not in seg:
                invalid_segmentations.append(ann['id'])
        else:
            invalid_segmentations.append(ann['id'])

    if not invalid_segmentations:
        print("Segmentation format check passed: All segmentations are in valid format.")
        return True
    else:
        print("Invalid segmentation formats found in annotations for annotation IDs:")
        for ann_id in invalid_segmentations:
            print(" ", ann_id)
        return False


def check_image_sizes(images_dir, ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    mismatched_sizes = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if width != img_info['width'] or height != img_info['height']:
                    mismatched_sizes.append((img_info['file_name'], (img_info['width'], img_info['height']), (width, height)))
        except Exception as e:
            print(f"Error opening image {img_info['file_name']}: {e}")

    if not mismatched_sizes:
        print("Image sizes check passed: All image sizes match annotations.")
        return True
    else:
        print("Image size mismatches found:")
        for fname, ann_size, real_size in mismatched_sizes:
            print(f" {fname}: Annotation size {ann_size}, Real size {real_size}")
        return False


def check_all_annotations_exist(images_dir, ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    img_ids = set(coco.getImgIds())
    annotations = coco.loadAnns(coco.getAnnIds())

    invalid_annotations = [ann['id'] for ann in annotations if ann['image_id'] not in img_ids]

    if not invalid_annotations:
        print("All annotations reference existing images.")
        return True
    else:
        print("Annotations referencing non-existing images found for annotation IDs:")
        for ann_id in invalid_annotations:
            print(" ", ann_id)
        return False


def check_image_ids_disjoint(ann_files, split_names):
    image_ids_per_split = {}
    for ann_file, split_name in zip(ann_files, split_names):
        if not os.path.exists(ann_file):
            print(f"Annotation file not found: {ann_file}")
            return False
        with open(ann_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        image_ids = set(coco.getImgIds())
        image_ids_per_split[split_name] = image_ids
        print(f"{split_name}: {len(image_ids)} image_ids")

    all_splits = list(image_ids_per_split.keys())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            split1 = all_splits[i]
            split2 = all_splits[j]
            overlap = image_ids_per_split[split1].intersection(image_ids_per_split[split2])
            if overlap:
                print(f"Overlap found between {split1} and {split2}: {len(overlap)} overlapping image_ids")
                print(f"Example overlapping image_ids: {list(overlap)[:5]}")
                return False
            else:
                print(f"No overlap between {split1} and {split2}")
    print("All image_ids across splits are unique and do not overlap.")
    return True


def visualize_dataset_statistics(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    categories = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in categories]

    annotations = coco.loadAnns(coco.getAnnIds())
    num_annotations = len(annotations)
    num_images = len(coco.getImgIds())

    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")

    category_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [coco.loadCats(cat_id)[0]['name'] for cat_id, _ in categories_sorted]
    counts = [count for _, count in categories_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel('Категории')
    plt.ylabel('Количество аннотаций')
    plt.title('Распределение категорий в датасете')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def check_custom_coco_dataset(images_dir, ann_file, num_samples=5, train_mode=False, random_seed=42):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    dataset = CustomCocoDataset(images_dir=images_dir, ann_file=ann_file, transforms=get_transform(train=train_mode))

    if len(dataset) == 0:
        print("No images found in dataset.")
        return

    random.seed(random_seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, target = dataset[idx]

        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if "masks" in target and target["masks"].shape[0] > 0:
            combined_mask = target["masks"].sum(dim=0).cpu().numpy()
            combined_mask = np.clip(combined_mask, 0, 1)
        else:
            combined_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_np)
        axs[0].set_title("Transformed Image")
        axs[0].axis('off')

        axs[1].imshow(img_np)
        axs[1].imshow(combined_mask, cmap='jet', alpha=0.5)
        axs[1].set_title("Image with Masks")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()


def check_predictions_image_ids(pred_file, ann_file):
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return False
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    gt_image_ids = set(coco_gt.getImgIds())

    pred_image_ids = set(map(int, predictions.keys()))
    missing_ids = pred_image_ids - gt_image_ids

    if not missing_ids:
        print("All predicted image_ids are present in ground truth.")
        return True
    else:
        print(f"Some predicted image_ids are missing in ground truth: {len(missing_ids)}")
        print(f"Example missing image_ids: {list(missing_ids)[:5]}")
        return False


def check_train_and_test(train_images_dir, train_ann_file, test_images_dir, test_ann_file):
    print("=== Checking Training Set ===")
    integrity_train = check_dataset_integrity(train_images_dir, train_ann_file)
    categories_train = check_categories(train_ann_file)
    duplicates_train = check_duplicate_image_ids(train_ann_file)
    annotation_fields_train = check_annotation_fields(train_ann_file)
    bbox_format_train = check_bbox_format(train_ann_file)
    segmentation_format_train = check_segmentation_format(train_ann_file)
    image_sizes_train = check_image_sizes(train_images_dir, train_ann_file)
    annotations_exist_train = check_all_annotations_exist(train_images_dir, train_ann_file)

    print("\n=== Checking Test Set ===")
    integrity_test = check_dataset_integrity(test_images_dir, test_ann_file)
    categories_test = check_categories(test_ann_file)
    duplicates_test = check_duplicate_image_ids(test_ann_file)
    annotation_fields_test = check_annotation_fields(test_ann_file)
    bbox_format_test = check_bbox_format(test_ann_file)
    segmentation_format_test = check_segmentation_format(test_ann_file)
    image_sizes_test = check_image_sizes(test_images_dir, test_ann_file)
    annotations_exist_test = check_all_annotations_exist(test_images_dir, test_ann_file)

    all_checks = [
        integrity_train, categories_train, duplicates_train,
        annotation_fields_train, bbox_format_train, segmentation_format_train,
        image_sizes_train, annotations_exist_train,
        integrity_test, categories_test, duplicates_test,
        annotation_fields_test, bbox_format_test, segmentation_format_test,
        image_sizes_test, annotations_exist_test
    ]

    if all(all_checks):
        print("\nAll dataset checks passed successfully.")
    else:
        print("\nSome dataset checks failed. Please review the above messages.")


def check_all_splits(train_images_dir, train_ann_file, test_images_dir, test_ann_file, use_augmented=False, augmented_train_ann_file=None, augmented_test_ann_file=None):
    if use_augmented:
        if not augmented_train_ann_file or not augmented_test_ann_file:
            print("Please provide augmented annotation files.")
            return
        print("=== Checking Augmented Training Set ===")
        integrity_aug_train = check_dataset_integrity(train_images_dir, augmented_train_ann_file)
        categories_aug_train = check_categories(augmented_train_ann_file)
        duplicates_aug_train = check_duplicate_image_ids(augmented_train_ann_file)
        annotation_fields_aug_train = check_annotation_fields(augmented_train_ann_file)
        bbox_format_aug_train = check_bbox_format(augmented_train_ann_file)
        segmentation_format_aug_train = check_segmentation_format(augmented_train_ann_file)
        image_sizes_aug_train = check_image_sizes(train_images_dir, augmented_train_ann_file)
        annotations_exist_aug_train = check_all_annotations_exist(train_images_dir, augmented_train_ann_file)

        print("\n=== Checking Augmented Test Set ===")
        integrity_aug_test = check_dataset_integrity(test_images_dir, augmented_test_ann_file)
        categories_aug_test = check_categories(augmented_test_ann_file)
        duplicates_aug_test = check_duplicate_image_ids(augmented_test_ann_file)
        annotation_fields_aug_test = check_annotation_fields(augmented_test_ann_file)
        bbox_format_aug_test = check_bbox_format(augmented_test_ann_file)
        segmentation_format_aug_test = check_segmentation_format(augmented_test_ann_file)
        image_sizes_aug_test = check_image_sizes(test_images_dir, augmented_test_ann_file)
        annotations_exist_aug_test = check_all_annotations_exist(test_images_dir, augmented_test_ann_file)

        all_aug_checks = [
            integrity_aug_train, categories_aug_train, duplicates_aug_train,
            annotation_fields_aug_train, bbox_format_aug_train, segmentation_format_aug_train,
            image_sizes_aug_train, annotations_exist_aug_train,
            integrity_aug_test, categories_aug_test, duplicates_aug_test,
            annotation_fields_aug_test, bbox_format_aug_test, segmentation_format_aug_test,
            image_sizes_aug_test, annotations_exist_aug_test
        ]

        print("\n=== Checking image_id uniqueness across splits ===")
        ann_files = [train_ann_file, test_ann_file, augmented_train_ann_file, augmented_test_ann_file]
        split_names = ['train', 'test', 'augmented_train', 'augmented_test']
        image_ids_disjoint = check_image_ids_disjoint(ann_files, split_names)

        if all(all_aug_checks) and image_ids_disjoint:
            print("\nAll augmented dataset checks passed successfully and image_ids are unique across splits.")
        else:
            print("\nSome augmented dataset checks failed or image_ids are not unique across splits. Please review the above messages.")

    else:
        print("=== Checking Training and Test Sets ===")
        integrity_train = check_dataset_integrity(train_images_dir, train_ann_file)
        categories_train = check_categories(train_ann_file)
        duplicates_train = check_duplicate_image_ids(train_ann_file)
        annotation_fields_train = check_annotation_fields(train_ann_file)
        bbox_format_train = check_bbox_format(train_ann_file)
        segmentation_format_train = check_segmentation_format(train_ann_file)
        image_sizes_train = check_image_sizes(train_images_dir, train_ann_file)
        annotations_exist_train = check_all_annotations_exist(train_images_dir, train_ann_file)

        print("\n=== Checking Test Set ===")
        integrity_test = check_dataset_integrity(test_images_dir, test_ann_file)
        categories_test = check_categories(test_ann_file)
        duplicates_test = check_duplicate_image_ids(test_ann_file)
        annotation_fields_test = check_annotation_fields(test_ann_file)
        bbox_format_test = check_bbox_format(test_ann_file)
        segmentation_format_test = check_segmentation_format(test_ann_file)
        image_sizes_test = check_image_sizes(test_images_dir, test_ann_file)
        annotations_exist_test = check_all_annotations_exist(test_images_dir, test_ann_file)

        all_checks = [
            integrity_train, categories_train, duplicates_train,
            annotation_fields_train, bbox_format_train, segmentation_format_train,
            image_sizes_train, annotations_exist_train,
            integrity_test, categories_test, duplicates_test,
            annotation_fields_test, bbox_format_test, segmentation_format_test,
            image_sizes_test, annotations_exist_test
        ]

        print("\n=== Checking image_id uniqueness across splits ===")
        ann_files = [train_ann_file, test_ann_file]
        split_names = ['train', 'test']
        image_ids_disjoint = check_image_ids_disjoint(ann_files, split_names)

        if all(all_checks) and image_ids_disjoint:
            print("\nAll dataset checks passed successfully and image_ids are unique across splits.")
        else:
            print("\nSome dataset checks failed or image_ids are not unique across splits. Please review the above messages.")


def visualize_dataset_statistics(ann_file):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    categories = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in categories]

    annotations = coco.loadAnns(coco.getAnnIds())
    num_annotations = len(annotations)
    num_images = len(coco.getImgIds())

    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")

    category_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [coco.loadCats(cat_id)[0]['name'] for cat_id, _ in categories_sorted]
    counts = [count for _, count in categories_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel('Категории')
    plt.ylabel('Количество аннотаций')
    plt.title('Распределение категорий в датасете')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def check_custom_coco_dataset(images_dir, ann_file, num_samples=5, train_mode=False, random_seed=42):
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    dataset = CustomCocoDataset(images_dir=images_dir, ann_file=ann_file, transforms=get_transform(train=train_mode))

    if len(dataset) == 0:
        print("No images found in dataset.")
        return

    random.seed(random_seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, target = dataset[idx]

        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if "masks" in target and target["masks"].shape[0] > 0:
            combined_mask = target["masks"].sum(dim=0).cpu().numpy()
            combined_mask = np.clip(combined_mask, 0, 1)
        else:
            combined_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_np)
        axs[0].set_title("Transformed Image")
        axs[0].axis('off')

        axs[1].imshow(img_np)
        axs[1].imshow(combined_mask, cmap='jet', alpha=0.5)
        axs[1].set_title("Image with Masks")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()


def check_predictions_image_ids(pred_file, ann_file):
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return False
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    with open(ann_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    gt_image_ids = set(coco_gt.getImgIds())

    pred_image_ids = set(map(int, predictions.keys()))
    missing_ids = pred_image_ids - gt_image_ids

    if not missing_ids:
        print("All predicted image_ids are present in ground truth.")
        return True
    else:
        print(f"Some predicted image_ids are missing in ground truth: {len(missing_ids)}")
        print(f"Example missing image_ids: {list(missing_ids)[:5]}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Integrity Checker")
    parser.add_argument('--train_images_dir', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--train_ann_file', type=str, required=True, help='Path to training annotations file (JSON)')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--test_ann_file', type=str, required=True, help='Path to test annotations file (JSON)')
    parser.add_argument('--use_augmented', action='store_true', help='Check augmented datasets')
    parser.add_argument('--aug_train_ann_file', type=str, help='Path to augmented training annotations file (JSON)')
    parser.add_argument('--aug_test_ann_file', type=str, help='Path to augmented test annotations file (JSON)')
    parser.add_argument('--visualize', action='store_true', help='Visualize dataset statistics')
    parser.add_argument('--pred_file', type=str, help='Path to predictions file (JSON) to check image_ids')

    args = parser.parse_args()

    if args.use_augmented:
        if not args.aug_train_ann_file or not args.aug_test_ann_file:
            parser.error("--use_augmented requires --aug_train_ann_file and --aug_test_ann_file.")

    check_all_splits(
        train_images_dir=args.train_images_dir,
        train_ann_file=args.train_ann_file,
        test_images_dir=args.test_images_dir,
        test_ann_file=args.test_ann_file,
        use_augmented=args.use_augmented,
        augmented_train_ann_file=args.aug_train_ann_file,
        augmented_test_ann_file=args.aug_test_ann_file
    )

    if args.visualize:
        print("\n=== Visualizing Dataset Statistics ===")
        visualize_dataset_statistics(args.train_ann_file)
        visualize_dataset_statistics(args.test_ann_file)
        if args.use_augmented:
            visualize_dataset_statistics(args.aug_train_ann_file)
            visualize_dataset_statistics(args.aug_test_ann_file)

    if args.pred_file:
        print("\n=== Checking Predictions Image IDs ===")
        check_predictions_image_ids(args.pred_file, args.test_ann_file)


if __name__ == "__main__":
    main()
