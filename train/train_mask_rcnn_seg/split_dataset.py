import os
import json
import shutil
import argparse
import random

def split_dataset(images_dir, ann_file, output_dir, train_ratio=0.8):

    os.makedirs(output_dir, exist_ok=True)
    train_images_dir = os.path.join(output_dir, 'images_train')
    test_images_dir = os.path.join(output_dir, 'images_test')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    random.shuffle(images)
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    test_images = images[train_count:]

    image_id_to_anns = {}
    for ann in annotations:
        image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    train_anns = []
    test_anns = []

    def copy_images_and_annotations(image_list, images_subdir, ann_list):
        for img_info in image_list:
            img_filename = img_info['file_name']
            src_path = os.path.join(images_dir, img_filename)
            dst_path = os.path.join(images_subdir, img_filename)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)
            if img_info['id'] in image_id_to_anns:
                ann_list.extend(image_id_to_anns[img_info['id']])

    copy_images_and_annotations(train_images, train_images_dir, train_anns)
    copy_images_and_annotations(test_images, test_images_dir, test_anns)

    train_coco = {
        "images": train_images,
        "annotations": train_anns,
        "categories": categories
    }
    with open(os.path.join(output_dir, 'annotations_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=4)

    test_coco = {
        "images": test_images,
        "annotations": test_anns,
        "categories": categories
    }
    with open(os.path.join(output_dir, 'annotations_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_coco, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    split_dataset(args.images_dir, args.ann_file, args.output_dir, args.train_ratio)
