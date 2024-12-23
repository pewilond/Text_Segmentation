from dataset import CustomDataset
from transform import get_transform
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_sample(image, target, idx):
    image_np = image.permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    ax = plt.gca()

    boxes = target["boxes"].numpy()
    labels = target["labels"].numpy()

    for box, label in zip(boxes, labels):
        if label == 0:
            continue
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    
    plt.title(f"Sample {idx} - Boxes")
    plt.axis('off')
    plt.show()

def visualize_masks(image, target, idx):
    image_np = image.permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    ax = plt.gca()

    masks = target["masks"].numpy()

    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        color = np.random.rand(3)
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
        colored_mask[..., :3] = color
        colored_mask[..., 3] = 0.4 * mask.astype(np.float32)  
        ax.imshow(colored_mask)

    plt.title(f"Sample {idx} - Masks")
    plt.axis('off')
    plt.show()

def validate_dataset(images_dir, ann_file, num_samples=5):
    dataset = CustomDataset(
        images_dir=images_dir,
        ann_file=ann_file,
        transforms=get_transform(train=True)
    )
    
    for idx in range(num_samples):
        image, target = dataset[idx]
        print(f"Sample {idx}:")
        print(f"Image shape: {image.shape}")
        print(f"Boxes: {target['boxes']}")
        print(f"Labels: {target['labels']}")
        print(f"Masks shape: {target['masks'].shape}")
        print(f"Image ID: {target['image_id']}")

        visualize_sample(image, target, idx)

        visualize_masks(image, target, idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate Custom COCO Dataset')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to annotation file (JSON)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to validate')
    args = parser.parse_args()

    validate_dataset(args.images_dir, args.ann_file, args.num_samples)
