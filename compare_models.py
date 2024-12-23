import os
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from train.train_mask_rcnn_seg.dataset import CustomDataset
from train.train_mask_rcnn_seg.transform import get_transform

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes, model_path, device):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def visualize_predictions(
    img,
    boxes,
    labels,
    masks,
    scores,
    score_thresh,
    save_path,
    title,
    show_plot
):
    img_np = img.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis('off')

    if boxes is not None and len(boxes) > 0:
        b = boxes.cpu().numpy()
        l = labels.cpu().numpy()
        s = scores.cpu().numpy() if scores is not None else None
        for i, box in enumerate(b):
            sc = s[i] if s is not None else 1.0
            if sc < score_thresh:
                continue
            if l[i] == 0:
                continue
            x1, y1, x2, y2 = box
            r = matplotlib.patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(r)

    if masks is not None and len(masks) > 0:
        m = masks.cpu().numpy()
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0, :, :]
        m = (m > 0.5)
        s = scores.cpu().numpy() if scores is not None else None
        for i, mask in enumerate(m):
            sc = s[i] if s is not None else 1.0
            if sc < score_thresh:
                continue
            if not mask.any():
                continue
            color = np.random.rand(3)
            cm = np.zeros((*mask.shape, 4), dtype=np.float32)
            cm[..., :3] = color
            cm[..., 3] = 0.4 * mask.astype(np.float32)
            ax.imshow(cm)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_aug_path", type=str, required=True)
    parser.add_argument("--model_non_aug_path", type=str, required=True)
    parser.add_argument("--test_images_dir", type=str, required=True)
    parser.add_argument("--test_ann_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="compare_results")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    model_aug = get_instance_segmentation_model(
        num_classes=2,
        model_path=args.model_aug_path,
        device=device
    )
    model_non_aug = get_instance_segmentation_model(
        num_classes=2,
        model_path=args.model_non_aug_path,
        device=device
    )

    dataset_test = CustomDataset(
        images_dir=args.test_images_dir,
        ann_file=args.test_ann_file,
        transforms=get_transform(train=False)  
    )

    os.makedirs(args.save_dir, exist_ok=True)

    num_samples = min(args.num_samples, len(dataset_test))
    images_list = []
    targets_list = []

    for i in range(num_samples):
        img, tgt = dataset_test[i]
        images_list.append(img)
        targets_list.append(tgt)

    preds_aug = []
    preds_non_aug = []

    with torch.no_grad():
        for img in images_list:
            p = model_aug([img.to(device)])[0]
            preds_aug.append(p)

        for img in images_list:
            p = model_non_aug([img.to(device)])[0]
            preds_non_aug.append(p)

    for i in range(num_samples):
        image_id = int(targets_list[i]["image_id"].item())

        save_path_aug = os.path.join(
            args.save_dir,
            f"{image_id}_img{i}_aug.jpg"
        )
        visualize_predictions(
            images_list[i],
            preds_aug[i]["boxes"],
            preds_aug[i]["labels"],
            preds_aug[i]["masks"],
            preds_aug[i]["scores"],
            args.score_thresh,
            save_path_aug,
            f"AUG Model Predictions (image {i})",
            args.show_plot
        )

        save_path_non_aug = os.path.join(
            args.save_dir,
            f"{image_id}_img{i}_non_aug.jpg"
        )
        visualize_predictions(
            images_list[i],
            preds_non_aug[i]["boxes"],
            preds_non_aug[i]["labels"],
            preds_non_aug[i]["masks"],
            preds_non_aug[i]["scores"],
            args.score_thresh,
            save_path_non_aug,
            f"Non-AUG Model Predictions (image {i})",
            args.show_plot
        )


if __name__ == "__main__":
    main()
