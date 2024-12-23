import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from transform import get_transform

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou


def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def compute_mean_iou_for_boxes(output, target):
    pred_boxes = output["boxes"].cpu()
    gt_boxes = target["boxes"].cpu()

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    max_ious_per_pred = iou_matrix.max(dim=1)[0]
    mean_iou = max_ious_per_pred.mean().item()
    return mean_iou


def mask_to_max_iou(pred_mask, gt_masks):
    pred_area = pred_mask.sum().item()
    best_iou = 0.0
    for gm in gt_masks:
        inter = (pred_mask & gm).sum().item()
        union = pred_area + gm.sum().item() - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
    return best_iou


def compute_mean_iou_for_masks(output, target):
    pred_masks = (output["masks"].cpu() > 0.5)
    gt_masks = target["masks"].cpu()

    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0

    ious = []
    for pm in pred_masks:
        iou = mask_to_max_iou(pm, gt_masks)
        ious.append(iou)
    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    ious_boxes = []
    ious_masks = []

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            iou_b = compute_mean_iou_for_boxes(output, target)
            ious_boxes.append(iou_b)

            if "masks" in output and len(output["masks"]) > 0:
                iou_m = compute_mean_iou_for_masks(output, target)
                ious_masks.append(iou_m)

    mean_iou_boxes = sum(ious_boxes) / len(ious_boxes) if len(ious_boxes) > 0 else 0.0
    mean_iou_masks = sum(ious_masks) / len(ious_masks) if len(ious_masks) > 0 else 0.0

    return mean_iou_boxes, mean_iou_masks


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_test = CustomDataset(
        images_dir=args.test_images_dir,
        ann_file=args.test_annotations_dir,
        transforms=get_transform(train=False)
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_instance_segmentation_model(num_classes=args.num_classes)
    model.to(device)

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model weights from {args.model_path}")

    mean_iou_boxes, mean_iou_masks = evaluate(model, data_loader_test, device)

    print(f"Mean IoU (Boxes): {mean_iou_boxes:.4f}")
    print(f"Mean IoU (Masks): {mean_iou_masks:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN on a test dataset")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model (e.g., best_model_epoch_X.pth)")
    parser.add_argument("--test_images_dir", type=str, required=True,
                        help="Path to the test/validation images directory")
    parser.add_argument("--test_annotations_dir", type=str, required=True,
                        help="Path to the test/validation annotation JSON file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (default=1)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes including background (default=2)")

    args = parser.parse_args()
    main(args)
