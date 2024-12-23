import os
import time
import argparse

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou

from dataset import CustomDataset
from transform import get_transform
from tqdm import tqdm


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
    return sum(ious)/len(ious) if len(ious) > 0 else 0.0


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    ious_boxes = []
    ious_masks = []
    for images, targets in tqdm(data_loader, desc="Eval"):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for output, target in zip(outputs, targets):
            iou_b = compute_mean_iou_for_boxes(output, target)
            ious_boxes.append(iou_b)
            if "masks" in output and len(output["masks"]) > 0:
                iou_m = compute_mean_iou_for_masks(output, target)
                ious_masks.append(iou_m)

    mean_iou_boxes = sum(ious_boxes)/len(ious_boxes) if len(ious_boxes) > 0 else 0.0
    mean_iou_masks = sum(ious_masks)/len(ious_masks) if len(ious_masks) > 0 else 0.0
    return mean_iou_boxes, mean_iou_masks


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += loss_value

        if i % print_freq == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], loss: {loss_value:.4f}, elapsed: {elapsed:.2f}s")

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} done. Average loss: {avg_loss:.4f}")
    return avg_loss


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    dataset_train = CustomDataset(images_dir=args.train_images_dir,
                                  ann_file=args.train_ann_file,
                                  transforms=get_transform(train=True))

    dataset_val = CustomDataset(images_dir=args.test_images_dir,
                                ann_file=args.test_ann_file,
                                transforms=get_transform(train=False))
    
    indecies = range(80)

    dataset_val = Subset(dataset_val, indecies)


    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0,
                                   collate_fn=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    best_iou = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_file = os.path.join(args.save_dir, "metrics.txt")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        mean_iou_boxes, mean_iou_masks = evaluate(model, data_loader_val, device)

        with open(metrics_file, 'a') as f:
            f.write(f"Epoch {epoch}: Loss={train_loss:.4f}, IoU_boxes={mean_iou_boxes:.4f}, IoU_masks={mean_iou_masks:.4f}\n")

        if mean_iou_masks > best_iou:
            best_iou = mean_iou_masks
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_epoch_{epoch}.pth"))
            print(f"New best model saved at epoch {epoch} with mean IoU masks: {best_iou:.4f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN with separate train and val datasets")
    parser.add_argument('--train_images_dir', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--train_ann_file', type=str, required=True, help='Path to training annotation file (JSON)')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to validation images directory')
    parser.add_argument('--test_ann_file', type=str, required=True, help='Path to validation annotation file (JSON)')
    parser.add_argument('--save_dir', type=str, default='models/models', help='Directory to save models and metrics')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    main(args)
