from torchvision.transforms import functional as F
import torch
from PIL import Image
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size 

    def __call__(self, image, target=None):
       
        orig_width, orig_height = image.size

        image = F.resize(image, self.size)
        new_height, new_width = self.size

        if target is not None:
            if "boxes" in target:
                boxes = target["boxes"]
                scale_x = new_width / orig_width
                scale_y = new_height / orig_height
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
                target["boxes"] = boxes

            if "masks" in target:
                masks = target["masks"].float()
                masks = F.resize(masks, self.size, interpolation=Image.NEAREST)
                masks = masks.byte()
                target["masks"] = masks

        return image, target

def get_transform(train = None):
    transforms = []
    transforms.append(Resize((224, 224)))
    transforms.append(ToTensor())
    return Compose(transforms)
