import json

AUG_ANN_FILE = 'data_seg_augmented/annotations_train.json'

with open(AUG_ANN_FILE, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

missing_segmentation = [ann for ann in dataset['annotations'] if 'segmentation' not in ann]
if missing_segmentation:
    print(f"Найдены {len(missing_segmentation)} аннотаций без 'segmentation':")
    for ann in missing_segmentation:
        print(f"ID аннотации: {ann['id']}, Image ID: {ann['image_id']}")
else:
    print("Все аннотации содержат поле 'segmentation'.")
