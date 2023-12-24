import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch
import sys
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--pred_path', default="submit.json", type=str)
    parser.add_argument('--target_path', default="../../dataset/cvpdl_gligen_mix/annotations/val.json", type=str)
    
    return parser.parse_args()

args = get_args_parser()
pred_path = args.pred_path
target_path = args.target_path

print("Pred path : {}; Target path: {}".format(pred_path, target_path))

def get_gt_data(imageID, annotations):
    gt_data = {'boxes':[],
               'labels':[],
               'image_id':[],
               'area':[],
               'iscrowd':[]}
    
    for annotation in annotations:
        if annotation["image_id"] == imageID:
            bbox = annotation['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_data['boxes'].append(bbox)
            gt_data['labels'].append(annotation['category_id'])
            gt_data['image_id'].append(imageID)
            gt_data['area'].append(annotation['area'])
            gt_data['iscrowd'].append(annotation['iscrowd'])
    return gt_data

with open(pred_path, 'r') as f:
    preds = json.load(f)

with open(target_path, 'r') as f:
    gt = json.load(f)
fname_to_imageID = {}

for image in gt["images"]:
    fname_to_imageID[image["file_name"]] = image["id"]
metric = MeanAveragePrecision()
device = 'cuda'
device = "cpu"
for fname, pred in tqdm(preds.items()):
    pred = [{k: torch.tensor(v).to(device) for k, v in pred.items()}]
    imageID = fname_to_imageID[fname]
    target = get_gt_data(imageID, gt["annotations"])
    target = [{k: torch.tensor(v).to(device) for k, v in target.items()}]
    metric.update(pred, target)
result = metric.compute()
print(result)
