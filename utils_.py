# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:35:38 2023

@author: pedro
"""

import torch
import numpy as np
import os
import time

def text_to_torch(txt_file):
    array = np.loadtxt(txt_file)
    if array.ndim == 2:
        return torch.from_numpy(array)
    if array.ndim == 1:
        return torch.unsqueeze(torch.from_numpy(array),0)
        

def IoU(boxes_preds, boxes_labels, box_format="midpoint"):


    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def fill_blank_preds(labels_folder, preds_folder):
    l1 = os.listdir(labels_folder)
    l2 = os.listdir(preds_folder)
    res = [x for x in l1 if x not in l2]
    for file in res:
        with open(f'{preds_folder}\\{file}', 'w') as f:
            f.write('-1 0.0 0.0 0.0 0.0 0.0')
    

def divide_detections(labels_folder, preds_folder, iou_thresh=0.5):
    fill_blank_preds(labels_folder, preds_folder)
    TP = []
    FP = []
    FN_count = 0
    
    for filename in os.listdir(labels_folder):
        
        TP_file = []
        
        labels_file = os.path.join(labels_folder, filename)
        preds_file = os.path.join(preds_folder, filename)
        labels_tensor = text_to_torch(labels_file)
        preds_tensor = text_to_torch(preds_file)
        
        #TRUE POSITIVES and FALSE NEGATIVES
        for i in range(labels_tensor.size(dim=0)):
            idx = -1
            iou = 0
            for j in range(preds_tensor.size(dim=0)):
                if labels_tensor[i,0] == preds_tensor[j,0] and IoU(labels_tensor[i,1:5],preds_tensor[j,1:5]) > iou and IoU(labels_tensor[i,1:5],preds_tensor[j,1:5]) >= iou_thresh:
                    iou = IoU(labels_tensor[i,1:5],preds_tensor[j,1:5])
                    idx = j
            if idx < 0: #if there is no match, add to False Negatives count
                FN_count += 1
            else: #append to True Positives
                TP_file.append(np.array([preds_tensor[idx,5],iou,idx]))
                
        TP.extend(TP_file)

        #FALSE POSITIVES
        for i in range(preds_tensor.size(dim=0)):
            c = -1
            for j in range(len(TP_file)):
                if i == TP_file[j][2]:
                    c = 1
            if c < 0 and preds_tensor[i,0] != -1:
                FP.append(np.array([preds_tensor[i,5]]))
        
    return TP, FP, FN_count





