# Evaluating Uncertainty Calibration in Object Detection

## Introduction

This page presents the code for the uncertaintry calibration metrics proposed in the paper "A Theoretical and Practical Framework for Evaluating Uncertainty Calibration in Object Detection" (pre-print version availabe at https://arxiv.org/abs/2309.00464): Quadratic Global Calibration (QGC) score, Spherical Global Calibration (SGC) score and Expected Global Calibration Error (EGCE). Additionally code for the Detection Expected Calibration Error (D-ECE) [1] is also available for comparison.

## Instructions

The metrics are in the file _metrics.py_, while the file _utils_.py_ has some necessary additional code. To apply the metrics, the input needed is the directory of a folder with all the ground-truth labels (1st argument, _i.e._ _labels_folder_) and the directory of a folder with all the predicted bounding boxes (2nd argument, _i.e._ _preds_folder_); these folders must be in the common YOLOv5 format. The output of each metric is in the format (_total,avg_) where _total_ represent the absolute value of the metrcis and _avg_ the average value (_i.e._ divided by the total number of bounding-box detections).

## References

[1] Kuppers, F., Kronenberger, J., Shantia, A., & Haselhoff, A. (2020). Multivariate confidence calibration for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (pp. 326-327).
