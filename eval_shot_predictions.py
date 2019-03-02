#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:56:15 2018

@author: Arpan

@Description: Evaluation script for cricket shots.
Compute the Temporal IoU metric for the labeled cricket shots for the test set sample
Refer: ActivityNet localization evaluations script. Here only single action is defined
therefore, instead of mean tIoU, we take tIoU.
"""

import json
import os

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

# Take the predictions dict and iterate and get the gt using the keys of the truth kept inside the folder
def calculate_tIoU(gt_dir, shots_dict):
    # Iterate over the gt files and collect labels into a gt dictionary
    tot_tiou = 0
    tot_segments = 0
    traversed = 0
    
    # match the val/test keys with the filenames present in the LABELS folder
    for sf in shots_dict.keys():
        k = ((sf.split("/")[1]).rsplit(".", 1)[0]) + ".json"
        labelfile = os.path.join(gt_dir, k)
        
        assert os.path.exists(labelfile), "Label file does not exist."
        with open(labelfile, 'r') as fp:
            vid_gt = json.load(fp)
            
        vid_key = vid_gt.keys()[0]      # only one key in dict is saved
        gt_list = vid_gt[vid_key]       # list of tuples [[preFNum, postFNum], ...]
        test_list = shots_dict[vid_key]
        print "Done "+str(traversed)+" : "+vid_key
        # calculate tiou for the video vid_key
        vid_tiou = get_vid_tiou(gt_list, test_list)
        # vid_tiou weighted with no of ground truth segments
        tot_tiou += (vid_tiou*len(gt_list))
        tot_segments += len(gt_list)
        traversed += 1
    
    print "Total segments : " + str(tot_segments)
    print "Total_tiou (all vids) : " + str(tot_tiou)
    print "Weighted Averaged TIoU  : " + str(tot_tiou/tot_segments)
    
    return (tot_tiou/tot_segments)

# get the tiou value for s = {s1, s2, ..., sN} and s' = {s'1, s'2, ..., s'M}
def get_vid_tiou(gt_list, test_list):
    # calculate the value
    N_gt = len(gt_list)
    M_test = len(test_list)
    if N_gt==0 or M_test==0:
        return 0
    # For all gt shots
    tiou_all_gt = 0
    for gt_shot in gt_list:
        max_gt_shot = 0
        for test_shot in test_list:
            # if segments are not disjoint, i.e., an overlap exists
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_gt_shot = max(max_gt_shot, get_iou(gt_shot, test_shot))
        tiou_all_gt += max_gt_shot
    # For all test shots
    tiou_all_test = 0
    for test_shot in test_list:
        max_test_shot = 0
        for gt_shot in gt_list:
            # if segments are not disjoint, i.e., an overlap exists
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_test_shot = max(max_test_shot, get_iou(gt_shot, test_shot))
        tiou_all_test += max_test_shot
    
    vid_tiou = ((tiou_all_gt/N_gt)+(tiou_all_test/M_test))/2.
    print "TIoU for video : "+str(vid_tiou)
    return vid_tiou

# calculate iou (using frame counts) between two segments
# function is called only when overlap exists
def get_iou(gt_shot, test_shot):
    # if overlap exists
    t = [gt_shot[0], gt_shot[1], test_shot[0], test_shot[1]]
    upper_b = max(t)
    lower_b = min(t)
    union = upper_b - lower_b + 1.0
    t.remove(upper_b)
    t.remove(lower_b)
    intersection = max(t) - min(t) + 1.0  # remaining values
    return (intersection/union)

if __name__ == '__main__':
    # Take 
    pred_shots_file = "predicted_localizations_th0_5_filt60.json"
    with open(pred_shots_file, 'r') as fp:
        shots_dict = json.load(fp)
        
    tiou = calculate_tIoU(LABELS, shots_dict)