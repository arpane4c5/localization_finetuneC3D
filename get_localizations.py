#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 03:25:30 2018

@author: hadoop
"""
import torch
import json
import os
import pickle

# Local Paths
LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

BATCH_SIZE = 20


def getLocalizations(val_keys, predictions, batchsize, threshold=0.5, seq_thr=0.5):
    """
    Get frame localizations for actions of interest, in the format as defined in the
    labels dataset. Write the dictionary and evaluate using TIoU evaluations 
    criteria.
    val_keys: list of batches with key values (repetitions for same videos)
    predictions: list of batch predictions. 
    Each batch has B  Seq_len sized (B X Seq_len) FloatTensors
    Length of both the above lists is the same
    """
    localizations = {}
    vid_preds = []
    seq_len = int(predictions[0].shape[0]/batchsize)  # 180L/20 = 9
    
    # concat all the predictions along the 1st axis
    # returns a (202473L, ) i.e., B-1 x 180 + last incomplete batchsize
    predictions = torch.cat((predictions), 0)   
    val_keys = [key for batch_keys in val_keys for key in batch_keys]
    seq_beg, seq_end = 0, seq_len
    
    # Convert into binary predictions using a threshold
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    
    # Iterate over the val_keys (now a list)
    for i, key in enumerate(val_keys):      # one key takes seq_len values
        # 
        if i == 0:
            prev_key = key
        else:
            if prev_key != key:     # change video 
                #len(vid_preds) = len(vid) - seq_len 
                # Append seq_len zeros to beginning only
                vid_preds = [0]*seq_len + vid_preds
                localizations[prev_key] = getVidLocalizations(vid_preds)
                vid_preds = []
                prev_key = key
                
        if torch.sum(predictions[seq_beg:seq_end])>(seq_len*seq_thr):
            vid_preds.append(1)
        else:
            vid_preds.append(0)
        
        seq_beg = seq_end
        seq_end = seq_end + seq_len
        
        #if len(batch_keys_unique) == 1:    # all the values are the same
        #      batch_keys_unique[0]
    
    # For last video
    if len(vid_preds) != 0 and key not in list(localizations.keys()):
        vid_preds = [0]*seq_len + vid_preds
        localizations[key] = getVidLocalizations(vid_preds)
    
    return localizations
    
    
def getVidLocalizations(binaryPreds): 
    """
    Receive a list of binary predictions and generate the action localizations
    for a video
    """
    vLocalizations = []
    act_beg, act_end = -1, -1
    isAction = False
    for i,pred in enumerate(binaryPreds):
        if not isAction:
            if pred == 1:  # Transition from non-action to action
                isAction = True
                act_beg = i
            # if p==0: # skip since it is already non-action
        else:           # if action is going on
            if pred == 0:    # Transition from action to non-action
                isAction = False
                act_end = (i-1)
                # Append to actions list
                vLocalizations.append((act_beg, act_end))
                act_beg, act_end = -1, -1   # Reset
                
    if isAction and act_beg != -1:
        act_end = len(binaryPreds) -1
        vLocalizations.append((act_beg, act_end))
        
    return vLocalizations


if __name__ == "__main__":
    
    destFile = "pred_localizations.json"
    threshold = 0.5
    seq_threshold = 0.6
    
    with open("predictions.pkl", "rb") as fp:
        predictions = pickle.load(fp)
    
    with open("val_keys.pkl", "rb") as fp:
        val_keys = pickle.load(fp)
    localization_dict = getLocalizations(val_keys, predictions, BATCH_SIZE, \
                                         threshold, seq_threshold)
    
    print(localization_dict)
    