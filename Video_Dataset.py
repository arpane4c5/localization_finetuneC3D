#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:09:32 2018

@author: Arpan
Refer: https://github.com/hunkim/PyTorchZeroToAll/blob/master/name_dataset.py
"""

# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import json
import os


class VideoDataset(Dataset):
    """ Cricket Strokes dataset."""

    # Initialize your data, download, etc.
    # vidsList: List of paths to labels files containing action instances
    def __init__(self, vidsList, vidsSizes, seq_size=10, is_train_set=False):
        
        #filename = './data/names_train.csv.gz' if is_train_set else './data/names_test.csv.gz'
        
        # read files and get training, testing and validation partitions
        #filesList = os.listdir(labelsPath)
        self.shots = [] # will contain list of dictionaries with vidKey:LabelTuples
        for i, labfile in enumerate(vidsList):
            assert os.path.exists(labfile), "Path does not exist"
            with open(labfile, 'r') as fobj:
                self.shots.append(json.load(fobj))
                
        
        # Check seq_size is valid ie., less than min action size
        
        self.keys = []
        self.frm_sequences = []
        self.labels = []
        for i, labfile in enumerate(vidsList):     
            k = list(self.shots[i].keys())[0] #key value for the ith dict
            pos = self.shots[i][k]  # get list of tuples for k
            pos.reverse()   # reverse list and keep popping
            
            # (start, end) frame no
            self.frm_sequences.extend([(t, t+seq_size-1) for t in \
                                       range(vidsSizes[i]-seq_size+1)])
            # file names (without full path), only keys
            self.keys.extend([k]*(vidsSizes[i]-seq_size+1))

            # Add labels for training set only
            (start, end) = (-1, -1)
            # Get the label
            if len(pos)>0:
                (start, end) = pos.pop()
            # Iterate over the list of tuples and form labels for each sequence
            for t in range(vidsSizes[i]-seq_size+1):
                if t <= (start-seq_size):
                    self.labels.append([0]*seq_size)   # all 0's 
                elif t < start:
                    self.labels.append([0]*(start-t)+[1]*(t+seq_size-start))
                elif t <= (end+1 - seq_size):       # all 1's
                    self.labels.append([1]*seq_size)
                elif t <= end:
                    self.labels.append([1]*(end+1-t) + [0]*(t+seq_size-(end+1)) )
                else:
                    if len(pos) > 0:
                        (start, end) = pos.pop()
                        if t <= (start-seq_size):
                            self.labels.append([0]*seq_size)
                        elif t < start:
                            self.labels.append([0]*(start-t) + [1]*(t+seq_size-start))
                        elif t <= (end+1 - seq_size):       # Check if more is needed
                            self.labels.append([1]*seq_size)
                    else:
                        # For last part with non-action frames
                        self.labels.append([0]*seq_size)
                    
            #if is_train_set:
                # remove values with transitions eg (1, 9), (8, 2) etc
                # Keep only (0, 10) or (10, 0) ie., single action sequences
                
        
        self.videosList = vidsList
        self.len = len(self.keys)
        self.seq_size = seq_size

    def __getitem__(self, index):
        return self.keys[index], self.frm_sequences[index], self.labels[index]

    def __len__(self):
        return self.len

    def __seq_size__(self):
        return self.seq_size
    
