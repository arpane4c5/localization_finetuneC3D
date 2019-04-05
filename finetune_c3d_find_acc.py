#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 01:53:27 2018

@author: Arpan

@Description: Same as the Finetune code. But load a pretrained model and find the 
accuracies on a sequence of DEPTH values for a finetuned network on val set. 
"""

import os
import torch
import time
import copy
import json
import pickle
import numpy as np
import torch.nn as nn
import utils
import model_c3d_finetune as c3d

from math import fabs
from Video_Dataset import VideoDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from get_localizations import getScoredLocalizations

# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

# Parameters and DataLoaders
BATCH_SIZE = 16     # for 32 out of memory, for 16 it runs
N_EPOCHS = 30
INP_VEC_SIZE = None
SEQ_SIZE = 16   # has to >=16 (ie. the number of frames used for c3d input)
threshold = 0.5
seq_threshold = 0.5
data_dir = "numpy_vids_112x112"
wts_path = "log_half_center_seq23/c3d_finetune_conv5b_FC678_ep30_w23_SGD.pt"
#wts_path = 'log/c3d_finetune_conv5b_FC678_ep10_w16_SGD.pt'




def get_1D_preds(preds):
    preds_new = []
    for pred in preds.data.cpu().numpy():
        # print("i preds", pred)
        idx = np.argmax(pred)
        preds_new.append(idx)
    return np.asarray(preds_new)

def get_accuracy(preds, targets):
    preds_new = get_1D_preds(preds)
    tar_new = targets.data.cpu().numpy()
    # print("preds", preds_new[:5])
    # print("targets", tar_new[:5])
    acc = sum(preds_new == tar_new)*1.0
    return acc


def getBatchFrames(features, videoFiles, sequences):
    """Select only the batch features from the dictionary of features (corresponding
    to the given sequences) and return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    SeqSize should be >= 2 for atleast one vector in sequence.
    """
    #grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        # get key value for the video. Use this to read features from dictionary
        videoFile = videoFile.split('/')[1].rsplit('.', 1)[0]
            
        start_frame = sequences[0][i]   # starting point of sequences in video
        end_frame = sequences[1][i]     # end point
        # Load features
        # (N-1) sized list of vectors of 1152 dim
        vidFeats = features[videoFile]  
        
        # get depth x 112 x 112 x 3 sized input cubiod
        vid_feat_seq = vidFeats[start_frame:(end_frame+1), :, :]
        
        # transpose to Ch x depth x H x W
        #vid_feat_seq = vid_feat_seq.transpose(3, 0, 1, 2)
        #vid_feat_seq = np.squeeze(vid_feat_seq, axis = 1)
        batch_feats.append(vid_feat_seq)
        
    return np.array(batch_feats)


# Inputs: feats: list of lists
def make_variables(feats, labels, use_gpu):
    # Create the input tensors and target label tensors
    # transpose to batch x ch x depth x H x W
    feats = feats.transpose(0, 4, 1, 2, 3)    
    feats = torch.from_numpy(np.float32(feats))
    
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []
    # Append the sequence of labels of len (seq_size-1) to the target list for OF.
    # Iterate over the batch labels, for each extract seq_size labels and extend 
    # in the target list
    
    for i in range(labels[0].size(0)):
        lbls = [y[i] for y in labels]      # get labels of frames (size seq_size)
        # for getting batch x 2 sized matrix, add vectors of size 2
        if sum(lbls)>=8:
            target.append(1)   # action is True
        else:
            target.append(0)

    # Form a wrap into a tensor variable as B X S X I
    # target is a vector of batchsize
    return utils.create_variable(feats, use_gpu), \
            utils.create_variable(torch.LongTensor(target), use_gpu)


if __name__=='__main__':

    seed = 1234
    utils.seed_everything(seed)
    use_gpu = torch.cuda.is_available()
    #####################################################################
    # Form dataloaders 
    
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of Training / Val / Test videos : {} / {} / {}".format(len(train_lst), \
          len(val_lst), len(test_lst)))
    print(60*"-")
    
    # form the names of the list of label files, should be at destination 
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    
    print("Test #VideoFrames : {}".format(val_sizes))
    
    
    #####################################################################
    
    framesPath = os.path.join(os.getcwd(), data_dir)
    
    # read into dictionary {vidname: np array, ...}
    print("Loading validation/test features from disk...")
    valFrames = utils.readAllNumpyFrames(framesPath, val_lst)
        
    #####################################################################
    
    # Load the model
    model = c3d.C3D()
    model.fc8 = nn.Linear(4096, 2)
    # get the network pretrained weights into the model    
    model.load_state_dict(torch.load(wts_path))

    # need to set requires_grad = False for all the layers
    for param in model.parameters():
        param.requires_grad = False
    if use_gpu:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model.cuda()
                
    #####################################################################

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()    
        
    #####################################################################
    #####################################################################
    
    model.eval()
    # Test a video or calculate the accuracy using the learned model
    print("Prediction video meta info.")
    print("Predicting on the validation/test videos...")

    for SEQ_SIZE in range(16, 24):
        print("SEQ_SIZE = {} | Model {}".format(SEQ_SIZE, wts_path))
        
        val_keys = []
        predictions = []
        # create VideoDataset object, create sequences(use meta info)
        hlvalDataset = VideoDataset(val_labs, val_sizes, seq_size=SEQ_SIZE, is_train_set = False)
        
        # total number of training examples (clips)
        print("No. of Test examples : {} ".format(hlvalDataset.__len__()))
        
        # Create a DataLoader object and sample batches of examples. (get meta-info)
        # These batch samples are used to extract the features from videos parallely
        val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # get no. of validation examples
        print("Validation size : {}".format(len(val_loader.dataset)))
    
        start = time.time()
        for i, (keys, seqs, labels) in enumerate(val_loader):
            # Testing on the sample
            batchFeats = getBatchFrames(valFrames, keys, seqs)
            # Validation stage
            inputs, target = make_variables(batchFeats, labels, use_gpu)
            
            output = model(inputs) # of size (BATCHESx2)
            
            #pred_probs = output.view(output.size(0)).data
            pred_probs = output[:,1].data
            
            val_keys.append(keys)
            predictions.append(pred_probs)  # append the 
            if i%400 == 0:
                print("i = {} / {}".format(i, len(hlvalDataset)))
            #if i % 2 == 0:
            #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
    #        if (i+1) % 100 == 0:
    #            break
        end = time.time()
        print("Predictions done on validation/test set...")
        #####################################################################
        
        with open("predictions_seq"+str(SEQ_SIZE)+".pkl", "wb") as fp:
            pickle.dump(predictions, fp)
        
        with open("val_keys_seq"+str(SEQ_SIZE)+".pkl", "wb") as fp:
            pickle.dump(val_keys, fp)
        
    #    with open("predictions.pkl", "rb") as fp:
    #        predictions = pickle.load(fp)
    #    
    #    with open("val_keys.pkl", "rb") as fp:
    #        val_keys = pickle.load(fp)
            
    
        # [4949, 4369, 4455, 4317, 4452]
        #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
        localization_dict = getScoredLocalizations(val_keys, predictions, BATCH_SIZE, \
                                             threshold, seq_threshold)
    
        print(localization_dict)
        
        # Apply filtering    
        i = 60  # optimum
        filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
        #i = 7  # optimum
        #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
        filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+"_seq"+str(SEQ_SIZE)+".json"
        with open(filt_shots_filename, 'w') as fp:
            json.dump(filtered_shots, fp)
        print("Prediction file written to disk !!")
        #####################################################################
        # count no. of parameters in the model
        #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        #params = sum([np.prod(p.size()) for p in model_parameters])
        # call count_paramters(model)  for displaying total no. of parameters
        #print("#Parameters : {} ".format(utils.count_parameters(model)))
        print("Total execution time : "+str(end-start))
