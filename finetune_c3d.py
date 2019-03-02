#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 01:53:27 2018

@author: Arpan

@Description: Finetune a pretrained C3D model model in PyTorch. 
Use the highlight videos dataset and re-trained the FC8 layer.
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
data_dir = "numpy_vids_112x112_sc032"
wts_path = 'c3d.pickle'
#wts_path = 'log/c3d_finetune_conv5b_FC678_ep10_w16_SGD.pt'
#chkpoint = 'c3d_finetune_FC78_ep5_w16_SGD.pt'
mod_name = "log/c3d_finetune_conv5b_FC678_ep"


# takes a model to train along with dataset, optimizer and criterion
def train(trainFrames, valFrames, model, datasets_loader, optimizer, \
          scheduler, criterion, nEpochs, use_gpu):
    global training_stats
    training_stats = defaultdict()
#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0
    
    for epoch in range(nEpochs):
        print("-"*60)
        print("Epoch -> {} ".format((epoch+1)))
        training_stats[epoch] = {}
        # for each epoch train the model and then evaluate it
        for phase in ['train']:
            #print("phase->", phase)
            dataset = datasets_loader[phase]
            training_stats[epoch][phase] = {}
            accuracy = 0
            net_loss = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)
            elif phase == 'test':
                #print("validation")
                model.train(False)
            
            for i, (keys, seqs, labels) in enumerate(dataset):
                
                # return a 16 x ch x depth x H x W 
                if phase == 'train':
                    batchFeats = getBatchFrames(trainFrames, keys, seqs)
                elif phase == 'test':
                    batchFeats = getBatchFrames(valFrames, keys, seqs)
                
                # return the torch.Tensor values for inputs and 
                x, y = make_variables(batchFeats, labels, use_gpu)
                # print("x type", type(x.data))
                
                preds = model(x)
                loss = criterion(preds, y)
                #print(preds, y)
                net_loss += loss.data.cpu().numpy()
                accuracy += get_accuracy(preds, y)
#                print "# Accurate : {}".format(accuracy)
                
#                print("Phase : {} :: Batch : {} :: Loss : {} :: Accuracy : {}"\
#                          .format(phase, (i+1), net_loss, accuracy))
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if (i+1) == 2000:
                    break
#            accuracy = fabs(accuracy)/len(datasets_loader[phase].dataset)
            accuracy = fabs(accuracy)/(BATCH_SIZE*(i+1))
            training_stats[epoch][phase]['loss'] = net_loss
            training_stats[epoch][phase]['acc'] = accuracy
            training_stats[epoch][phase]['lr'] = optimizer.param_groups[0]['lr']
            
        # Display at end of epoch
        print("Phase {} : Train :: Epoch : {} :: Loss : {} :: Accuracy : {} : LR : {}"\
              .format(i, (epoch+1), training_stats[epoch]['train']['loss'],\
                      training_stats[epoch]['train']['acc'], \
                                    optimizer.param_groups[0]['lr']))
#        print("Phase : Test :: Epoch : {} :: Loss : {} :: Accuracy : {}"\
#              .format((epoch+1), training_stats[epoch]['test']['loss'],\
#                      training_stats[epoch]['test']['acc']))
        
        if ((epoch+1)%10) == 0:
            save_model_checkpoint(model, epoch+1, "SGD", win=SEQ_SIZE, use_gpu=use_gpu)

#        s7, s8 = 0, 0
#        for p in model.fc7.parameters():
#            s7 += torch.sum(p)
#            #print(p[...,-3:])
#        for p in model.fc8.parameters():
#            s8 += torch.sum(p)
#            #print(p[...,-3:])
#        print("FC7 Sum : {} :: FC8 Sum : {}".format(s7, s8))

    # Save dictionary after all the epochs
    save_stats_dict(training_stats)
    # Training finished
    return model

def save_model_checkpoint(model, ep, opt, win=16, use_gpu=True):
    # Save only the model params
    name = mod_name+str(ep)+"_w"+str(win)+"_"+opt+".pt"
    if use_gpu and torch.cuda.device_count() > 1:
#        model.conv5a = model.conv5a.module
        model.conv5b = model.conv5b.module
        model.fc6 = model.fc6.module    # good idea to unwrap from DataParallel and save
        model.fc7 = model.fc7.module
        model.fc8 = model.fc8.module
    torch.save(model.state_dict(), name)
    print("Model saved to disk... {}".format(name))
    # Again wrap into DataParallel
    model.fc8 = nn.DataParallel(model.fc8)
    model.fc7 = nn.DataParallel(model.fc7)
    model.fc6 = nn.DataParallel(model.fc6)
    model.conv5b = nn.DataParallel(model.conv5b)

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

# save training stats
def save_stats_dict(stats):
    with open('stats.pickle', 'wb') as fr:
        pickle.dump(stats, fr, protocol=pickle.HIGHEST_PROTOCOL)
    return None

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
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    # get complete path lists of label files
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    
    print("Train #VideoFrames : {}".format(sizes))
    print("Test #VideoFrames : {}".format(val_sizes))
    
    # create VideoDataset object, create sequences(use meta info)
    hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    hlvalDataset = VideoDataset(val_labs, val_sizes, seq_size=SEQ_SIZE, is_train_set = False)
    
    # total number of training examples (clips)
    print("No. of Train examples : {} ".format(hlDataset.__len__()))
    print("No. of Test examples : {} ".format(hlvalDataset.__len__()))
    
    # Create a DataLoader object and sample batches of examples. (get meta-info)
    # These batch samples are used to extract the features from videos parallely
    train_loader = DataLoader(dataset=hlDataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=hlvalDataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # get no. of training examples
    print("Training size : {}".format(len(train_loader.dataset)))
    # get no. of validation examples
    print("Validation size : {}".format(len(val_loader.dataset)))
    
    # dataloaders formed for training and validation sets.
    datasets_loader = {'train': train_loader, 'test': val_loader}
    
    #####################################################################
    
    framesPath = os.path.join(os.getcwd(), data_dir)
    
    # read into dictionary {vidname: np array, ...}
    print("Loading training features from disk...")
    # load np matrices of size N x H x W x Ch (N is #frames, resized to 180 x 320 and 
    # taken center crops of 112 x 112 x 3)
    trainFrames = utils.readAllNumpyFrames(framesPath, train_lst)
    print("Loading validation/test features from disk...")
    valFrames = utils.readAllNumpyFrames(framesPath, val_lst)
        
    #####################################################################
    
    # Load the model
    model = c3d.C3D()
    # get the network pretrained weights into the model
#    model.fc8 = nn.Linear(4096, 2)
    model.load_state_dict(torch.load("../localization_rnn/"+wts_path))
    # need to set requires_grad = False for all the layers
    for param in model.parameters():
        param.requires_grad = False
#    for i in model.fc8.parameters():
#        i.requires_grad = True
#    for i in model.fc7.parameters():
#        i.requires_grad = True
#    for i in model.fc6.parameters():
#        i.requires_grad = True
#    for i in model.conv5b.parameters():
#        i.requires_grad = True
    # reset the last layer (default requires_grad is True)
    model.fc8 = nn.Linear(4096, 2)
    model.fc7 = nn.Linear(4096, 4096)
    model.fc6 = nn.Linear(8192, 4096)
    model.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#    model.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    # Load on the GPU, if available
    if use_gpu:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # Parallely run on multiple GPUs using DataParallel
            model.fc8 = nn.DataParallel(model.fc8)
            model.fc7 = nn.DataParallel(model.fc7)
            model.fc6 = nn.DataParallel(model.fc6)
            model.conv5b = nn.DataParallel(model.conv5b)
#            model.conv5a = nn.DataParallel(model.conv5b)
            model.cuda()
            
        elif torch.cuda.device_count() == 1:
            model.cuda()
    
    #####################################################################

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    
    #optimizer = torch.optim.SGD(model.fc8.parameters(), lr=0.001, momentum=0.9)
#    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
#                                 lr = 0.001)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                                 lr = 0.001, momentum=0.9)
    
    sigm = nn.Sigmoid()
    
    # set the scheduler, optimizer and retrain (eg. SGD)
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    #####################################################################
    
    start = time.time()
    s = start
    
    # Training (finetuning) and validating
    model = train(trainFrames, valFrames, model, datasets_loader, optimizer, \
                     step_lr_scheduler, criterion, nEpochs=N_EPOCHS, use_gpu=use_gpu)
        
    end = time.time()
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    
    #####################################################################
    #####################################################################
    
    val_keys = []
    predictions = []
    model.eval()
    # Test a video or calculate the accuracy using the learned model
    print("Prediction video meta info.")
    print("Predicting on the validation/test videos...")
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
        #if i % 2 == 0:
        #    print('i: {} :: Val keys: {} : seqs : {}'.format(i, keys, seqs)) #keys, pred_probs))
#        if (i+1) % 100 == 0:
#            break
    print("Predictions done on validation/test set...")
    #####################################################################
    
    with open("predictions.pkl", "wb") as fp:
        pickle.dump(predictions, fp)
    
    with open("val_keys.pkl", "wb") as fp:
        pickle.dump(val_keys, fp)
    
#    with open("predictions.pkl", "rb") as fp:
#        predictions = pickle.load(fp)
#    
#    with open("val_keys.pkl", "rb") as fp:
#        val_keys = pickle.load(fp)
    
    from get_localizations import getLocalizations
    from get_localizations import getVidLocalizations

    # [4949, 4369, 4455, 4317, 4452]
    #predictions = [p.cpu() for p in predictions]  # convert to CPU tensor values
    localization_dict = getLocalizations(val_keys, predictions, BATCH_SIZE, \
                                         threshold, seq_threshold)

    print(localization_dict)
    
#    for i in range(0,101,10):
#        filtered_shots = filter_action_segments(localization_dict, epsilon=i)
#        filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
#        with open(filt_shots_filename, 'w') as fp:
#            json.dump(filtered_shots, fp)

    # Apply filtering    
    i = 60  # optimum
    filtered_shots = utils.filter_action_segments(localization_dict, epsilon=i)
    #i = 7  # optimum
    #filtered_shots = filter_non_action_segments(filtered_shots, epsilon=i)
    filt_shots_filename = "predicted_localizations_th0_5_filt"+str(i)+".json"
    with open(filt_shots_filename, 'w') as fp:
        json.dump(filtered_shots, fp)
    print("Prediction file written to disk !!")
    #####################################################################
    # count no. of parameters in the model
    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(utils.count_parameters(model)))
