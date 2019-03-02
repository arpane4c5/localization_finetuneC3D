#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:36:05 2018

@author: Arpan
"""

import torch
import numpy as np
import cv2
import os
import pickle
import shutil
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import skimage.io as io
from skimage.transform import resize

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)

def create_variable(tensor, use_gpu=False):
    # Do cuda() before wrapping with variable
    if use_gpu:
        if torch.cuda.is_available():
            return Variable(tensor.cuda())
        else:
            print("GPU not available ! Tensor loaded in main memory.")
            return Variable(tensor)
    else:
        return Variable(tensor)

# Split the dataset files into training, validation and test sets
# All video files present at the same path (assumed)
def split_dataset_files(datasetPath):
    filenames = sorted(os.listdir(datasetPath))         # read the filename
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]
    

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# send a list of lists containing start and end frames of actions
# eg [98, 218], [376, 679], [2127, 2356], [4060, 4121], [4137, 4250]]
# Return a sequence of action labels corresponding to frame features
def get_vid_labels_vec(labels, vid_len):
    if labels is None or len(labels)==0:
        return []
    v = []
    for i,x in enumerate(labels):
        if i==0:
            v.extend([0]*(x[0]-1))
        else:
            v.extend([0]*(x[0]-labels[i-1][1]-1))
        v.extend([1]*(x[1]-x[0]+1))  
    v.extend([0]*(vid_len - labels[-1][1]))
    return v

# return the number of frames present in a video vid
def getNFrames(vid):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        import sys
        print("Capture Object not opened ! Abort")
        sys.exit(0)
        
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return l


def getFeatureVectors(datasetpath, videoFiles, sequences):
    """
     Iteratively take the batch information and extract the feature sequences 
     from the videos
     datasetpath : Prefix of the path to the dataset containing the videos
     videoFiles : list/tuple of filenames for the videos (size n)
     sequences :  list of start frame numbers and end frame numbers 
     sequences[0] and [1] are torch.LongTensor of size n each.
     returns a list of lists. Inner list contains a sequence of arrays 
    """
    grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        videoFile = videoFile.split('/')[1]
        vid_feat_seq = []
        # use capture object to get the sequences
        cap = cv2.VideoCapture(os.path.join(datasetpath, videoFile))
        if not cap.isOpened():
            print("Capture object not opened : {}".format(videoFile))
            import sys
            sys.exit(0)
            
        start_frame = sequences[0][i]
        end_frame = sequences[1][i]
        ####################################################    
        #print "Start Times : {} :: End Times : {}".format(start_frame, end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if ret:
            # convert frame to GRAYSCALE
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            print("Frame not read: {} : {}".format(videoFile, start_frame))

        for stime in range(start_frame+1, end_frame+1):
            ret, frame = cap.read()
            if not ret:
                print("Frame not read : {} : {}".format(videoFile, stime))
                continue
            
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, 
            #                iterations, poly_n, poly_sigma, flags[, flow])
            # prev(y,x)~next(y+flow(y,x)[1], x+flow(y,x)[0])
            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print "For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape)
            
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # stack sliced arrays along the first axis (2, 12, 16)
            sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                    ang[::grid_size, ::grid_size]), axis=0)
            sliced_flow = sliced_flow.ravel()   # flatten
            vid_feat_seq.append(sliced_flow.tolist())    # append to list
            prev_frame = curr_frame
        cap.release()            
        batch_feats.append(vid_feat_seq)
        
    return batch_feats

def readAllPartitionFeatures(featuresPath, keys):
    """
    Load the OF/HOG/C3D etc features of the train/val/test set into a dictionary. 
    Dictionary has key as the filename (without ext) of video and value as the 
    numpy feature matrix. (Assuming all have .npy extension)
    
    Parameters:
    ------
    OFfeaturesPath: str
        path till the binary files of OF features
    keys: list of strings
        list of filenames (without ext), which will be keys in the dictionary
        and the corresponding files are the OF/HOG/C3D numpy dumps.
    
    Returns:
    ------
    feats: dict
        a dictionary of key: numpy matrix of size (N-1 x 1 x (2x(H/grid)x(W/grid))) 
        for OF, size (N x 1 x 3600) for HOG, size (N-depth+1 x 1 x 4096) for C3D,
        where N is the total number of frames in video, H=360, W=640 and grid is the
        sampling distance between two consecutive pixels (refer extract_denseOF_par.py).
        depth is the number of frames sent to c3d pretrained model for 
        feature extraction (extract_c3d_feats.py).
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(featuresPath, k)+".npy"
        assert os.path.exists(featpath), "featpath not found {}".format(featpath)
        with open(featpath, "rb") as fobj:
            feats[k] = np.load(fobj)
            
    print("Features loaded into dictionary ...")
    return feats


def readAllNumpyFrames(numpyFramesPath, keys):
    """
    Load the frames of the train/val/test set into a dictionary. Dictionary 
    has key as the filename(without ext) of video and value as the Nxhxw numpy  
    matrix.
    """
    feats = {}
    for k in keys:
        featpath = os.path.join(numpyFramesPath, k)+".npy"
        feats[k] = np.load(featpath)
            
    print("Features loaded into dictionary ...")
    return feats

    
def getFeatureVectorsFromDump(features, videoFiles, sequences, motion=True):
    """Select only the batch features from the dictionary of features (corresponding
    to the given sequences) and return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
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
        if motion:
            vid_feat_seq = vidFeats[start_frame:end_frame]
        else:
            vid_feat_seq = vidFeats[start_frame:(end_frame+1)]
        
        batch_feats.append(vid_feat_seq)
        
    return batch_feats

def getBatchFeatures(features, videoFiles, sequences, motion=True):
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
        if motion:
            vid_feat_seq = vidFeats[start_frame:end_frame, :, :]
        else:
            vid_feat_seq = vidFeats[start_frame:(end_frame+1), :, :]
        
        vid_feat_seq = np.squeeze(vid_feat_seq, axis = 1)
        batch_feats.append(vid_feat_seq)
        
    return batch_feats

def getC3DFeatures(features, videoFiles, sequences):
    """
    Select the batch frames from the dictionary of numpy frames (corresponding
    to the given sequences) and extract C3D feature vector from them (fc7 layer)
    return them as a list of lists. 
    OFfeatures: a dictionary of features {vidname: numpy matrix, ...}
    videoFiles: the list of filenames for a batch
    sequences: the start and end frame numbers in the batch videos to be sampled.
    """
    #grid_size = 20
    batch_feats = []
    # Iterate over the videoFiles in the batch and extract the corresponding feature
    for i, videoFile in enumerate(videoFiles):
        # get key value for the video. Use this to read features from dictionary
        videoFile = videoFile.split('/')[1].rsplit('.', 1)[0]
            
        start_frame = sequences[0][i]   # starting point of sequences in video
        end_frame = sequences[1][i]     # end point
        # verify
        assert (end_frame-start_frame+1)>=16, "SEQ_SIZE should be greater than 16"
        
        # Load features
        # (N-16+1 x 1 x 4096) sized numpy array
        vidFeats = features[videoFile]  
        # extract (seq_size-15) x 1 x 4096 matrix and append to 
        vid_feat_seq = vidFeats[start_frame:(end_frame-15+1), :, :]
        # dissolve the centre single-dimension, result is list of (seq_size-15) x 4096 
        vid_feat_seq = np.squeeze(vid_feat_seq, axis=1)
        batch_feats.append(vid_feat_seq)
        
    return batch_feats


# should be called only for seq_len >=16
def make_c3d_variables(feats, labels, use_gpu=False):
    # Create the input tensors and target label tensors
    #for item in feats:
        # item is a list with (sequence of) 9 1D vectors (each of 1152 size)
    # form a tensor of size = (batch x (seq_size-16) x feat_size) eg (256 x 1 x 4096)
    #feats = torch.Tensor(feats)  # Runtime error: https://github.com/pytorch/pytorch/issues/2246
    feats = torch.Tensor(np.array(feats).astype(np.double))
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []

#   Append the sequence of labels of len (seq_size-16+1) to the target list.
    # Iterate over the batch labels, for each extract seq_size labels and extend 
    # in the target list
    seq_size = len(labels)  
    for i in range(labels[0].size(0)):
        lbls = [y[i] for y in labels]      # get labels of frames (size seq_size)
        temp = [1 if sum(lbls[s:e])>=8 else 0 for s,e in enumerate(range(16, seq_size+1))]
        target.extend(temp)   # for c3d
        
    # Form a wrap into a tensor variable as B X S X I
    return create_variable(feats, use_gpu), create_variable(torch.Tensor(target), use_gpu)

# Inputs: feats: list of lists
def make_variables_new(feats, labels, motion=True):
    # Create the input tensors and target label tensors
    #for item in feats:
        # item is a list with (sequence of) 9 1D vectors (each of 1152 size)
        
    feats = torch.Tensor(np.array(feats).astype(np.double))
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []
    # Append the sequence of labels of len (seq_size-1) to the target list for OF.
    # Iterate over the batch labels, for each extract seq_size labels and extend 
    # in the target list
    seq_size = len(labels)  
    if motion:      # For OF features
        for i in range(labels[0].size(0)):
            lbls = [y[i] for y in labels]      # get labels of frames (size seq_size)
            temp = [1 if sum(lbls[s:e])>=1 else 0 for s,e in enumerate(range(2, seq_size+1))]
            target.extend(temp)   
            
    else:       # For HOG or other frame features
        for i in range(labels[0].size(0)):
            lbls = [y[i] for y in labels]      # get labels of frames (size seq_size)
            target.extend(lbls)

    # Form a wrap into a tensor variable as B X S X I
    return create_variable(feats), create_variable(torch.Tensor(target))


# Inputs: feats: list of lists
def make_variables(feats, labels, motion=True):
    # Create the input tensors and target label tensors
    #for item in feats:
        # item is a list with (sequence of) 9 1D vectors (each of 1152 size)
        
    feats = torch.Tensor(feats)
    feats[feats==float("-Inf")] = 0
    feats[feats==float("Inf")] = 0
    # Form the target labels 
    target = []
    
    #target.extend([y[i] for y in labels])   # for HOG
    for i in range(labels[0].size(0)):
        if labels[0][i]>0:
            if motion:
                target.extend([0]*(labels[0][i]-1) + [1]*labels[1][i])
            else:
                target.extend([0]*labels[0][i] + [1]*labels[1][i])
        else:
            if motion:
                target.extend([0]*labels[0][i] + [1]*(labels[1][i]-1))
            else:
                target.extend([0]*labels[0][i] + [1]*labels[1][i])
    # Form a wrap into a tensor variable as B X S X I
    return create_variable(feats), create_variable(torch.Tensor(target))

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


# function to remove the action segments that have less than "epsilon" frames.
def filter_action_segments(shots_dict, epsilon=10):
    filtered_shots = {}
    for k,v in shots_dict.items():
        vsegs = []
        for segment in v:
            if (segment[1]-segment[0] >= epsilon):
                vsegs.append(segment)
        filtered_shots[k] = vsegs
    return filtered_shots

# function to remove the non-action segments that have less than "epsilon" frames.
# Here we need to merge one or more action segments
def filter_non_action_segments(shots_dict, epsilon=10):
    filtered_shots = {}
    for k,v in shots_dict.items():
        vsegs = []
        isFirstSeg = True
        for segment in v:
            if isFirstSeg:
                prev_st, prev_end = segment
                isFirstSeg = False
                continue
            if (segment[0] - prev_end) <= epsilon:
                prev_end = segment[1]   # new end of segment
            else:       # Append to list
                vsegs.append((prev_st, prev_end))
                prev_st, prev_end = segment     
        # For last segment
        if len(v) > 0:      # For last segment
            vsegs.append((prev_st, prev_end))
        filtered_shots[k] = vsegs
    return filtered_shots


def get_sport_clip(frames_list, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str      OR frames_list : list of consecutive N framePaths
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    #clip = sorted(glob(os.path.join('data', clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in frames_list])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels
