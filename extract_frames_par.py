#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 8 01:34:25 2018
@author: Arpan
@Description: Utils file to extract frame from folder videos and dump to disk.
Feature : Frames converted to numpy files(after resizing to half H and W and 
taking the centre crops 112 x 112 x 3), for passing them to the c3d model for 
training or finetuning. Easy to subset consecutive frames and feed them to the 
deep network.
Total Execution Time for 26 vids : 57.25 secs (njobs=10, batch=10)
"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed
    

def extract_vid_frames(srcFolderPath, destFolderPath, njobs=1, batch=10, stop='all'):
    """
    Function to extract the features from a list of videos
    
    Parameters:
    ------
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the frame pixel values in .npy files
    njobs: int
        no. of cores to be used parallely
    batch: int
        no. of video files in a batch. A batch executed parallely and 
        is dumped to disk before starting another batch. Depends on RAM.
    stop: str
        to traversel 'stop' no of files in each subdirectory.
    
    Returns: 
    ------
    traversed: int
        no of videos traversed successfully
    """
    # iterate over the subfolders in srcFolderPath and extract for each video 
    vfiles = os.listdir(srcFolderPath)
    
    infiles, outfiles, nFrames = [], [], []
    
    traversed = 0
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
            
    # iterate over the video files inside the directory sf
    for vid in vfiles:
        if os.path.isfile(os.path.join(srcFolderPath, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
            infiles.append(os.path.join(srcFolderPath, vid))
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.',1)[0]+".npy"))
            nFrames.append(getTotalFramesVid(os.path.join(srcFolderPath, vid)))
            # save at the destination, if extracted successfully
            traversed += 1
#            print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                    
                # to stop after successful traversal of 2 videos, if stop != 'all'
            if stop != 'all' and traversed == stop:
                break
                    
    print("No. of files to be written to destination : "+str(traversed))
    if traversed == 0:
        print("Check the structure of the dataset folders !!")
        return traversed
    ###########################################################################
    #### Form the pandas Dataframe and parallelize over the files.
    filenames_df = pd.DataFrame({"infiles":infiles, "outfiles": outfiles, "nframes": nFrames})
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True])
    filenames_df = filenames_df.reset_index(drop=True)
    nrows = filenames_df.shape[0]
    
    for i in range(int(nrows/batch)):
        #batch_diffs = getHOGVideo(filenames_df['infiles'][i], hog)
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFrames) \
                          (filenames_df['infiles'][i*batch+j]) \
                          for j in range(batch))
        print("i = "+str(i))
        # Writing the diffs in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                #with open(filenames_df['outfiles'][i*batch+j] , "wb") as fp:
                #    pickle.dump(batch_diffs[j], fp)
                np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                print("Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j])
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - (int(nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFrames) \
                              (filenames_df['infiles'][int(nrows/batch)*batch+j]) \
                              for j in range(last_batch_size)) 
        # Writing the diffs in a serial manner
        for j in range(last_batch_size):
            if batch_diffs[j] is not None:
#                with open(filenames_df['outfiles'][(nrows/batch)*batch+j] , "wb") as fp:
#                    pickle.dump(batch_diffs[j], fp)
                np.save(filenames_df['outfiles'][int(nrows/batch)*batch+j], batch_diffs[j])
                print("Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][int(nrows/batch)*batch+j])
    
    ###########################################################################
    print(len(batch_diffs))
    return traversed


def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of the source input video file
        
    Returns:
    ------
    total frames present in the given video file
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames    


def getFrames(srcVideoPath):
    """
    Function to read all the frames of the video file into a single numpy matrix
    and return that matrix to the caller. This function can be called parallely 
    based on the amount of memory available.
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    max_height = 112
    max_width = 112
    scaling_factor=0.32     # not used in this run
    # only shrink if img is bigger than required
    #if max_height < H: #or max_width < W:
        # get scaling factor
     #   scaling_factor = max_height / float(H)
        #if max_width/float(W) < scaling_factor:
        #    scaling_factor = max_width / float(W)
    
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        frameCount +=1
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # resize to 180 x 320 
        curr_frame = cv2.resize(curr_frame, (int(W/2), int(H/2)), cv2.INTER_AREA)
#        curr_frame = cv2.resize(curr_frame, None, fx=scaling_factor, \
#                                fy=scaling_factor, interpolation=cv2.INTER_AREA)
        (h, w) = curr_frame.shape[:2]
#        print("Size : {}".format((h,w)))
        # take the centre crop size is 112 x 112 x 3
        curr_frame = curr_frame[(int(h/2)-56):(int(h/2)+56), (int(w/2)-56):(int(w/2)+56), :]
        
        #cv2.imshow("Frame 112x112", curr_frame)
        #direction = waitTillEscPressed()
        
        # saving as a list of float values (after converting into 1D array)
        features_current_file.append(curr_frame)

    # When everything done, release the capture
    #cv2.destroyAllWindows()
    cap.release()
    #print "{}/{} frames in {}".format(frameCount, totalFrames, srcVideoPath)
    #return features_current_file
    return np.array(features_current_file)      # convert to N x w x h


def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward.")
            return 1
        # For moving back
        elif cv2.waitKey(0)==98:
            print("'b' pressed. Move Back.")
            return 0
        # start of shot
        elif cv2.waitKey(0)==115:
            print("'s' pressed. Start of shot.")
            return 2
        # end of shot
        elif cv2.waitKey(0)==102:
            print("'f' pressed. End of shot.")
            return 3


if __name__=='__main__':
    batch = 1  # No. of videos in a single batch
    njobs = 1   # No. of threads

    #srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    srcPath = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    destPath = "/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D/numpy_vids_112x112"
    if not os.path.exists(srcPath):
        srcPath = "/opt/datasets/cricket/ICC_WT20"
        destPath = "/home/arpan/VisionWorkspace/localization_finetuneC3D/numpy_vids_112x112"
    
    start = time.time()
    nfiles = extract_vid_frames(srcPath, destPath, njobs, batch, stop='all')
    end = time.time()
    print("Total execution time for {} files : {}".format(nfiles, str(end-start)))
    