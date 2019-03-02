# C3D model finetuned for localization of Cricket Strokes in telecast videos. 

Finetune a C3D model on the highlights videos and then on the main dataset training samples.

## Steps:

1. Convert raw videos to numpy matrices for easy loading to and from the disk.

2. The 360x640 sized frames are resized with a scaling factor for 0.32 (for height) and then taking the 112x112 center crop of the frames.

3. VideoDataset creates examples from the video frames, by taking seqLen (16) consecutive frames randomly from the videos and passing them in batches to the model.

4. We use Stochastic Gradient Descent with momentum 0.9, learning rate of 0.001, used Cross Entropy Loss and running for 30 Epochs. The learning rate is decreased by a factor of 0.1 after every 10 epochs. 
The batch size was chosen as 16, since a larger batch size did not fit on the two K4200 GPU cards available for training. Each epoch was run for 2k iterations and then stopped.

5. The finetuning took 91721.92 secs and total number of parameters in the model were 57426434. 

6. The evaluation on the highlights videos was performed using Weighted Temporal IoU metric and got 0.6026 on the validation set videos.
