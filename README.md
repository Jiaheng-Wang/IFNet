# IFNet
## IFNet: An Interactive Frequency Convolutional Neural Network for Enhancing Motor Imagery Decoding from EEG

This is the PyTorch implementation of the IFNet architecture for MI-EEG classification. 

## IFNet: Architecture

![The IFNet architecture](/IFNet.png)

IFNet aims to explore cross-frequency interactions for enhancing feature representation of MI tasks. Guide by neurophysiological priors and efficient convolution operations, IFNet is capable to extract spectro-spatio-temporally robust features for MI decoding from EEG.

## IFNet: Implementation

- [ ] Set up a virtual environment that meets requirements for code running. 
- [ ] Organize the original data as the following file structure:

    	DatasetDir/A01
    	            -/training.mat
    	            -/evaluation.mat
    	            .
    	          /A02
    	            -/training.mat
    	            -/evaluation.mat
    	            .
    	          /...

We provide an example of loading BCIC-IV-2A data from original .gdf files. It is showed in ***dataload.m*** file.

Specifically, for each .mat file, it contains two items *EEG_data* and *labels* with the shape of (*C, T, N*) and (*N*,), respectively.

- [ ] Configure the file ***config.py*** with personalized  settings.
- [ ] Run the file ***within_subject.py*** !

## IFNet: Results

The classification results for IFNet and other competing architectures are as follows: 
<div align=center><img src="/results.png" alt="The IFNet results" style="zoom:70%;"/></div>

We also introduce IFNet V2 which yields the highest **79.89%** classification accuracy on BCIC-IV-2A. This is currently under research  in online settings.

## Cite:

*IEEE Transactions on Neural Systems and Rehabilitation Engineering*.  (Under Review)


## Acknowledgment
We thank Mane Ravikiran et al  for their wonderful works. 

Ravikiran Mane, Effie Chew, Karen Chua, Kai Keng Ang, Neethu Robinson, A.P. Vinod, Seong-Whan Lee, and Cuntai Guan, **"FBCNet: An Efficient Multi-view Convolutional Neural Network for Brain-Computer Interface,"** arXiv preprint arXiv:2104.01233 (2021) https://arxiv.org/abs/2104.01233*
