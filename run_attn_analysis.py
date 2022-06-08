#attention_analysis.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import os
import copy
import torch
import numpy as np
import pandas as pd

from attn_analysis import cams

import warnings
warnings.filterwarnings('ignore')

class AttentionAnalysis(object):
    def __init__(self, attention_type, attention_type_args,
                 setname, custom_net, custom_net_args, params_path,
                 stop_epoch, dataset_class, dataset_args, results_dir):
        """
        Variables:
        <attention_type>: str; either
            'gradcam-vanilla' (for vanilla Grad-CAM), or
            'proposed-hirescam' (for HiResCAM, in which feature maps and
                gradients are element-wise multiplied before taking the avg
                over the feature dimension)
        <attention_type_args>: dict; additional arguments needed to calculate
            the specified kind of attention. In this dict we need to specify 
            'model_name' and 'target_layer_name'
        <setname>: str; which split to use e.g. 'train' or 'val' or 'test'; will
            be passed to the <dataset_class>
        <custom_net>: a PyTorch model class
        <custom_net_args>: dict; arguments to pass to the PyTorch model
        <params_path>: str; path to the model parameters that will be loaded in
        <stop_epoch>: int; epoch at which the model saved at <params_path> was
            saved
        <dataset_class>: a PyTorch dataset class
        <dataset_args>: dict; arguments to pass to the <dataset_class>
        <results_dir>: the directory in which to store results"""
        self.attention_type = attention_type
        assert self.attention_type in ['gradcam-vanilla','proposed-hirescam']
        self.attention_type_args = attention_type_args
        assert 'model_name' in self.attention_type_args.keys()
        assert 'target_layer_name' in self.attention_type_args.keys()
        self.setname = setname
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args #dict of args
        self.params_path = params_path
        self.stop_epoch = stop_epoch
        self.CTDatasetClass = dataset_class
        self.dataset_args = dataset_args #dict of args
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        self.device = torch.device('cuda:0')
        self.verbose = False #print less
        
        #Run
        self.load_model()
        self.load_dataset()
        self.loop_over_dataset_and_labels()
    
    ######################################################
    # Methods to Load Model, Dataset, and Chosen Indices #----------------------
    ######################################################
    def load_model(self):
        print('Loading model')
        self.model = self.custom_net(**self.custom_net_args).to(self.device)
        check_point = torch.load(self.params_path)
        self.model.load_state_dict(check_point['params'])
        self.model.eval()
        #If everything loads correctly you will see the following message:
        #IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    
    def load_dataset(self):
        print('Loading dataset')
        self.chosen_dataset = self.CTDatasetClass(setname = self.setname, **self.dataset_args)
        self.label_meanings = self.chosen_dataset.return_label_meanings()
    
    ###########
    # Looping #-----------------------------------------------------------------
    ###########
    def loop_over_dataset_and_labels(self):
        #Could make self.chosen_dataset_indices some subset of the data, or
        #could do the following, which includes all available data:
        self.chosen_dataset_indices = [x for x in range(0,len(self.chosen_dataset))]
        print('Looping over dataset and labels')
        five_percent = max(1,int(0.05*len(self.chosen_dataset_indices)))
        
        #Iterate through the examples in the dataset
        for list_position in range(len(self.chosen_dataset_indices)):
            if self.verbose: print('Starting list_position',list_position)
            idx = self.chosen_dataset_indices[list_position] #int, e.g. 5
            example = self.chosen_dataset[idx]
            ctvol = example['data'].unsqueeze(0).to(self.device) #unsqueeze to create a batch dimension. out shape [1, 135, 3, 420, 420]
            gr_truth = example['gr_truth'].cpu().data.numpy() #out shape [80]
            volume_acc = example['volume_acc'] #this is a string, e.g. 'RHAA12345_5.npz'
            if self.verbose: print('Analyzing',volume_acc)
            
            #Now pick the labels for this particular image that you want to
            #make heatmap visualizations for. Pick them based on the ground
            #truth for that image
            chosen_label_indices = np.where(gr_truth==1)[0] #e.g. array([32, 37, 43, 46, 49, 56, 60, 62, 64, 67, 68, 71])
            num_labels_this_ct = int((example['gr_truth']).sum()) #integer
            #when we are using the ground truth to determine the chosen_label_indices, then
            #the number of chosen_label_indices must be equal to the sum of the ground truth
            #multi-hot vector:
            assert num_labels_this_ct == len(chosen_label_indices)
            
            #Calculate label-specific attn and make label-specific attn figs
            for chosen_label_index in chosen_label_indices:
                label_name = self.label_meanings[chosen_label_index] #e.g. 'lung_atelectasis'
                #segprediction is the raw attention
                segprediction = cams.RunExplanationMethod(self.attention_type, self.model, self.device,
                      self.label_meanings, **self.attention_type_args).return_explanation(ctvol, gr_truth, volume_acc, chosen_label_index)
                
                segprediction_clipped_and_normed = clip_and_norm_volume(segprediction)
                #segprediction_clipped_and_normed is the final output of
                #Grad_CAM or HiResCAM. From here it can be used for
                #making visualizations or calculating overlap with a
                #segmentation mask.
                print('Attention explanation for',volume_acc,'and',label_name,'is a',type(segprediction_clipped_and_normed),'with shape',segprediction_clipped_and_normed.shape)
            
            #Report progress
            if list_position % five_percent == 0:
                print('Done with',list_position,'=',round(100*list_position/len(self.chosen_dataset_indices),2),'%')
            del example, ctvol, gr_truth, volume_acc
    
#############
# Functions #-------------------------------------------------------------------
#############
def clip_and_norm_volume(volume):
    volume = np.maximum(volume, 0) #ReLU operation
    volume = volume - np.min(volume)
    if np.max(volume)!=0:
        volume = volume / np.max(volume)
    return volume