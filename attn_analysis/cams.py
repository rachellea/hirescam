#cams.py
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

#Conceptually related to https://github.com/jacobgil/pytorch-grad-cam/blob/master/grad-cam.py

import os
import copy
import numpy as np
import torch, torch.nn as nn

from . import model_outputs_classes

class RunExplanationMethod():
    def __init__(self, attention_type, model, device, label_meanings,
                 model_name, target_layer_name):
        self.attention_type = attention_type
        self.model = model
        self.model.eval()
        self.modeloutputsclass = model_outputs_classes.return_modeloutputsclass(model_name)
        self.device = device
        self.label_meanings = label_meanings #all the abnormalities IN ORDER
        self.model_name = model_name
        self.target_layer_name = target_layer_name #e.g. '2'
    
    def return_explanation(self, ctvol, gr_truth, volume_acc, chosen_label_index):
        """Obtain explanation for prediction of <chosen_label_index>
        on a CT volume <ctvol>. The ground truth labels for
        this ctvol are provided in <gr_truth>. The name of the ctvol is
        provided as a string in <volume_acc>.
            ctvol is a torch Tensor with shape [1, 134, 3, 420, 420]; in
                model_outputs_classes.py the batch dimension of 1 gets removed
                before putting it through the model.
            chosen_label_index is an integer"""
        #obtain gradients and activations:
        extractor = self.modeloutputsclass(self.model, self.target_layer_name)
        self.all_target_activs_dict, output = extractor.run_model(ctvol)
        
        #Use <one_hot> to multiply by zero every score except the score
        #for the target disease:
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][chosen_label_index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device)
        one_hot = torch.sum(one_hot * output)
        
        #Backward pass:
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        #grads_list is a list of gradients, for each of the target layers.
        #Hooks are registered when we do the backward pass, which is why
        #we needed to wait until after calling backward() to get the
        #gradients.
        self.all_grads_dict = extractor.get_gradients()
        
        #Select gradients and activations for the target layer:
        target_grads = self.all_grads_dict[self.target_layer_name].cpu().data.numpy() #e.g. out shape [134, 16, 6, 6]
        target_activs = self.all_target_activs_dict[self.target_layer_name].cpu().data.numpy() #e.g. out shape [134, 16, 6, 6]
        
        if self.attention_type == 'gradcam-vanilla':
            return gradcam_vanilla(target_grads, target_activs)
        elif self.attention_type == 'proposed-hirescam':
            return proposed_hirescam(target_grads, target_activs)

def gradcam_vanilla(target_grads, target_activs):
    """Calculate vanilla GradCAM attention volume.
    An alpha_k is produced by taking the average of the gradients going in
    to the k^th feature map. The alpha_k is multipled against that feature map.
    The final Grad-CAM attention is the result of summing all the
    alpha_k*feature_map arrays.
    
    <target_grads> is a np array with shape [134, 16, 6, 6] which is
        height, features, square, square.
    <target_activs> is a np array [134, 16, 6, 6]"""
    target_grads_reshaped = np.transpose(target_grads,axes=(1,0,2,3)) #out shape [16, 134, 6, 6]
    alpha_ks = np.mean(target_grads_reshaped,axis=(1,2,3)) #out shape [16]
    alpha_ks_unsq = np.expand_dims(np.expand_dims(np.expand_dims(alpha_ks,axis=0),axis=2),axis=3) #out shape [1,16,1,1]
    product = np.multiply(target_activs,alpha_ks_unsq) #out shape [134, 16, 6, 6] from broadcasting
    raw_cam_volume = np.sum(product,axis=1) #out shape [134, 6, 6]
    
    #Note that the raw_cam_volume has not yet been ReLU'd or normalized
    return raw_cam_volume

def proposed_hirescam(target_grads, target_activs):
    """Calculate new proposed GradCAM attention volume.
    Here, the gradients going in to the k^th feature map are element-wise
    multiplied against the k^th feature map, and then the average is taken
    over the feature dimension."""
    #Improved step: obtain the CAM by doing element-wise multiplication of
    #the target_grads and the target_activs and then collapsing across the
    #feature dimension. It's important to do the summing over the
    #feature dimension AFTER you have multiplied the grads and the activations,
    #so that you obtain the most accurate explanation possible.
    raw_cam_volume = np.multiply(target_grads,target_activs) #e.g. out shape [134, 16, 6, 6]
    #Now sum over the feature dimension:
    raw_cam_volume = np.sum(raw_cam_volume,axis=1) #e.g. out shape [134, 6, 6]
    return raw_cam_volume
