#2020-11-24-run-example.py
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
import torch
import timeit

import run_attn_analysis
from load_dataset import custom_datasets
from attn_analysis import model_outputs_classes

if __name__=='__main__':
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    #first save some random parameters of a model for example purposes
    model = model_outputs_classes.AxialNet(n_outputs=83,slices=134)
    check_point = {'params': model.state_dict()}
    torch.save(check_point, os.path.join(results_dir, 'AxialNet_epoch34'))
    
    #run the Grad-CAM and HiResCAM code on this model
    for attention_type in ['gradcam-vanilla','proposed-hirescam']:
        print('\nCalculating explanations with',attention_type)
        tot0 = timeit.default_timer()
        run_attn_analysis.AttentionAnalysis(
                     attention_type=attention_type,
                     attention_type_args={'model_name':'AxialNet',
                                          'target_layer_name':'7'}, #'7' is the last layer of the custom convolutional layers
                     setname='valid',
                     custom_net=model_outputs_classes.AxialNet,
                     custom_net_args={'n_outputs':83,'slices':134},
                     params_path=os.path.join(results_dir, 'randombodyconv_epoch34'),
                     stop_epoch=34, #made up in this example
                     dataset_class=custom_datasets.CTDataset_2019_10,
                    dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'','valid':''}},
                    results_dir = results_dir)
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')