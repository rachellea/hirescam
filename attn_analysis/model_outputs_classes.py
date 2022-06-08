#model_outputs_classes.py
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

import torch
import torch.nn as nn
import torchvision.models as models

class AxialNet(nn.Module): #Baseline 1 (formerly named Body_Cll_Avg)
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNet, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = resnet_features()
        self.conv2d = final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

def resnet_features():
    resnet = models.resnet18(pretrained=True)
    return nn.Sequential(*(list(resnet.children())[:-2]))

def final_conv():
    """Return 2d conv layers that collapse the representation as follows:
    input: [512, 14, 14]
           [64, 12, 12]
           [32, 10, 10]
           [16, 8, 8]
           [16, 6, 6]"""
    return nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))

#########################
# Model Outputs Classes #-------------------------------------------------------
#########################
def return_modeloutputsclass(model_name):
    if model_name == 'AxialNet':
        return ModelOutputs_AxialNet
    else:
        assert False, 'Invalid model_name'

class ModelOutputs_AxialNet():
    """Class for running an AxialNet <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the custom convolutional part
    of the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape 1, 134, 3, 420, 420]
        x = x.squeeze(dim=0) #[134, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        conv2d = self.model.conv2d._modules.items()
        fc = self.model.fc
        avgpool_1d = self.model.avgpool_1d
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from. 
        for name, module in conv2d:
            if self.verbose: print('Applying conv2d layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #names are e.g. '4'. target_layer_name can be e.g. ['2','4']
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data
        
        #Apply the rest of the model to get the final output
        if self.verbose: print('Applying FC and avg pool')
        x = fc(x) #[slices, 83, 1, 1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x_perslice_scores = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]. Scores for 83 diseases on every slice
        x = avgpool_1d(x_perslice_scores) #out shape [1, 83, 1]
        output = torch.squeeze(x, dim=2) #out shape [1, 83]
        return target_activations, output

#AxialNet conv2d layer numbers:
# (conv2d): Sequential(
#     (0): Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
#     (5): ReLU(inplace)
#     (6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
#     (7): ReLU(inplace)
#   )
