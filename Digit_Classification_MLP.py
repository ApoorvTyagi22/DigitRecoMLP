#this is the MLP model for digit classification using the MNIST dataset
#the model is implemented using the pytorch library
#the model is a simple feed forward neural network with 5 hidden layers and a tanh activation function
#the model uses batch normalization to normalize the input to each layer


#importing the necessary libraries
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#------------------------------------------------------------------------------------------------------------------------------------
# Linear Layer
class Linear: 

    def __init__(self, feat_in, feat_out, bias=True):
        # Xavier initialization, gain value taken from https://pytorch.org/docs/stable/nn.init.html
        self.weight = torch.randn((feat_in, feat_out)) / feat_in**0.5 
        if bias:
            self.bias = torch.randn(feat_out) * 0.01

    def __call__(self, x): # forward pass
        self.out = x @ self.weight
        if self.bias is not None: 
            self.out += self.bias 
        return self.out 

    def parameters(self): # return the parameters of the layer
        if self.bias is None: 
            return [self.weight]
        else: return [self.weight] + [self.bias]

#------------------------------------------------------------------------------------------------------------------------------------
# Batch Normalization
class BatchNormOneDim:

    def __init__(self, dim, eps =1e-5, momentum = 0.1): # dim is the number of features in the input
        self.eps = eps
        self.momentum = momentum
        self.training = True 

        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        # momentum update paramters 
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self,x):
        
        if self.training:
            xmean = x.mean(0, keepdims=True) # batch mean
            xvar  = x.var(0, keepdims = True) # batch varience 
        else:
            # use running mean and varience during inference
            xmean = self.running_mean
            xvar  = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # formula from batchnorm paper 
        #print(f"xh:{xhat.shape}, gamma:{self.gamma.shape}, beta: {self.beta.shape}")
        self.out = self.gamma * xhat + self.beta 
        if self.training:
            with torch.no_grad():
                # update running mean and varience using momentum 
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar 

        return self.out 

    def parameters(self):
        return [self.gamma, self.beta] # to be updated using back propgation and SGD

#------------------------------------------------------------------------------------------------------------------------------------
# Tanh activation function
class Tanh: 

    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out 

    def parameters(self):
        return[]
        
#------------------------------------------------------------------------------------------------------------------------------------

def inference(X):
    with torch.no_grad():
        X = X.view(X.shape[0], -1).float()
        X = X / 255.0
        
        for layer in model:
            X = layer(X)
        logits = X
        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        # Predicted class is the one with the highest probability
        predicted_class = torch.argmax(probs, dim=1)
        
        return predicted_class, probs


def load_split_data(path, split_ratio=0.8):

    X, Y = torch.load(path)
    X = X/255
    
    total_samples = X.size(0)
    training_amount = int(split_ratio * total_samples)
    
    # Split the data into training and validation sets
    Xtr = X[:training_amount, :, :]
    Ytr = Y[:training_amount]
    Xval = X[training_amount:, :, :]
    Yval = Y[training_amount:]
    
    F.one_hot(Ytr)
    Xtr = Xtr.view(48000,-1) 
    Xval = Xval.view(12000,-1)

    return Xtr, Ytr, Xval, Yval

@torch.no_grad()
def split_loss(split, model): 
    X, Y = {
        'train' : (Xtr, Ytr),
        'val': (Xval,Yval), 
        
    }[split]

    # Flatten the images
    X = X.view(X.shape[0], -1).float()
    X = X/ 255
    
    # Forward pass
    for layer in model:
        X = layer(X)
    logits = X
    # Compute loss
    loss = F.cross_entropy(logits, Y)
    print(f"{split} loss:", loss.item())


#------------------------------------------------------------------------------------------------------------------------------------

# Inilization of the model
Xtr, Ytr, Xval, Yval = load_split_data('MNIST/processed/training.pt')
numChannels = Xtr[0,:].shape[0]
n_hidden = 100
nums = sorted(set(Ytr.tolist()))
nums = len(nums) # the number of different classes in the dataset
model = [
    
    Linear(numChannels, n_hidden), BatchNormOneDim(n_hidden), Tanh(), # input layer (784, 100)
    Linear(   n_hidden, n_hidden), BatchNormOneDim(n_hidden), Tanh(), # hidden layers (100, 100)
    Linear(   n_hidden, n_hidden), BatchNormOneDim(n_hidden), Tanh(),
    Linear(   n_hidden, n_hidden), BatchNormOneDim(n_hidden), Tanh(),
    Linear(   n_hidden, n_hidden), BatchNormOneDim(n_hidden), Tanh(),
    Linear(   n_hidden, nums), BatchNormOneDim(nums), # output layer (100, 10)
   
]

for layer in model[:-1]: 
    if isinstance(layer, Linear):
        layer.weight *= 5/3  # scalling the weights for a gain 

parameters = []
for layer in model:
    for p in layer.parameters():
        parameters.append(p)
print(sum(p.nelement() for p in parameters)) # number of parameters in total 

for p in parameters: 
    p.requires_grad = True

max_steps = 60000
batch_size = 32 
lossi = []

#------------------------------------------------------------------------------------------------------------------------------------

# Actuall training loop 
g = torch.Generator().manual_seed(42)
for i in range (max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,),  generator = g, )
    Xb, Yb = Xtr[ix], Ytr[ix]

    #Forward pass
    for layer in model:
        Xb = layer(Xb)
    loss = F.cross_entropy(Xb, Yb)      

    #Backward Pass
    for p in parameters: 
        p.grad = None 
    loss.backward()

    #Update
    lr = 0.01 if i < 30000 else 0.001 # learning rate decay
    for p in parameters: 
        p.data += -lr *p.grad 

    # Tracking 
    if i % 5000 == 0: 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())

#------------------------------------------------------------------------------------------------------------------------------------

# Calculate and print the training and validation loss
split_loss("val", model)
split_loss("train", model)

#------------------------------------------------------------------------------------------------------------------------------------
# Testing the model
Xtest, Ytest  = torch.load("MNIST/processed/test.pt")
predicted_class, probs = inference(Xtest)
print(predicted_class[:10])
print(Ytest[:10])
print((predicted_class == Ytest).float().mean())

