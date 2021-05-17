##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from tqdm import tqdm
import model
from  util import *
import os
import itertools
import numpy as np 
import copy

'''
    < var >
    Convert tensor to Variable
'''
def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    var = Variable(tensor.type(dtype), requires_grad=requires_grad)
    
    return var

class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

# To make cuda tensor
def tocuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def mse_loss(score, target=1):
    dtype = type(score)

    if target == 1:
        label = var(torch.ones(score.size()), requires_grad=False)
    elif target == 0:
        label = var(torch.zeros(score.size()), requires_grad=False)
    
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    
    return loss

def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def lr_decay_rule(epoch, start_decay=100, lr_decay=100):
    decay_rate = 1.0 - (max(0, epoch - start_decay) / float(lr_decay))
    return decay_rate

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss