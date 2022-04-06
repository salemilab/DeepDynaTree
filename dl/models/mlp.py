# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:32:35 2021

@author: hamel

MLP classifier
"""

import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dl import feat_dict


######################################## Network Model ############################################


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.h_feat = 64
        self.n_feats = len(feat_dict[args.node_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])-1
        self.device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
        
        #self.a = nn.Sigmoid()
        self.a = nn.ReLU()
        
        # linear classifier
        self.fc1 = nn.Linear(self.n_feats, self.h_feat)
        self.fc2 = nn.Linear(self.h_feat, self.h_feat)
        self.fc3 = nn.Linear(self.h_feat, self.h_feat)
        self.fc4 = nn.Linear(self.h_feat, self.h_feat)
        self.fc5 = nn.Linear(self.h_feat, 32)
        self.fc6 = nn.Linear(32, 10)
        self.fc7 = nn.Linear(10, 3)
    
    def forward(self, x):
        info = dict()
        x = self.a(self.fc1(x))
        x = self.a(self.fc2(x)) 
        x = self.a(self.fc3(x)) 
        x = self.a(self.fc4(x)) 
        x = self.a(self.fc5(x)) 
        x = self.a(self.fc6(x)) 
        output = self.a(self.fc7(x)) 
        # linear classifier

        return output, info
    
    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}