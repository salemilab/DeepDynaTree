import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dl import feat_dict
import numpy as np


######################################## Network Model ############################################


class Net(nn.Module):
    def __init__(self, args, pool="mean"):
        super(Net, self).__init__()
        dim_hidden = 64
        dim_input = len(feat_dict[args.node_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])-1
        
        self.dim_output = num_classes
        self.enc = nn.Sequential(
                nn.Linear(1, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, self.dim_output))
        self.pool = pool
    
    def forward(self, x):
        info = dict()
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x, info
    
    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}