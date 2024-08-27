# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:11:58 2021

@author: Suncy
"""

import numpy as np # linear algebra
import dgl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dl import feat_dict


######################################## Network Model ############################################

# message passing user-defined functions
# could define different message passing strategy 

def edge_udf(edges):
    # cat states of edge and source node
    #cat_feat = torch.cat((edges.src['h_feat'],edges.data['h_feat']),1)
    cat_feat = torch.cat((edges.src['h_feat'],edges.data['h_feat'],edges.dst['h_feat']),1)
    return {'cat': cat_feat}
  

def node_udf(edges):
    # send edge state to dst node
    return {'h_feat': edges.data['h_feat']}


def reducer(nodes):
    # cat states of node and in-bound edge
    cat_feat = torch.cat((torch.sum(nodes.mailbox['h_feat'],1),nodes.data['h_feat']),1)
    #cat_feat = torch.cat((nodes.mailbox['h_feat'][:,0,:],nodes.data['h_feat']),1)
    return {'cat': cat_feat} 


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.h_feat = 64
        self.num_iter = 13
        n_feats = len(feat_dict[args.node_feat_cols])
        e_feats = len(feat_dict[args.edge_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])
        self.device = torch.device("cuda" if args.num_gpus > 0 else "cpu")

        self.embed = args.embedding
        
        # node LSTM & edge LSTM
        # 21 dim node feature & 2 dim edge feature
        self.Node_LSTM = nn.LSTMCell(n_feats, self.h_feat)
        self.Edge_LSTM = nn.LSTMCell(e_feats, self.h_feat)
        
        self.m = nn.LeakyReLU()
        
        # message passing network
        self.node_mpn = nn.Linear(2*self.h_feat, n_feats)
        self.edge_mpn = nn.Linear(3*self.h_feat, e_feats)

        # batch normalization on messages
        self.node_bn = nn.BatchNorm1d(n_feats)
        self.edge_bn = nn.BatchNorm1d(e_feats)

        # linear classifier
        self.fc = nn.Linear(self.h_feat, num_classes)
    
    def forward(self, g):
        info = dict()
        # num of nodes & edges in batched g
        num_nodes = g.ndata['feat'].shape[0]
        num_edges = g.edata['feat'].shape[0]
        
        # initialization of hidden state and cell state
        g.ndata['h_feat'] = torch.zeros(num_nodes, self.h_feat).to(self.device)
        g.ndata['c_feat'] = torch.zeros(num_nodes, self.h_feat).to(self.device)
        g.edata['h_feat'] = torch.zeros(num_edges, self.h_feat).to(self.device)
        g.edata['c_feat'] = torch.zeros(num_edges, self.h_feat).to(self.device)
        
        for i in range(self.num_iter):
            if i == 0: # first iteration, input is feature vec
                g.ndata['h_feat'], g.ndata['c_feat'] = self.Node_LSTM(g.ndata['feat'], (g.ndata['h_feat'], g.ndata['c_feat']))
                g.edata['h_feat'], g.edata['c_feat'] = self.Edge_LSTM(g.edata['feat'], (g.edata['h_feat'], g.edata['c_feat']))
            else: # later iteration, input is message
                g.ndata['h_feat'], g.ndata['c_feat'] = self.Node_LSTM(g.ndata['msg'], (g.ndata['h_feat'], g.ndata['c_feat']))
                g.edata['h_feat'], g.edata['c_feat'] = self.Edge_LSTM(g.edata['msg'], (g.edata['h_feat'], g.edata['c_feat']))
            
            
            # message passing
            g.apply_edges(edge_udf) # update the feature vector of edges
            g.edata['msg'] = self.edge_bn(self.m(self.edge_mpn(g.edata['cat']))) # generate edge message
            g.update_all(node_udf,reducer) # send edge state to dst nodes
            g.ndata['msg'] = self.node_bn(self.m(self.node_mpn(g.ndata['cat']))) # generate node message
           
        # linear classifier
        output = self.fc(g.ndata['h_feat'])
        
        #return output, g.ndata['h_feat'], info  # for embedding vectors
        
        return output, info
    
    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}
