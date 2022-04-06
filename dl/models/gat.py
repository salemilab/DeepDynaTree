#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
gat.py: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dl import feat_dict


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        in_feats = len(feat_dict[args.node_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])
        self.embed = args.embedding
        
        h_feat = 64
        num_heads = 3
        embed_edge_feat = False
        e_in_feats = len(feat_dict[args.edge_feat_cols])
        e_out_feat = 64
        
        self.layer1 = MultiHeadGATLayer(in_feats, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_in_feats, e_out_dim=e_out_feat)
        self.layer2 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer3 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer4 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer5 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer6 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer7 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.layer8 = MultiHeadGATLayer(h_feat * num_heads, h_feat, num_heads,
                                        embed_edge_feat=embed_edge_feat, e_in_dim=e_out_feat*num_heads, e_out_dim=e_out_feat)
        self.fc = nn.Linear(num_heads*h_feat, num_classes)
        
    def forward(self, g):
        info = dict()
        node_feat = g.ndata["feat"]
        #edge_feat = g.edata["feat"]
        #g.edata["z"] = edge_feat

        h = self.layer1(g, node_feat)
        h = F.elu(h)
        h = self.layer2(g, h)
        h = F.elu(h)
        h = self.layer3(g, h)
        h = F.elu(h)
        h = self.layer4(g, h)
        h = F.elu(h)
        h = self.layer5(g, h)
        h = F.elu(h)
        h = self.layer6(g, h)
        h = F.elu(h)
        h = self.layer7(g, h)
        h = F.elu(h)
        h = self.layer8(g, h)
        h1 = F.elu(h)
        h = self.fc(h1)
        #return h, h1, info   # for embedding vectors
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat',
                 embed_edge_feat=False, e_in_dim=None, e_out_dim=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, embed_edge_feat, e_in_dim, e_out_dim))
        self.merge = merge

    def forward(self, g, h):
        node_outs = [attn_head(g, h) for attn_head in self.heads]
        #node_outs = zip(*head_outs)
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(node_outs, dim=1)
            # return torch.cat(node_outs, dim=1), torch.cat(edge_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(node_outs))
            # return torch.mean(torch.stack(node_outs)), torch.mean(torch.stack(edge_outs))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, embed_edge_feat, e_in_dim, e_out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.embed_edge_feat = embed_edge_feat
        if self.embed_edge_feat:
            self.e_fc = nn.Linear(e_in_dim, e_out_dim, bias=False)
            # equation (2)
            self.attn_fc = nn.Linear(2 * out_dim + e_out_dim, 1, bias=False)
        else:
            # equation (2)
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        # z2 = torch.cat([edges.data["z"], edges.src['z'], edges.dst['z']], dim=1) # Consider the edge weights
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h1):
        # equation (1)
        z = self.fc(h1)
        g.ndata['z'] = z
        '''
        if self.embed_edge_feat:
            g.edata['z'] = self.e_fc(e_h)
        '''
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')