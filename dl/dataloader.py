# -*- coding: utf-8 -*-
"""
@author: Suncy

dataloader for Graph NN Models
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torchvision import transforms, utils
import os.path as osp
import dgl
import torch
import json
from dl import feat_dict, logger
from dgl.data import DGLDataset
from collections import Counter
from torch.utils.data import Dataset

# dataloader for graph models
class Dataset(DGLDataset):
    def __init__(self, args, phase, device="cpu"):
        self.device = device
        ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
        
        self.phase = phase
        if phase in ["train", "valid", "test"]:
            self.node_df = pd.read_csv(f"{ds_folder}/{phase}_s.csv", low_memory=False)
            self.edge_df = pd.read_csv(f"{ds_folder}/{phase}_edge.csv", low_memory=False)
            self.nbg_df = self.node_df[self.node_df['cluster_id']!='Background']
            # shuffle the target column for permutation importance analysis
            if args.permutation_test:
                if args.permutation_feat != "None":
                    bg_df = self.node_df[self.node_df['cluster_id']=='Background']
                    #nbg_df = self.node_df[self.node_df['cluster_id']!='Background']
                    if args.permutation_feat in ['ltt_shape_cat', 'gamma_cat']:
                        feat = args.permutation_feat
                        shuffle_cols = pd.DataFrame(self.nbg_df, columns=[feat+'_0',feat+'_1',feat+'_2',feat+'_3']).values
                        np.random.shuffle(shuffle_cols)
                        for i in range(4):
                            self.nbg_df[feat+'_'+str(i)] = shuffle_cols[:,i]
                    else:
                        shuffle_col = self.nbg_df[args.permutation_feat].values
                        np.random.shuffle(shuffle_col)
                        self.nbg_df[args.permutation_feat] = shuffle_col
                    self.node_df = pd.concat([bg_df, self.nbg_df])
                            
        else:
            raise NotImplementedError

        self.tree_ids = self.node_df["sim"].unique()  # num of trees

        self.node_feat_cols = feat_dict[args.node_feat_cols]
        #self.node_feat_org = ['sim','state_id']+feat_dict["raw_feats"][0:11]
        #self.node_feat_org = ['sim']+feat_dict["raw_feats"][0:11]
        self.node_label_cols = args.node_label_cols
        self.edge_feat_cols = feat_dict[args.edge_feat_cols]
        self.n_label = len(feat_dict[args.node_label_cols.split("_cat")[0]])

        # Pre-process the bg nodes
        if args.pro_bg_nodes == "all_zero":
            self.node_df.loc[self.node_df["cluster_id"] == 'Background', self.node_feat_cols] = 0
        else:
            raise NotImplementedError

        self.add_self_loop = args.add_self_loop
        self.bidirection = args.bidirection

    def process(self):        
        pass

    def __getitem__(self, index):

        tree_id = self.tree_ids[index]  # tree of index

        # dgl tree of index
        onetree_node_df = self.node_df[self.node_df['sim'] == tree_id]
        onetree_edge_df = self.edge_df[self.edge_df['sim'] == tree_id]
        src_ids = torch.tensor(onetree_edge_df['from'].values)
        dst_ids = torch.tensor(onetree_edge_df['to'].values)
        src_ids -= 1
        dst_ids -= 1
        g = dgl.graph((src_ids, dst_ids))  # create dgl
        sorted_onetree_node_df = onetree_node_df.sort_values(by='node')
        
        # assign features and labels for background nodes
        node_feat = sorted_onetree_node_df[self.node_feat_cols].values
        node_label = sorted_onetree_node_df[self.node_label_cols].values
        num_nodes = node_feat.shape[0]
        num_feat = node_feat.shape[1]

        # assign features for nodes and edges, assign labels
        g.ndata["feat"] = torch.tensor(node_feat, dtype=torch.float32)
        g.ndata["label"] = torch.tensor(node_label, dtype=torch.int64)
        g.edata["feat"] = torch.tensor(onetree_edge_df[self.edge_feat_cols].values, dtype=torch.float32)
        
        #if self.phase in ['test']:
        #    node_org_feat = sorted_onetree_node_df[self.node_feat_org].values
        #    g.ndata["org_feat"] = torch.tensor(node_org_feat, dtype=torch.float32)
        # wait for reading weight norm-asinh

        if self.add_self_loop:
            g = dgl.add_self_loop(g)  # TODO: Add self-loop with self-edge weight filled with zero
        if self.bidirection:
            g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)

        g = g.to(self.device)
        return g
        
    def __len__(self):
        return len(self.tree_ids)   # number of trees

# dataloader for node-based models
class MLPDataset(Dataset):
    def __init__(self, args, phase, device="cpu"):
        self.device = device
        ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)

        if phase in ["train", "valid", "test"]:
            self.node_df = pd.read_csv(f"{ds_folder}/{phase}_s.csv", low_memory=False)
            self.edge_df = pd.read_csv(f"{ds_folder}/{phase}_edge.csv", low_memory=False)
            self.nbg_df = self.node_df[self.node_df['cluster_id']!='Background']  
            if args.permutation_test:
                if args.permutation_feat != "None":
                    if args.permutation_feat in ['ltt_shape_cat', 'gamma_cat']:
                        feat = args.permutation_feat
                        shuffle_cols = pd.DataFrame(self.nbg_df, columns=[feat+'_0',feat+'_1',feat+'_2',feat+'_3']).values
                        np.random.shuffle(shuffle_cols)
                        for i in range(4):
                            self.nbg_df[feat+'_'+str(i)] = shuffle_cols[:,i]
                    else:
                        shuffle_col = self.nbg_df[args.permutation_feat].values
                        np.random.shuffle(shuffle_col)
                        self.nbg_df[args.permutation_feat] = shuffle_col                         
        else:
            raise NotImplementedError

        self.node_feat_cols = feat_dict[args.node_feat_cols]
        #self.node_feat_org = feat_dict["raw_feats"][0:11]
        self.node_label_cols = args.node_label_cols
        #self.edge_feat_cols = feat_dict[args.edge_feat_cols]
        self.n_label = len(feat_dict[args.node_label_cols.split("_cat")[0]])
        self.sample = self.nbg_df[self.node_feat_cols].values
        self.label = self.nbg_df[self.node_label_cols].values
        self.sim = self.nbg_df['sim'].values
        self.sample = torch.tensor(self.sample, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.int64)
        self.sim = torch.tensor(self.sim, dtype=torch.int64)
        self.model = args.model
        
    def __getitem__(self, index):
        x = self.sample[index]
        y = self.label[index]
        z = self.sim[index]
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)
        return x,y,z
    
    def __len__(self):
        return len(self.label)   # number of trees

# create batch of trees(aggregate multiples trees to a single tree)
def collate_fn(batch_graphs):
    g = dgl.batch(batch_graphs)
    return g


def gen_label_weight(args):
    # Get the weights for the unbalanced sample based on the positive sample
    # weights inversely proportional to class frequencies in the training data
    #ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
    #node_df = pd.read_csv(f'{ds_folder}/train_s.csv')

    #node_label = node_df[args.node_label_cols].values
    #label_counter = Counter(node_label)
    #n_samples = len(node_label)
    #n_classes = len(label_counter)

    #label_weights = [n_samples / (n_classes * label_counter[i])+1 for i in range(n_classes)]
    #label_weights = list(np.load(f'../aly/label_weights_resp+TB.npy'))
    #label_weights = [2.0218350812053556, 14.299666391664246, 6.999405162936564]
    label_weights = [0.476749498798085, 3.371866898828094, 1.6504624607291616]
    label_weights[0] = label_weights[0]
    label_weights[1] = label_weights[1]
    label_weights[2] = label_weights[2]

    if args.loss_ignore_bg:
        #label_weights[-1] = 0
        label_weights.append(0)
    if args.graph_info == False:
        label_weights = label_weights[0:-1]
    return label_weights




