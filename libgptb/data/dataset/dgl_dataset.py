import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
import dgl
from dgl.dataloading import GraphDataLoader


class DGLDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data_dgl')


        if self.datasetName in ["Cora", "CiteSeer", "PubMed"]:
            dgl = getattr(importlib.import_module('dgl.data'), f'{self.datasetName.capitalize()}GraphDataset')
        if self.datasetName in ["Computers", "Photo"]:
            if self.datasetName == "Computers":
                dgl = getattr(importlib.import_module('dgl.data'), f'AmazonCoBuyComputerDataset')
            else:
                dgl = getattr(importlib.import_module('dgl.data'), f'AmazonCoBuy{self.datasetName}Dataset')
        if self.datasetName in ["CS", "Physics"]:
            dgl = getattr(importlib.import_module('dgl.data'), f'Coauthor{self.datasetName}Dataset')
        if self.datasetName in ["Yelp"]:
            dgl = getattr(importlib.import_module('dgl.data'), f'{self.datasetName}Dataset')

        if self.datasetName not in ["PPI"]:
            self.dataset = dgl(path)
            self.data = self.dataset[0]

        if self.datasetName in ["PPI"]:
            PPI = getattr(importlib.import_module('dgl.data'), f'{self.datasetName}Dataset')       
            self.train_dataset = PPI( mode='train' , raw_dir = path)
            val_dataset = PPI( mode='valid',  raw_dir = path)
            test_dataset = PPI(mode='test' , raw_dir = path)
            self.train_loader = GraphDataLoader(self.train_dataset, batch_size=1, shuffle=False)
            self.val_loader = GraphDataLoader(val_dataset, batch_size=2, shuffle=False)
            self.test_loader = GraphDataLoader(test_dataset, batch_size=2, shuffle=False)
    
    def get_data(self):
        device = torch.device('cuda')
        if self.datasetName not in ["PPI"]:
            return self.data
        else:
            return self.train_loader,self.val_loader,self.test_loader
        
    
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        if self.datasetName not in ["PPI"]:
            return {"input_dim": self.data.ndata['feat'].shape[1]}
        else:
            return {"input_dim": self.train_dataset[0].ndata['feat'].shape[1]}
        
    
