from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
import os.path as osp
import sys
from torch_geometric.nn import GINConv, global_add_pool
from libgptb.models import SingleBranchContrast
import libgptb.losses as L
import math

    
def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)

     
class InfoGraph(nn.Module):
    def __init__(self, config, data_feature):
        super(InfoGraph, self).__init__()

        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 1.)
        self.gamma = config.get('gamma', 0.1)
        self.prior = config.get('prior', False)
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 3)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = max( data_feature.get('input_dim'), 1)

        self.embedding_dim  = self.nhid * self.layers
        self.encoder_model = Encoder(self.input_dim, self.nhid, self.layers)

        gconv = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, activation=torch.nn.ReLU, num_layers=self.layers).to(self.device)
        fc1 = FC(hidden_dim=self.nhid * self.layers)
        fc2 = FC(hidden_dim=self.nhid * self.layers)
        self.encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(self.device)
        self.contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(self.device)

        # optimizer = Adam(encoder_model.parameters(), lr=0.01)