# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
import sys  
from torch import Tensor
from utility import uniform
import config
device = config.CUDA_ID

# Removed Adversarial_loss

class InfoMax_loss(nn.Module):
    def __init__(self, hidden_dim, EPS = 1e-8):
        super(InfoMax_loss,self).__init__()
        self.EPS = EPS
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.hidden_dim, self.weight)
    
    def discriminate(self, z, summary, sigmoid: bool = True):
        """computes the probability scores assigned to this patch(z)-summary pair."""
        z = torch.unsqueeze(z, dim = 1)
        summary = torch.unsqueeze(summary, dim = -1)
        

        if z.shape[2] != summary.shape[1]:

            linear = nn.Linear(z.shape[2], summary.shape[1]).to(device)
            z = linear(z)
        
        #value = torch.matmul(z, torch.matmul(self.weight.to(device), summary))
        value = torch.matmul(z, summary)
        value = value.squeeze()
        mask = torch.where(value != 0.0)[0]

        return torch.sigmoid(value[mask]) if sigmoid else value[mask]
    
    def process(self, data_type, p_key, p_value):
        summary = []
        for i in data_type:
            idx = torch.where(p_key == i)
            if (len(idx[0]) != 0):
           
                feat = p_value[idx[0]]
                if feat.shape[1] != self.hidden_dim:
                
                    linear = nn.Linear(feat.shape[1], self.hidden_dim).to(device)
                    feat = linear(feat)
                summary.append(feat)
            else:
                summary.append(torch.zeros(1, self.hidden_dim).to(device))
                
        return torch.cat(summary, dim = 0)
    
    def forward(self, s_feat, t_feat, s_type, t_type, prototype):
        """Computes the mutual information maximization objective"""
        batch_s = next(iter(prototype[0]))
        batch_t = next(iter(prototype[1]))
        # Extract the anchor (source) prototypes 
        pos_summary_t = self.process(t_type, batch_s[0].to(device), batch_s[1].to(device))
        neg_summary_t = self.process(t_type, batch_t[0].to(device), batch_t[1].to(device))
        
        # Compute the contrastive loss of InfoMax
        pos_loss_t = -torch.log(self.discriminate(t_feat, pos_summary_t, sigmoid=True) + self.EPS).mean()
        neg_loss_t = -torch.log(1-self.discriminate(t_feat, neg_summary_t, sigmoid=True) + self.EPS).mean()

        return (pos_loss_t + neg_loss_t)/2