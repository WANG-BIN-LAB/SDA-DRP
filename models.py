import torch
import sys  
import os
import torch.nn as nn
import torch.utils.data as Data
from typing import TypeVar, List
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, Sequential
from copy import deepcopy
import numpy as np
from utility import *


class FeatureFusion(nn.Module):
    def __init__(self, dim_2d, dim_3d, output_dim):
        super(FeatureFusion, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim_2d + dim_3d, 1),
            nn.Sigmoid()
        )

        self.transform = nn.Linear(dim_2d + dim_3d, output_dim)
    
    def forward(self, feat_2d, feat_3d):
        combined = torch.cat([feat_2d, feat_3d], dim=-1)
        gate = self.gate(combined)
        fused = gate * feat_2d + (1 - gate) * feat_3d

        output = self.transform(combined)
        return output

Tensor = TypeVar('torch.tensor')

def random_zero(tensor, probability):
    mask = torch.bernoulli(torch.full_like(tensor, probability))
    return tensor * mask

# Models for training
class ConnectNetwork(nn.Module):
    def __init__(self, encoder, decoder, noise_flag=False, fix_source=False):
        super(ConnectNetwork, self).__init__()
        self.encoder = encoder
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
        self.decoder = decoder
        self.noise_flag = noise_flag

    def forward(self, inputs, node_x, edge_index):
        if self.noise_flag and self.training:
            encoded_input = self.encode(inputs + torch.randn_like(inputs, requires_grad=False)*0.1)
        else:
            encoded_input = self.encode(inputs)
        output = self.decoder(encoded_input, node_x, edge_index)
        return output

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, z, edge_index, edge_weight):
        return self.decoder(z, edge_index)

    def loss_function(self, inputs, recons):
        recons_loss = F.mse_loss(inputs, recons)
        return recons_loss


class GraphMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, drug_num, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(GraphMLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        self.node_num = drug_num
        

        self.feature_fusion = FeatureFusion(dim_2d=64, dim_3d=64, output_dim=64)
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        hidden_dims = deepcopy(hidden_dims)

        hidden_dims.insert(0, input_dim + 64)
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append((GATConv(hidden_dims[i], hidden_dims[i+1], add_self_loops=True, heads=2, concat=False), 'x, edge_index -> x'))
            modules.append(act_fn())
            modules.append(nn.Dropout(self.drop))
        self.convseq = Sequential('x, edge_index', modules)    
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.hidden_dims = hidden_dims
        self.reset_para()
    
    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def batch_graph(self, x, edge_index):
        graphs = []
        for item in range(x.size(0)):
            graphs.append([x[item], edge_index])
        graph_batch = Data.DataLoader(dataset=GraphDataset(graphs_dict=graphs), 
                                      collate_fn=collate, batch_size=len(graphs), shuffle=False)   
        return graph_batch
    
    def forward(self, x, node_x, edge_index):   
        # Map features to different drug label nodes
        x = x.repeat(self.node_num, 1, 1)
        x = x.transpose(0, 1)
        node_x = node_x.repeat(x.size(0), 1, 1)
        

        node_x_2d = node_x[..., :64]
        node_x_3d = node_x[..., 64:128]
        

        fused_drug_feat = self.feature_fusion(node_x_2d, node_x_3d)
        

        feat = torch.cat((x, fused_drug_feat), -1)
        
        # Build graph structure data
        graphs = self.batch_graph(feat, edge_index)
        batch_graph = next(iter(graphs))
        embed = self.convseq(batch_graph.x, batch_graph.edge_index)
        embed = embed.reshape([-1, self.node_num, self.hidden_dims[-1]])
        # Use the GCN layer to encode features
        output = self.output_layer(embed)
        return output.squeeze()


class FeatMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(FeatMLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_dim, output_dim, hidden_dims: List = None, 
                 drop: float = 0.2, noise_flag: bool = True, norm_flag: bool = False, fix_source=False):
        super(EncoderDecoder, self).__init__()
        self.drop = drop
        self.shared_encoder = encoder
        self.decoder = decoder
        self.noise_flag = noise_flag
        self.norm_flag = norm_flag
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)    
        # Create the private encoder            
        modules = [] 
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=True),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.SELU(),
                    nn.Dropout(self.drop))
            )
        modules.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.private_encoder = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded_input = self.encode(inputs)
        output = self.decoder(encoded_input)
        return [inputs, output, encoded_input]
    
    def p_encode(self, inputs: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            encoded_input = self.private_encoder(inputs + torch.randn_like(inputs, requires_grad=False) * 0.1)
        else:
            encoded_input = self.private_encoder(inputs)

        if self.norm_flag:
            return F.normalize(encoded_input, p=2, dim=1)
        else:
            return encoded_input

    def s_encode(self, inputs: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            encoded_input = self.shared_encoder(inputs + torch.randn_like(inputs, requires_grad=False) * 0.1)
        else:
            encoded_input = self.shared_encoder(inputs)
        if self.norm_flag:
            return F.normalize(encoded_input, p=2, dim=1)
        else:
            return encoded_input
    
    def encode(self, inputs: Tensor) -> Tensor:
        p_latent_code = self.p_encode(inputs)
        s_latent_code = self.s_encode(inputs)
        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def loss_function(self, inputs: Tensor, recons: Tensor, z: Tensor) -> dict:
        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]
        recons_loss = F.mse_loss(inputs, recons)
        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-8)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-8)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
        loss = recons_loss + ortho_loss
        return loss


class TransferNetwork(nn.Module):
    def __init__(self, encoder, classifier, fix_source=True):
        super(TransferNetwork, self).__init__()
        self.encoder = encoder
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))        
        self.classifier = classifier       
        # input_dim = encoder.output_layer[0].out_features
        # Removed Discriminator layers

    def forward(self, input_data, node_x, edge_index):
        # Forward pass returning feature and class output
        feature = self.encoder(input_data)
        class_output = self.classifier(feature, node_x, edge_index)
        return class_output, feature

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]