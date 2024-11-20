import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from util.time import *
from util.env import *

import math
import torch.nn.functional as F
from .GCNcov_layer import GCNConv


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, dim, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, dim))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, dim ))
                modules.append(nn.BatchNorm1d(dim))
                modules.append(nn.LeakRelu())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out


class GCNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, out_dim, inter_dim=0, heads=1, node_num=100):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channel, out_dim)
        self.conv2 = GCNConv(out_dim, out_channel)
        self.conv3 = GCNConv(in_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_channel)
 
    def forward(self, x, edge_index, embedding=None, node_num=0, weight_matrix=None):
        device = get_device()
        y = self.conv1(x, edge_index).to(device)
    
        y = F.relu(self.bn1(y))

        y = F.dropout(y, training=self.training)
        y = self.bn2(self.conv2(y, edge_index))
        
        return F.log_softmax(y, dim=1)

class MGDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=60, out_layer_num=1, topk=20):

        super(MGDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        embed_dim = dim
        
        self.mul_embedding = nn.Embedding(node_num, embed_dim)
        self.phy_embedding = nn.Embedding(node_num, embed_dim)
        self.net_embedding = nn.Embedding(node_num, embed_dim)

        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.net_bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.phy_bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.share_gcn_layers = nn.ModuleList([
            GCNLayer(in_channel = 60, out_channel = dim, out_dim=16, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.phy_gcn_layers = nn.ModuleList([
            GCNLayer(in_channel = 16, out_channel = 64, out_dim=16, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.net_gcn_layers = nn.ModuleList([
            GCNLayer(in_channel = 48, out_channel = 64, out_dim=16, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.mul_conv = nn.Conv1d(60,64,kernel_size=1)
        self.phy_conv = nn.Conv1d(15,64,kernel_size=1)
        self.net_conv = nn.Conv1d(45,64,kernel_size=1)

        self.out_layer = OutLayer(64, node_num, out_layer_num, dim=64, inter_num = out_layer_inter_dim)
        self.phy_out_layer = OutLayer(64, node_num, out_layer_num, dim=1, inter_num = out_layer_inter_dim)
        self.net_out_layer = OutLayer(64, node_num, out_layer_num, dim=3, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.mul_embedding.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phy_embedding.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.net_embedding.weight, a=math.sqrt(5))


    def forward(self, data, phy_edge_index, net_edge_index, mul_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()


        gcn_outs = []
        phy_gcn_outs = []
        net_gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            all_embeddings = self.mul_embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk
            
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji
            
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)

            mul_edge_index = mul_edge_index.to(device)
            gcn_out = self.share_gcn_layers[i](x,batch_gated_edge_index, node_num=node_num, embedding=all_embeddings, weight_matrix=cos_ji_mat)
            gcn_outs.append(gcn_out)

        y = torch.cat(gcn_outs, dim=1)
        y = y.view(batch_num,node_num, -1)

        indexes = torch.arange(0,node_num).to(device)

        out = torch.mul(y, self.mul_embedding(indexes))
        out = out.permute(0,2,1)

        out = F.relu(self.bn_outlayer_in(out))

        out = out.permute(0,2,1)

        out = self.dp(out)

        out = self.out_layer(out)
        out = out.view(-1, 64).float()

        mul_x = x.permute(1,0)
        mul_x = self.mul_conv(mul_x)
        mul_x = mul_x.permute(1,0) 

        for i, edge_index in enumerate(edge_index_sets):
            
            phy_all_embeddings = self.phy_embedding(torch.arange(node_num).to(device))
            net_all_embeddings = self.net_embedding(torch.arange(node_num).to(device))

            phy_weights_arr = phy_all_embeddings.detach().clone()
            phy_all_embeddings = phy_all_embeddings.repeat(batch_num, 1)
            net_weights_arr = net_all_embeddings.detach().clone()
            net_all_embeddings = net_all_embeddings.repeat(batch_num, 1)

            phy_weights = phy_weights_arr.view(node_num, -1)
            net_weights = net_weights_arr.view(node_num, -1)

            phy_cos_ji_mat = torch.matmul(phy_weights, phy_weights.T)
            phy_normed_mat = torch.matmul(phy_weights.norm(dim=-1).view(-1,1), phy_weights.norm(dim=-1).view(1,-1))
            phy_cos_ji_mat = phy_cos_ji_mat / phy_normed_mat
            net_cos_ji_mat = torch.matmul(net_weights, net_weights.T)
            net_normed_mat = torch.matmul(net_weights.norm(dim=-1).view(-1,1), net_weights.norm(dim=-1).view(1,-1))
            net_cos_ji_mat = net_cos_ji_mat / net_normed_mat

            topk_num = self.topk
            
            phy_topk_indices_ji = torch.topk(phy_cos_ji_mat, topk_num, dim=-1)[1]
            net_topk_indices_ji = torch.topk(net_cos_ji_mat, topk_num, dim=-1)[1]

            self.phy_learned_graph = phy_topk_indices_ji
            self.net_learned_graph = net_topk_indices_ji
            
            phy_gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            phy_gated_j = phy_topk_indices_ji.flatten().unsqueeze(0)
            phy_gated_edge_index = torch.cat((phy_gated_j, phy_gated_i), dim=0)
            phy_batch_gated_edge_index = get_batch_edge_index(phy_gated_edge_index, batch_num, node_num).to(device)
            net_gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            net_gated_j = net_topk_indices_ji.flatten().unsqueeze(0)
            net_gated_edge_index = torch.cat((net_gated_j, net_gated_i), dim=0)
            net_batch_gated_edge_index = get_batch_edge_index(net_gated_edge_index, batch_num, node_num).to(device)

            phy_gcn_out = self.phy_gcn_layers[i](out[:,:16], phy_batch_gated_edge_index, node_num=node_num*batch_num,embedding=phy_all_embeddings,weight_matrix=phy_cos_ji_mat)
            phy_gcn_outs.append(phy_gcn_out)
            net_edge_index = net_edge_index.to(device)
            net_gcn_out = self.net_gcn_layers[i](out[:,16:], net_batch_gated_edge_index, node_num=node_num*batch_num,embedding=net_all_embeddings,weight_matrix=net_cos_ji_mat)
            net_gcn_outs.append(net_gcn_out)


        outs = []
        x_phy = torch.cat(phy_gcn_outs, dim=1)
        x_net = torch.cat(net_gcn_outs, dim=1)


        x_phy = x_phy.view(batch_num, node_num, -1)
        x_net = x_net.view(batch_num, node_num, -1)

        indexes = torch.arange(0,node_num).to(device)

        phy_out = torch.mul(x_phy, self.phy_embedding(indexes))
        net_out = torch.mul(x_net, self.net_embedding(indexes))
        
        net_out = net_out.permute(0,2,1)
        phy_out = phy_out.permute(0,2,1)

        net_out = F.relu(self.net_bn_outlayer_in(net_out))
        phy_out = F.relu(self.phy_bn_outlayer_in(phy_out))

        net_out = net_out.permute(0,2,1)
        phy_out = phy_out.permute(0,2,1)

        net_out = self.dp(net_out)
        phy_out = self.dp(phy_out)
        net_out = self.net_out_layer(net_out)
        phy_out = self.phy_out_layer(phy_out)
        net_out = net_out.view(-1, node_num*3).float()
        phy_out = phy_out.view(-1, node_num).float()

        outs.append(phy_out)
        outs.append(net_out)
        return outs, self.learned_graph, x_net, x_phy
        