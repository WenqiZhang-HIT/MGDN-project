#Graph convolutional block
import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
 
        if x.size()[-1] != self.in_channel:
            x = x.reshape(-1,self.in_channel)
        x = self.lin(x)
        x = self.propagate(edge_index, size=(x.size()[0], x.size()[0]), x=x)
        return x
 
    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
 
        # Step 3: Normalize node features.
        row, col = edge_index
        
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j
 
    def update(self, aggr_out):
 
        # Step 5: Return new node embeddings.
        return aggr_out