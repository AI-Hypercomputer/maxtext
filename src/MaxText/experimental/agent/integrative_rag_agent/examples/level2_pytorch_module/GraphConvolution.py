python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, adj_matrix):
 
        adj_self_loop = adj_matrix + torch.eye(adj_matrix.size(0))
 
        d = adj_self_loop.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
 
        normalized_adj = adj_self_loop @ d_mat_inv_sqrt.t() @ d_mat_inv_sqrt
 
        support = self.linear(node_features)
        output = torch.mm(normalized_adj, support)
        return F.relu(output)