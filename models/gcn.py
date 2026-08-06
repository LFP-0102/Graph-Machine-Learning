import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.5
    ):
        super().__init__()


        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )


        self.conv2 = GCNConv(
            hidden_dim,
            output_dim
        )


        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = torch.dropout(
            x,
            p=self.dropout,
            train=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )


        return x