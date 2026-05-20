import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        jk: str = "last",
        act=torch.nn.ReLU(),
        pool_type='mean',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk_mode = jk
        self.act = act
        self.pool_type = pool_type
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if jk == "last":
            self.jk = None
            jk_out_channels = hidden_channels
        elif jk in {"cat", "max"}:
            self.jk = JumpingKnowledge(mode=jk, channels=hidden_channels, num_layers=num_layers)
            jk_out_channels = self.jk.out_channels
        elif jk == "list":
            self.jk = None
            jk_out_channels = hidden_channels
        else:
            raise ValueError(f"Unsupported jk mode: {jk}.")

        self.out_channels = jk_out_channels
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}.")

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        if self.jk_mode == "list":
            return xs
        elif self.jk_mode == "last":
            return xs[-1]
        elif self.jk_mode in {"cat", "max"}:
            return self.jk(xs)
        else:
            raise RuntimeError(f"Unknown jk mode: {self.jk_mode}.")
    
    def forward_weighted(self, x, edge_index, edge_weight):
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        if self.jk_mode == "list":
            return xs
        elif self.jk_mode == "last":
            return xs[-1]
        elif self.jk_mode in {"cat", "max"}:
            return self.jk(xs)
        else:
            raise RuntimeError(f"Unknown jk mode: {self.jk_mode}.")
    
    def pool_embeddings(self, embeddings, batch):
        return self.pool(embeddings, batch)
     