import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

def consistent_topology_loss_with_fused_sim(fused_similarity, edge_index, target_adj, tau=0.1, percentile=80):
    N = fused_similarity.size(0)
    with torch.no_grad():
        if N > 1000:
            sample_size = min(5000, N * (N-1) // 2)
            idx1 = torch.randint(0, N, (sample_size,), device=fused_similarity.device)
            idx2 = torch.randint(0, N, (sample_size,), device=fused_similarity.device)
            valid_mask = idx1 != idx2
            if valid_mask.sum() > 100:
                # 直接在原始fused_similarity上采样计算阈值
                feat_thresh = torch.quantile(fused_similarity[idx1[valid_mask], idx2[valid_mask]], percentile / 100.0)
            else:
                # 默认阈值改为0（因为不归一化，范围是[-1,1]）
                feat_thresh = torch.tensor(0.0, device=fused_similarity.device)
            del idx1, idx2, valid_mask
        else:
            # 排除对角线
            mask = ~torch.eye(N, device=fused_similarity.device, dtype=bool)
            feat_thresh = torch.quantile(fused_similarity[mask], percentile / 100.0)
            del mask

        constraint_mask = ((fused_similarity > feat_thresh) & (target_adj > 0)) | \
                          ((fused_similarity <= feat_thresh) & (target_adj <= 0))
        
    feat_prob = torch.sigmoid(fused_similarity / tau)
        
    result = (F.binary_cross_entropy(feat_prob, target_adj * constraint_mask, reduction='none') * constraint_mask).sum() / (constraint_mask.sum() + 1e-8)
        
    return result

class CrossViewContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
        self.refl_mask = None
        self.between_mask = None
        
    def build_positive_mask(self, edge_index, num_nodes):
        device = edge_index.device
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        eye_matrix = torch.eye(num_nodes, device=device)
        
        self.refl_mask = adj_matrix - eye_matrix
        self.refl_mask = (self.refl_mask > 0).float()
        
        self.between_mask = (adj_matrix > 0).float()
        self.between_mask.fill_diagonal_(1.0)

        return self.refl_mask, self.between_mask
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.mm(z1, z2.t())
    
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        self_sim = self.sim(z1, z1)
        refl_sim = f(self_sim)
        between_sim = f(self.sim(z1, z2))
        
        loss = -torch.log(
            ((refl_sim * self.refl_mask).sum(dim=1) + (between_sim * self.between_mask).sum(dim=1)) /
            (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + 1e-8)
         )
        return loss, self_sim
    
    def forward(self, frozen_output, tuned_output, batch_size=0):
        if self.refl_mask is None or self.between_mask is None:
            raise ValueError("请先调用 build_positive_mask()")
        if batch_size == 0:
            l1, frozen_sim = self.semi_loss(frozen_output, tuned_output)
            l2, tuned_sim = self.semi_loss(tuned_output, frozen_output)
        
        ret = (l1 + l2) * 0.5
        return ret.mean(), frozen_sim, tuned_sim

class DualBranchFramework(nn.Module):
    def __init__(self, hidden_dim, mode='adaptive', alpha_init=0.5):
        super().__init__()
        self.mode = mode  # 'adaptive', 'sum', 'concat'
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        if mode == 'concat':
            self.concat_mapper = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def _compute_similarity_matrix(self, features):
        features_norm = F.normalize(features, dim=-1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        return similarity_matrix

    def forward(self, frozen_output, prompt_output, frozen_sim=None, prompt_sim=None):
        frozen_sim = frozen_sim if frozen_sim is not None else self._compute_similarity_matrix(frozen_output)
        prompt_sim = prompt_sim if prompt_sim is not None else self._compute_similarity_matrix(prompt_output)
        if self.mode == 'sum':
            fused_output = frozen_output + prompt_output
            fused_sim = frozen_sim + prompt_sim

        elif self.mode == 'concat':
            concat_feat = torch.cat([frozen_output, prompt_output], dim=-1)
            fused_output = self.concat_mapper(concat_feat)
            fused_sim = self._compute_similarity_matrix(fused_output)

        else:  # adaptive
            fused_output = self.alpha * frozen_output + (1 - self.alpha) * prompt_output
            fused_sim = self.alpha * frozen_sim + (1 - self.alpha) * prompt_sim

        return fused_sim, fused_output, self.alpha
    
    def encode(self, frozen_output, prompt_output):
        if self.mode == 'sum':
            fused_output = frozen_output + prompt_output
        elif self.mode == 'concat':
            concat_feat = torch.cat([frozen_output, prompt_output], dim=-1)
            fused_output = self.concat_mapper(concat_feat)
        else:  # adaptive
            fused_output = self.alpha * frozen_output + (1 - self.alpha) * prompt_output
        return fused_output

class PromptModule(nn.Module):
    def __init__(self, hidden_dim, num_layers, bottleneck_dim=32, alpha_init=0.1):
        super().__init__()
        self.adapters = nn.ModuleList()
        self.alphas = nn.ParameterList() # 专门用于存储可学习标量
        
        for _ in range(num_layers):
            # 1. 定义 Adapter 层
            self.adapters.append(nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, hidden_dim)
            ))
            
            # 2. 定义可学习系数 alpha (标量)
            # 初始化为 alpha_init (0.1)
            self.alphas.append(nn.Parameter(torch.tensor(alpha_init)))

    def forward(self, x, edge_index, gnn_model, edge_weight=None):
        """
        Args:
            x: 初始节点特征
            edge_index: 图结构
            gnn_model: 冻结的 GNN 模型
        """
        xs = []
        num_layers = len(gnn_model.convs)
        
        # 同时遍历: GNN层, Adapter, 和对应的 Alpha
        iterator = zip(gnn_model.convs, self.adapters, self.alphas)
        
        for i, (conv, adapter, alpha) in enumerate(iterator):
            if edge_weight is None:
                # 1. GNN 前向计算 (Frozen)
                x_gnn = conv(x, edge_index)
            else:
                x_gnn = conv(x, edge_index, edge_weight)
            
            # 2. Adapter 计算
            x_adapter = adapter(x_gnn)
            
            # 3. 加权融合 (Scaled Residual Connection)
            # x = GNN_out + alpha * Adapter_out
            x = x_gnn + alpha * x_adapter
            
            # 4. 激活与 Dropout (保持原 GNN 逻辑, Pre-activation)
            if i < num_layers - 1:
                if hasattr(gnn_model, 'act'):
                    x = gnn_model.act(x)
                else:
                    x = F.relu(x)
                
                # 使用 PromptModule 的 training 状态控制 dropout
                x = F.dropout(x, p=gnn_model.dropout, training=self.training)
            
            # 收集中间层供 JK 使用
            xs.append(x)

        # 5. JK 后处理
        if gnn_model.jk_mode == "last":
            return xs[-1]
        elif gnn_model.jk_mode == "list":
            return xs
        elif gnn_model.jk_mode in {"cat", "max"}:
            return gnn_model.jk(xs)
        else:
            raise RuntimeError(f"Unknown jk mode: {gnn_model.jk_mode}.")
