import torch
import torch.nn as nn
import torch.nn.functional as F
class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.act = nn.PReLU()
        # 可学习的松弛系数，初始很小
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        return self.act(self.fc(x))

    def orthogonality_loss(self):
        # W: [out_dim, in_dim]
        W = self.fc.weight
        gram = W @ W.t()  # [out_dim, out_dim]
        I = torch.eye(gram.size(0), device=W.device)
        # alpha 越大，越强调正交；训练中自动调节
        return torch.sigmoid(self.alpha) * F.mse_loss(gram, I)