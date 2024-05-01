from torch.nn import Module
from torch import nn


class FiLM(Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).broadcast_to(x.shape)
        beta = beta.unsqueeze(-1).broadcast_to(x.shape)
        return x * gamma + beta