import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super.__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x