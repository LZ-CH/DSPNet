import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
eps = 1e-6
class FC(nn.Module):
    def __init__(self, in_size, out_size, pdrop=0., use_gelu=True):
        super(FC, self).__init__()
        self.pdrop = pdrop
        self.use_gelu = use_gelu
        self.linear = nn.Sequential(nn.Linear(in_size, out_size),
                                    nn.LayerNorm(out_size)) 
        if use_gelu:
            self.gelu = nn.GELU()
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.linear(x)
        if self.use_gelu:
            x = self.gelu(x)
        x = self.dropout(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, pdrop=0., use_gelu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, pdrop=pdrop, use_gelu=use_gelu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size=512, flat_glimpses=1, flat_out_size=1024, pdrop=0.1):
        """
        Args:
            hidden_size (int): The input dimension.
            flat_mlp_size (int, optional): The hidden size of the MLP. Defaults to 512.
            flat_glimpses (int, optional): The number of glimpses. Defaults to 1.
            flat_out_size (int, optional): The output dimension. Defaults to 1024.
            pdrop (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            pdrop=pdrop,
            use_gelu=True
        )
        self.flat_glimpses = flat_glimpses

        self.linear_merge = nn.Sequential(nn.Linear(hidden_size * flat_glimpses,flat_out_size),
                                          nn.BatchNorm1d(flat_out_size)
                                          ) 

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.logical_not().squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted