import torch
import torch.nn as nn


class scSE(nn.Module):
    def __init__(self, in_channels, mode=None, h=None):
        '''
        :param in_channels:
        :param preference_vector:
        :param mode[List]: [0, 0]-> don't use scse, [1, 1]-> use sSE, [2, 2]-> use cSE, [1, 2]-> use scSE
        '''
        super().__init__()
        self.pre = None
        self.mode = mode
        # sSE
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm_s = nn.Sigmoid()
        # cSE

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1,
                                      bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1,
                                         bias=False)
        self.norm_c = nn.Sigmoid()
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 30)
        )
        setattr(self, f"channel_attention", nn.Linear(30, in_channels))
        setattr(self, f"sptial_attention", nn.Linear(30, h*h))

    def preference(self, preference_vector):
        self.pre = preference_vector

    def forward(self, U):

        v = self.ray_mlp(self.pre)
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        if 'cSE' in self.mode:
            ca = getattr(self, f"channel_attention")(v).reshape(-1, 1, 1)
            z = z * ca
        z = self.norm_c(z)
        if 'sSE' in self.mode:
            q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
            sa = getattr(self, f"sptial_attention")(v).reshape(U.shape[-1], U.shape[-2])
            q = q * sa
            q = self.norm_s(q)
        if self.mode == ['sSE', 'cSE']:
            return U * z.expand_as(U) + U * q
        elif 'cSE' in self.mode:
            return U * z.expand_as(U)
        else:
            return U * q


class yoto_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pre = None
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 30)
        )
        setattr(self, f"channel", nn.Linear(30, in_channels))
        setattr(self, f"bias", nn.Linear(30, in_channels))

    def preference(self, preference_vector):
        self.pre = preference_vector

    def forward(self, U):

        v = self.ray_mlp(self.pre)
        weight = getattr(self, f'channel')(v).reshape(-1, 1, 1)
        bias = getattr(self, f'bias')(v).reshape(-1, 1, 1)

        z = U*weight + bias

        return z





