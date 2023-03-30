import torch.nn as nn
from .resnet.attention import scSE, yoto_block


class MultiLeNet(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720, 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)

    def forward(self, batch):
        x = batch['data']
        x = self.shared(x)
        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))

    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']


class MultiLeNet_attention(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.scse1 = scSE(in_channels=10, mode=['sSE', 'cSE'], h=16)
        self.scse2 = scSE(in_channels=20, mode=['sSE', 'cSE'], h=6)
        self.shared1 =  nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.shared2 =  nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.share3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(720 , 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)

    def forward(self, batch, preference):
        self.scse1.preference(preference)
        self.scse2.preference(preference)
        x = batch['data']
        x = self.shared1(x)
        x = self.scse1(x)
        x = self.shared2(x)
        x = self.scse2(x)
        x = self.share3(x)
        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))


    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']


class MultiLeNet_yoto(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.yoto1 = yoto_block(in_channels=10)
        self.yoto2 = yoto_block(in_channels=20)
        self.shared1 =  nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.shared2 =  nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.share3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(720 , 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)

    def forward(self, batch, preference):
        self.yoto1.preference(preference)
        self.yoto2.preference(preference)
        x = batch['data']
        x = self.shared1(x)
        x = self.yoto1(x)
        x = self.shared2(x)
        x = self.yoto2(x)
        x = self.share3(x)
        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))


    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']

class FullyConnected(nn.Module):


    def __init__(self, dim, **kwargs):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim[0], 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )


    def forward(self, batch):
        x = batch['data']
        return dict(logits=self.f(x))
