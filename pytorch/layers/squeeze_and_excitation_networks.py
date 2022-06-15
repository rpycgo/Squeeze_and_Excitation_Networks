import torch
import torch.nn as nn

class SqueezeAndExcitationNetworks(nn.Module):
    def __init__(self, reduction_ratio, **kwargs):
        super(SqueezeAndExcitationNetworks, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio                

    def forward(self, x):
        global_average_pooling = torch.mean(x, axis=(0, 1), keepdim=True)
        fc1 = nn.Sequential(
            nn.Linear(in_features=x.shape[-1], out_features=(x.shape[-1]//self.reduction_ratio)),
            nn.ReLU()
        )(global_average_pooling)
        fc2 = nn.Sequential(
            nn.Linear(in_features=(x.shape[-1] // self.reduction_ratio), out_features=x.shape[-1]),
            nn.Sigmoid()
        )(fc1)

        return x * fc2
