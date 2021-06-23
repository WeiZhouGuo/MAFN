from __future__ import print_function
import torch
import torch.nn as nn
class My_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.k=k

    def forward(self, X,output, y_std):
        a=0.5
        self.gamma = 2.0
        self.alpha = 0.75
        output=torch.reshape(output,[-1,1])
        y_std=torch.reshape(y_std,[-1,1])
        loss1=-torch.mean(a*y_std*torch.log(torch.clamp(output,1e-10, 1.0))+(1-a)*(1. - y_std) *torch.log(torch.clamp(1. - output, 1e-10, 1.0)))
        return loss1

