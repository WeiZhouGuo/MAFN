from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
class Attention_Net(nn.Module):
    def __init__(self,dropout_attention, E_node=32,A_node=2):
        super(Attention_Net, self).__init__()
        self.add_module('linear0',nn.Linear(E_node,A_node))
        self.add_module('tanh0',nn.Tanh())
        self.add_module('softmax',nn.Softmax(dim=1))
        self.dropout_attention=dropout_attention

    def forward(self,E):
        Nk=self.linear0(E)
        Nk=self.tanh0(Nk)
        Nk=self.softmax(Nk)
        return Nk

class Attention_Model(nn.Module):
    def __init__(self, in_size,E_node, dropout_attention):
        self.size=in_size
        super(Attention_Model, self).__init__()
        A_node=2
        for i in range(in_size):
            Attention_layer=Attention_Net(E_node=E_node,A_node=A_node,dropout_attention=dropout_attention)
            self.add_module('Attention_Net%d'%(i+1),Attention_layer)

    def forward(self, E):
        idx=0
        for name,layer in self.named_children():
            new_feature=layer(E)
            if idx==0:
                feature=new_feature[:,1].unsqueeze(dim=1)
            else:
                feature=torch.cat((feature,new_feature[:,1].unsqueeze(dim=1)),dim=1)
            idx=idx+1
        A=torch.squeeze(feature)
        return A


class geneNet(nn.Module):
    def __init__(self, in_size ,dropout):
        super(geneNet, self).__init__()
        hidden_size=in_size
        self.norm_last=nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)


    def forward(self, x):
        y_1=self.linear_1(x)
        y_1=F.elu(y_1)
        return y_1

class cnvNet(nn.Module):
    def __init__(self, in_size ,dropout):
        super(cnvNet, self).__init__()
        hidden_size=in_size
        self.norm_last=nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)


    def forward(self, x):
        y_1=self.linear_1(x)
        y_1=F.elu(y_1)
        return y_1

class clinNet(nn.Module):
    def __init__(self, in_size ,dropout):
        super(clinNet, self).__init__()
        hidden_size=in_size
        self.norm_last=nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)


    def forward(self, x):
        y_1=self.linear_1(x)
        y_1=F.elu(y_1)
        return y_1

class Affinity_net(nn.Module):
    def __init__(self, dim_feature):
        super(Affinity_net, self).__init__()
        in_size = dim_feature[0] + dim_feature[1]
        hidden_size = in_size
        self.dim_feature = dim_feature[0] + dim_feature[1]
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, x,Affinity_matrix):
        x1=torch.mm(x[:,:self.dim_feature],Affinity_matrix)
        y_1 = self.linear_1(x1)
        y_1 = F.elu(y_1)
        return torch.cat([y_1,x[:,self.dim_feature:]],dim=1)

class Net(nn.Module):
    def __init__(self, in_size, E_node, dropout, dropout_attention,dim_feature):
        super(Net, self).__init__()
        self.norm_E = nn.BatchNorm1d(E_node)
        self.E_layer = nn.Linear(in_size, E_node)
        self.Att_modul = Attention_Model(in_size, E_node, dropout_attention)
        self.dim_feature=dim_feature
        self.gene_ful=geneNet(dim_feature[0],dropout)
        self.cnv_ful=cnvNet(dim_feature[1],dropout)
        self.clin_ful=clinNet(dim_feature[2],dropout)
        self.Affinity_net=Affinity_net(dim_feature)

        input=in_size+in_size
        hidden_size = in_size+in_size
        self.norm2=nn.BatchNorm1d(input)
        self.norm_last = nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(input, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 2)

    def forward(self, x,Affinity_matrix):
        E = torch.tanh(self.E_layer(x))
        A = self.Att_modul(E)
        middle_tensor = torch.mul(x, A)
        gene_feature=self.gene_ful(middle_tensor[:,:self.dim_feature[0]])
        cnv_feature=self.cnv_ful(middle_tensor[:,self.dim_feature[0]:self.dim_feature[0]+self.dim_feature[1]])
        clin_feature=self.clin_ful(middle_tensor[:,self.dim_feature[0]+self.dim_feature[1]:])
        Fusion_feature=self.Affinity_net(middle_tensor,Affinity_matrix)
        concate_feature=torch.cat([torch.cat([gene_feature,cnv_feature],dim=-1),clin_feature],dim=-1)

        y_1 = self.linear_1(torch.cat([concate_feature,Fusion_feature],dim=-1))
        y_1 =F.elu(y_1)

        y_2 = self.linear_2(y_1)
        y_2 = F.elu(y_2)
        y_2=self.drop(y_2)
        y_3 = self.linear_3(y_2)
        y_3 = F.elu(y_3)
        y_3=self.drop(y_3)
        y_4 = torch.sigmoid(y_3)
        return y_4,y_3