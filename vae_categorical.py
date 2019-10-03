from __future__ import division

import torch
import pdb
import argparse
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn

class CATVAE(nn.Module):
    """
           Categorical Variational Autoencoder
           K: Number of Cateories or Classes
           N: Number of Categorical distributions
           N x K: Dimension of latent variable
           hidden_layers: A list containing number of nodes in each hidden layers
                           of both encoder and decoder
           """
    def __init__(self,xdim,N,K,temperature,nhin=[],nhout=[]):
        super(CATVAE, self).__init__()
        self.N=N
        self.K=K
        self.nhin=nhin
        self.nhout=nhout
        self.temperature=temperature
        self.xdim=xdim
        self.encoderlayers = nn.ModuleList()
        if len(nhin) > 0:
            self.encoderlayers.append(nn.Linear(xdim,nhin[0]).double())
            for i in range(len(nhin)-1):
                self.encoderlayers.append(nn.Linear(nhin[i],nhin[i+1]).double())

            self.encoderlast=nn.Linear(nhin[-1],N*K).double()

        self.decoderlayers = nn.ModuleList()
        if len(nhin) > 0:
            self.decoderlayers.append(nn.Linear(N*K,nhout[0]).double())
            for i in range(len(nhout)-1):
                self.decoderlayers.append(nn.Linear(nhout[i],nhout[i+1]).double())

            self.decoderlast=nn.Linear(nhout[-1],xdim).double()


    def forward(self, x):
        logits_z, q_z, log_qz, z= self.encoder(x)
        logits_x = self.decoder(z)
        return logits_x,log_qz,q_z

    def generator(self,x):
        logits_z, q_z, log_qz, z= self.encoder(x)
        logits_x = self.decoder(z)
        r=torch.sigmoid(logits_x)
        xknock=torch.bernoulli(r)
        return xknock


    def gumbel_sample(self,logits_z, tou, K):
        eps = 1e-20
        shape=logits_z.size()
        U = torch.rand(shape).double()
        # for doing operation between Variable and Tensor, a tensor has to be wrapped
        # insider Variable. However, set requires_grad as False so that back propagation doesn't
        # pass through it
        # gumbel sample is -log(-log(U))
        g = logits_z-torch.log(-torch.log(U + eps) + eps)/tou
        z = torch.exp(g)/torch.sum(torch.exp(g),dim=-1).view(-1,self.N,1)
        #z = F.softmax((logits_z + g) / tou, dim=-1).double() #this is a N * K * mb_size

        return z


    def encoder(self,x):
        for layer in self.encoderlayers:
            x = F.relu(layer(x))

        x=self.encoderlast(x)
        logits_z=torch.reshape(x,(-1,self.N,self.K))
        tou = self.temperature
        #pdb.set_trace()
        z=self.gumbel_sample(logits_z,tou,self.K)
        q_z= F.softmax(logits_z,dim=-1)
        log_qz = torch.log(q_z+1e-20)
        return logits_z, q_z, log_qz, z

    def decoder(self,z):
        z=z.view(-1,self.N*self.K)
        #pdb.set_trace()
        for layer in self.decoderlayers:
            z= F.relu(layer(z))

        logits_x=self.decoderlast(z)
        return logits_x



def vae_loss(x,logits_x, q_z, log_qz, model):
    p_x=torch.distributions.Bernoulli(logits=logits_x)
    #log_px = -x * torch.log((1 + torch.exp(-logits_x))) - (1-x) *(1 + torch.exp(logits_x))
    kl_tmp = (q_z * (log_qz - torch.log(torch.DoubleTensor([1/model.K])))).view(-1,model.N, model.K)
    KL=torch.sum(kl_tmp, [1,2])
    #shape=x.size()
    #elbo=torch.sum(log_px,1)-KL
    elbo=torch.sum(p_x.log_prob(x),dim=1) -KL
    loss=torch.mean(-elbo)
    return loss





