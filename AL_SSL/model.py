import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import *
import numpy as np
from tools import *
from utils import *
import math


class ConditionalVAE(nn.Module):
    def __init__(self,  opt):
        super().__init__()
        self.z_dim=opt['z_dim']
        self.device=opt['device']
        self.data_type=opt['data_type']
        self.classifier_gradient=opt['classifier_gradient']
        self.y_dim=10
        self.x_flat_dim=784
        if self.data_type=='binary':
            self.out_channels=1
            self.criterion = lambda real, fake: -(real * torch.log(fake + 1e-8) + (1 - real) * torch.log(1 - fake + 1e-8)).sum([1,2,3])
        elif self.data_type=='grey':
            self.out_channels=3
            self.criterion  = lambda  real,fake :discretized_mix_logistic_loss_1d(real, fake)

        self.en_input_dim = self.x_flat_dim + self.y_dim
        self.de_input_dim = self.y_dim + self.z_dim

        self.encoder=densenet_encoder(input_dim=self.en_input_dim, z_dim=self.z_dim)
        self.decoder=densenet_decoder(o_dim=self.out_channels, z_dim=self.de_input_dim)
        if (opt['classifier'] == 'additional'):
            self.classify = mlp_classifier()
        else:
            self.classify = self.generative_classifier
        self.device=opt['device']
        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())

    def generative_classifier(self,xs):
        batch_size = xs.size(0)
        ys = torch.from_numpy(np.arange(self.y_dim))
        ys = ys.view(-1,1).repeat(1, batch_size).view(-1)
        ys = one_hot(ys, self.y_dim)
        ys = Variable(ys.float())
        ys = ys.to(self.device) if xs.is_cuda else ys
        xs = xs.repeat(self.y_dim, 1, 1, 1)
        z_mu, z_std = self.encoder(xs, ys)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        x_recon = self.decoder(zs, ys)
        loglikelihood = -self.criterion(xs, x_recon)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        c_likelihood=torch.transpose((loglikelihood-kl-math.log(self.y_dim)).view(self.y_dim,batch_size),0,1)
        return F.softmax(c_likelihood, dim=-1)


    # Calculating ELBO (Both labelled or unlabelled)
    def forward(self, x, y=None):
        labelled = False if y is None else True
        # Duplicating samples and generate labels if not labelled
        xs, ys = (x, y)
        if not labelled:
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(self.y_dim))
            ys = ys.view(-1,1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, self.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(self.device) if xs.is_cuda else ys
            xs = xs.repeat(self.y_dim, 1, 1, 1)
        
        # Reconstruction
        z_mu, z_std = self.encoder(xs, ys)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        x_recon = self.decoder(zs, ys)

        # p(x|y,z)
        
        loglikelihood = -self.criterion(xs, x_recon)

        # p(y)
        logprior_y  = -math.log(self.y_dim)

        # KL(q(z|x,y)||p(z))
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))

        # ELBO : -L(x,y)
        neg_L = loglikelihood + logprior_y - kl
        #print(loglikelihood.mean(), logprior_y,  -kl.mean())

        if labelled:
            return torch.mean(neg_L)

        if self.classifier_gradient==False:
            with torch.no_grad():
                prob_y = self.classify(x)
        else:
            prob_y = self.classify(x)
        neg_L = neg_L.view_as(prob_y.t()).t()

        # H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(prob_y, torch.log(prob_y + 1e-8)), dim=-1)
        neg_L = torch.sum(torch.mul(prob_y, neg_L), dim=-1)

        # ELBO : -U(x)
        neg_U = neg_L + H
        return torch.mean(neg_U)








    
    
    
    
