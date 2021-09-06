import torch
from torch import nn
import torch.nn.functional as F


class LinearVariational(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bayes_kl = 0

        # mean:
        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_dim, out_dim).normal_(mean=0, std=0.1)
        )

        # variance log(1 + exp(p))◦ eps:
        self.w_p = nn.Parameter(
            torch.FloatTensor(in_dim, out_dim).normal_(mean=1, std=0.001)
        )

    def reparameterize(self, mu, p, mc_num):
        # variance log(1 + exp(p))◦ eps:
        sigma = F.softplus(p)
        dim = torch.cat((torch.tensor(sigma.shape), torch.tensor(mc_num).view(1)))  # [in_feature, out_feature, mc_num]
        eps = torch.randn(tuple(dim)).to(sigma.device)

        return mu.unsqueeze(axis=-1) + (sigma.unsqueeze(axis=-1) * eps)

    def forward(self, x, mc_num=100):
        w = self.reparameterize(self.w_mu, self.w_p, mc_num).permute(2, 1, 0)  # [mc_num, out_feature, in_feature]

        # x: [num, in_feature], z: [mc_num, out_feature, num]
        z = torch.matmul(w, x.t())

        # return size: [num, out_feature, mc_num]
        # print(z.shape)
        return z.permute(2, 1, 0), self.w_mu, self.w_p


class Latent_sample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # layer for mean:
        self.w_mu = nn.Linear(in_dim, out_dim)
        # layer for variance:
        self.w_p = nn.Linear(in_dim, out_dim)

    def reparameterize(self, mu, p):
        # variance log(1 + exp(p))◦ eps:
        sigma = F.softplus(p)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x):
        z_mu = self.w_mu(x)
        z_p = self.w_p(x)
        sample = self.reparameterize(z_mu, z_p)
        return (sample, z_mu, z_p)

