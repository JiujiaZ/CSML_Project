import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import chain
import numpy as np
from scipy.stats import mode
from tools import log_gaussian, log_standard_gaussian

def kl_gaussian(mu_theta, p_theta, batch_num, prior_mu=0, prior_sd=1):
    ratio = F.softplus(p_theta) / prior_sd
    kl = torch.sum(0.5 * (mu_theta - prior_mu) ** 2 / prior_sd) + 0.5 * torch.sum(ratio) - 0.5 * torch.sum(
        torch.log(ratio)) - len(mu_theta.view(-1)) / 2
    kl = kl / len(mu_theta.view(-1))
    return kl * batch_num

def eval_single_kl(mu_theta, p_theta, mu):
    log_p = log_standard_gaussian(mu)
    log_q = log_gaussian(mu, mu_theta, F.softplus(p_theta))
    return log_q - log_p

def bald(model, x_train, label_status, device,  n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    x_pool = x_train[pool_idx]

    data_loader = DataLoader(x_pool, shuffle=False, batch_size = int(len(x_pool)/6))
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            outputs = (torch.exp(model(x)).permute(2,0,1)).detach().cpu().numpy()
            pc = outputs.mean(axis=0)
            H   = (-pc*np.log(pc)).sum(axis=-1)
            E_H = - np.mean(np.sum(outputs * np.log(outputs+1e-8), axis=-1), axis=0)
            acquisition.append(H - E_H)

    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


def entropy(model, x_train, label_status, device, n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    x_pool = x_train[pool_idx]

    data_loader = DataLoader(x_pool, shuffle=False, batch_size=int(len(x_pool) / 6))
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            outputs = (torch.exp(model(x)).permute(2, 0, 1)).detach().cpu().numpy()
            pc = outputs.mean(axis=0)
            H = (-pc * np.log(pc)).sum(axis=-1)
            acquisition.append(H)
    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def variation_ratio(model, x_train, label_status, device,  n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    x_pool = x_train[pool_idx]

    data_loader = DataLoader(x_pool, shuffle=False, batch_size=int(len(x_pool) / 6))
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            outputs = (torch.exp(model(x)).permute(2, 0, 1)).detach().cpu().numpy() #[mc_num, num, class]
            mc_preds = outputs.argmax(axis = -1)
            mc_mode_count = mode(mc_preds, axis = 0).count[0]
            V = 1 - (mc_mode_count/100)
            acquisition.append(V)
    acquisition = np.array(list(chain(*acquisition)))
    max_val = np.amax(acquisition)
    max_num = int(np.sum(acquisition == max_val)) # might be many multipe data with the same score
    max_entries = (-acquisition).argsort()[:max_num]
    select_idx = np.random.choice(max_entries, size = n_instances, replace=False)
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def mean_std(model, x_train, label_status, device, n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    x_pool = x_train[pool_idx]

    data_loader = DataLoader(x_pool, shuffle=False, batch_size=int(len(x_pool) / 6))
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            outputs = (torch.exp(model(x)).permute(2, 0, 1)).detach().cpu().numpy() #[mc_num, num, class]
            E_p2 = (np.square(outputs)).mean(axis = 0)
            E_p = outputs.mean(axis = 0)
            S = (np.sqrt(E_p2 - np.square(E_p))).sum(axis = -1)
            acquisition.append(S)
    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


