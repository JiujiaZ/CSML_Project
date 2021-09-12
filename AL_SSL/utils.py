import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from operator import __or__
import torch
from itertools import chain
from tools import *
from scipy.stats import mode

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def plot_latent(autoencoder, data, device, num_batches=500):
    for i, (x, y) in enumerate(data):
        z,_ = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.plot()
    plt.show()
    
def gray_show_many(image,number_sqrt):
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
    
def color_show_many(image,number_sqrt,dim=32, channels=3):
    image=image.view(-1,3,32,32).permute(0,2,3,1)
    canvas_recon = np.empty((dim * number_sqrt, dim * number_sqrt, channels))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim,:] = \
            image[count]
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon)
    plt.show()


def get_sampler(labels, n=None):
    # Choose digits in 0-9 
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(10)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(10)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)

    return sampler


def entropy(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 600),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for xs, _ in data_loader:
            xs = xs.to(device)
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(model.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, model.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(model.device) if xs.is_cuda else ys
            xs = xs.repeat(model.y_dim, 1, 1, 1)

            # Reconstruction
            z_mu, z_std = model.encoder(xs, ys)
            eps = torch.randn_like(z_mu.repeat(mc_num, 1, 1)).to(model.device)  # [mc_num, num, z_dim]
            zs = eps.mul(z_std.unsqueeze(0)).add_(z_mu.unsqueeze(0)).to(model.device)

            zs = zs.view(-1, z_mu.shape[1])
            ys = ys.repeat(mc_num, 1)

            x_recon = model.decoder(zs, ys)  # (mc_num * model.y_dim * batch_size, chanel, 28, 28)
            prob_ys = model.classify(x_recon).view(mc_num, model.y_dim, batch_size, -1).permute(2, 3, 1,
                                                                                                0)  # [num, y_dim(classify), con_y, mc_num]
            prob_y = prob_ys.mean(-1)

            H = -prob_y.mul(torch.log(prob_y+1e-8)).sum(1).mean(-1)

            acquisition.append(H.detach().cpu().numpy())
    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def bald(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 600),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for xs, _ in data_loader:
            xs = xs.to(device)
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(model.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, model.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(model.device) if xs.is_cuda else ys
            xs = xs.repeat(model.y_dim, 1, 1, 1)

            # Reconstruction
            z_mu, z_std = model.encoder(xs, ys)
            eps = torch.randn_like(z_mu.repeat(mc_num,1,1)).to(model.device) #[mc_num, num, z_dim]
            zs = eps.mul(z_std.unsqueeze(0)).add_(z_mu.unsqueeze(0)).to(model.device)

            zs = zs.view(-1, z_mu.shape[1])
            ys = ys.repeat(mc_num, 1)

            x_recon = model.decoder(zs, ys) # (mc_num * model.y_dim * batch_size, chanel, 28, 28)
            prob_ys = model.classify(x_recon).view(mc_num, model.y_dim, batch_size, -1).permute(2,3,1,0) #[num, y_dim(classify), con_y, mc_num]
            prob_y = prob_ys.mean(-1)

            H = -prob_y.mul(torch.log(prob_y+1e-8)).sum(1)
            E_H = -prob_ys.mul(torch.log(prob_ys+1e-8)).sum(1).mean(-1)

            Is = (H-E_H).mean(-1)
            acquisition.append(Is.detach().cpu().numpy())

    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


def variation_ratio(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 600),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for xs, _ in data_loader:
            xs = xs.to(device)
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(model.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, model.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(model.device) if xs.is_cuda else ys
            xs = xs.repeat(model.y_dim, 1, 1, 1)

            # Reconstruction
            z_mu, z_std = model.encoder(xs, ys)
            eps = torch.randn_like(z_mu.repeat(mc_num,1,1)).to(model.device) #[mc_num, num, z_dim]
            zs = eps.mul(z_std.unsqueeze(0)).add_(z_mu.unsqueeze(0)).to(model.device)

            zs = zs.view(-1, z_mu.shape[1])
            ys = ys.repeat(mc_num, 1)

            x_recon = model.decoder(zs, ys) # (mc_num * model.y_dim * batch_size, chanel, 28, 28)
            prob_ys = model.classify(x_recon).view(mc_num, model.y_dim, batch_size, -1).permute(2,3,1,0) #[num, y_dim(classify), con_y, mc_num]
            prob_ys = prob_ys.detach().cpu()
            mc_preds = prob_ys.argmax(dim=1)
            mc_mode_count = mode(mc_preds, axis=2).count
            V = 1 - np.hstack(mc_mode_count)/mc_num #[con_y, num]
            V = V.mean(axis=0)

            acquisition.append(V)

    acquisition = np.array(list(chain(*acquisition)))
    max_val = np.amax(acquisition)
    max_num = int(np.sum(acquisition == max_val))  # might be many multipe data with the same score
    max_entries = (-acquisition).argsort()[:max_num]
    select_idx = np.random.choice(max_entries, size=n_instances, replace=False)
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


def mean_std(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 600),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for xs, _ in data_loader:
            xs = xs.to(device)
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(model.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = one_hot(ys, model.y_dim)
            ys = Variable(ys.float())
            ys = ys.to(model.device) if xs.is_cuda else ys
            xs = xs.repeat(model.y_dim, 1, 1, 1)

            # Reconstruction
            z_mu, z_std = model.encoder(xs, ys)
            eps = torch.randn_like(z_mu.repeat(mc_num,1,1)).to(model.device) #[mc_num, num, z_dim]
            zs = eps.mul(z_std.unsqueeze(0)).add_(z_mu.unsqueeze(0)).to(model.device)

            zs = zs.view(-1, z_mu.shape[1])
            ys = ys.repeat(mc_num, 1)

            x_recon = model.decoder(zs, ys) # (mc_num * model.y_dim * batch_size, chanel, 28, 28)
            prob_ys = model.classify(x_recon).view(mc_num, model.y_dim, batch_size, -1).permute(2,3,1,0) #[num, y_dim(classify), con_y, mc_num]
            prob_ys = prob_ys.detach().cpu().numpy()

            E_p2 = (np.square(prob_ys)).mean(axis=-1)
            E_p = prob_ys.mean(axis=-1)
            S = (np.sqrt(E_p2 - np.square(E_p))).sum(axis=1).mean(axis=-1)

            acquisition.append(S)

    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


def AL_update(model, train_data, label_status, device,  n=1, strategy = 'random'):
    if (strategy == 'random'):
        p = 1 - label_status   # revert original one means label avaliable
        p = p/p.sum()
        new_idx = np.random.choice(range(len(label_status)), size = n, replace = False, p = p)
    elif (strategy == 'entropy'):
        new_idx = entropy(model, train_data, label_status, device, n_instances = 1, mc_num = 10)
    elif (strategy == 'bald'):
        new_idx = bald(model, train_data, label_status, device, n_instances = 1, mc_num = 10)
    elif (strategy == 'variation_ratio'):
        new_idx = variation_ratio(model, train_data, label_status, device, n_instances = 1, mc_num = 10)
    elif (strategy == 'mean_std'):
        new_idx = mean_std(model, train_data, label_status, device, n_instances = 1, mc_num = 10)
    else:
        raise ValueError('Strategy not implemented')

    label_status[new_idx] = 1
    return label_status, new_idx



