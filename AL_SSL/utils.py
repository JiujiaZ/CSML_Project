import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import reduce
from operator import __or__
import torch
from itertools import chain
from tools import *

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

def entropy(model, train_data, label_status, device, n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 10),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for x,_ in data_loader:
            x = x.to(device)
            prob_y = model.classify(x).detach().cpu().numpy()
            H = (-prob_y * np.log(prob_y+1e-8)).sum(axis=-1)
            acquisition.append(H)
    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def batch_entropy(model, train_data, label_status, device, n_per_class = 1):
    acquisition = list()
    preds = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 10),
                                              sampler=SubsetRandomSampler(pool_idx[0]))
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            prob_y = model.classify(x).detach().cpu().numpy()
            H = (-prob_y * np.log(prob_y + 1e-8)).sum(axis=-1)
            acquisition.append(H)
            preds.append(prob_y.argmax(1))
    acquisition = np.array(list(chain(*acquisition)))
    preds = np.array(list(chain(*preds)))

    sorted_idx = (-acquisition).argsort()
    sorted_preds = preds[sorted_idx]
    print('top preds:', sorted_preds[:model.y_dim])

    unique_preds = np.unique(sorted_preds)

    # ensure diversity
    if (len(unique_preds) == model.y_dim):
        diverse_idx = [np.where(sorted_preds == i)[0][:n_per_class] for i in range(model.y_dim)]
        select_idx = np.asarray(sorted_idx)[np.asarray(diverse_idx)]
    else:
        print('model doesnot preds all class')
        print(unique_preds)
        if ((model.y_dim - len(unique_preds)) > len(unique_preds)):
            expanded_preds = np.random.choice(unique_preds, size=(model.y_dim - len(unique_preds)), replace=True)
        else:
            expanded_preds = np.random.choice(unique_preds, size=(model.y_dim - len(unique_preds)), replace=False)
        expanded_preds = np.append(unique_preds, expanded_preds)
        print(expanded_preds)
        diverse_idx = [np.where(sorted_preds == i)[0][:(expanded_preds == i).sum()] for i in unique_preds]
        diverse_idx = np.array(list(chain(*diverse_idx)))
        select_idx = np.asarray(sorted_idx)[diverse_idx]

    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def bald(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 100),
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

            H = -prob_y.mul(torch.log(prob_y)).sum(1)
            E_H = -prob_ys.mul(torch.log(prob_ys)).sum(1).mean(-1)

            Is = (H-E_H).mean(-1)
            acquisition.append(Is.detach().cpu().numpy())

    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def Gen_uncertainty(model, train_data, label_status, device, n_instances = 1, mc_num = 10):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 100),
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

            x_recon = model.decoder(zs, ys).view(mc_num, model.y_dim * batch_size, -1)
            Var = (torch.pow(x_recon, 2).mean(0) - torch.pow(x_recon.mean(0),2)).view(model.y_dim, batch_size, -1).mean([0,2])
            acquisition.append(Var.detach().cpu().numpy())

    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx

def GenCH(model, train_data, label_status, device, n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 100),
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
            eps = torch.randn_like(z_mu).to(model.device)
            zs = eps.mul(z_std).add_(z_mu)
            x_recon = model.decoder(zs, ys)

            prob_y = model.classify(x_recon)
            H = (-ys * torch.log(prob_y+1e-8)).sum(-1)
            H = torch.transpose(H.view(model.y_dim,batch_size), 0, 1)
            H = H.mean(-1).detach().cpu().numpy()

            acquisition.append(H)
    acquisition = np.array(list(chain(*acquisition)))
    select_idx = (-acquisition).argsort()[:n_instances]
    query_idx = np.asarray(pool_idx[0])[select_idx]
    return query_idx


def GenH(model, train_data, label_status, device, n_instances = 1):
    acquisition = list()
    pool_idx = np.where(label_status == 0)
    pool_size = np.sum((label_status == 0))
    data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=int(pool_size / 100),
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
            eps = torch.randn_like(z_mu).to(model.device)
            zs = eps.mul(z_std).add_(z_mu)
            x_recon = model.decoder(zs, ys)

            prob_y = model.classify(x_recon)
            H = (-prob_y * torch.log(prob_y+1e-8)).sum(-1)
            H = torch.transpose(H.view(model.y_dim,batch_size), 0, 1)
            H = H.mean(-1).detach().cpu().numpy()

            acquisition.append(H)
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
        new_idx = entropy(model, train_data, label_status, device)
    elif (strategy == 'GenH'):
        new_idx = GenH(model, train_data, label_status, device)
    elif (strategy == 'batch_entropy'):
        new_idx = batch_entropy(model, train_data, label_status, device)
    elif (strategy == 'bald'):
        new_idx = bald(model, train_data, label_status, device, n_instances = 1, mc_num = 10)
    elif (strategy == 'Gen_uncertainty'):
        new_idx = Gen_uncertainty(model, train_data, label_status, device, n_instances=1, mc_num=10)
    elif (strategy == 'GenCH'):
        new_idx = GenCH(model, train_data, label_status, device, n_instances=1)
    else:
        raise ValueError('Strategy not implemented')

    label_status[new_idx] = 1
    return label_status, new_idx

def Transfer_updata(model, x_train, label_status, device,  n=1, strategy = 'random', label_seq = None):
    if (strategy == 'random'):
        p = 1- label_status   # revert orginal one means label avaliable
        p = p/p.sum()
        new_idx = np.random.choice(range(len(label_status)), size = n, replace = False, p = p)
    else:
        new_idx = int(label_seq.pop(0))
    label_status[new_idx] = 1
    return label_status, label_seq


