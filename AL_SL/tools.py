import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


def filter_79(x, y):
    keep = (y == 7) | (y == 9)
    x, y = x[keep], y[keep]
    y[y==7] = 0
    y[y==9] = 1
    return x,y

def data_in_range(x):
    # x:[num, feature_num]
    min_val = x.min(axis=0, keepdims=True)
    max_val = x.max(axis=0, keepdims=True)
    return (x - min_val) / (max_val - min_val)
def get_mnist_data(location, n_label=1, flatten = False, one_hot = False):

    if (flatten and one_hot):
        #transform = lambda x: ToTensor()(x).view(-1).bernoulli()
        transform  = lambda x: ToTensor()(x).view(-1)
        mnist_train = datasets.MNIST(location, train=True, download=True, transform=transform, target_transform=onehot(10))
        mnist_test = datasets.MNIST(location, train=False, download=True, transform=transform, target_transform=onehot(10))

    elif (flatten or one_hot):
        if flatten:
            transform = lambda x: ToTensor()(x).view(-1)
            mnist_train = datasets.MNIST(location, train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST(location, train=False, download=True, transform=transform)

        else:
            mnist_train = datasets.MNIST(location, train=True, download=True, transform=ToTensor(), target_transform=onehot(10))
            mnist_test = datasets.MNIST(location, train=False, download=True, transform=ToTensor(), target_transform=onehot(10))

    else:
        mnist_train = datasets.MNIST(location, train=True, download=True, transform=ToTensor())
        mnist_test = datasets.MNIST(location, train=False, download=True, transform=ToTensor())

    # zip together
    train_loader = DataLoader(mnist_train, shuffle=False, batch_size=len(mnist_train))
    x_train, y_train = next(iter(train_loader))
    #x_train, y_train = filter_79(x_train, y_train)
    test_loader = DataLoader(mnist_test, shuffle=False, batch_size=len(mnist_test))
    x_test, y_test = next(iter(test_loader))
    #x_test, y_test = filter_79(x_test, y_test)
    #train_info = list(zip(x_train, y_train, label_status))

    # record labelled and unlabelled index:
    labelled_idx = np.random.choice(len(y_train), size=n_label, replace=False)
    label_status = np.zeros(len(y_train))
    label_status[labelled_idx] = 1

    return x_train, y_train, label_status, x_test, y_test


def get_half_moon(n_samples=1000, noise=0.1, n_label=1, random_state=0, one_hot=False, to_plot=False):
    datasets = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X, y = datasets
    X = StandardScaler().fit_transform(X)
    X = X.astype(float)

    X = data_in_range(X)

    if to_plot:
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig, ax = plt.subplots()
        ax.set_title("Input data")

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                   edgecolors='k')

        ax.set_xticks(())
        ax.set_yticks(())

        plt.tight_layout()
        plt.show()

    if one_hot:
        one_hot_y = np.zeros((len(X), 2))
        one_hot_y[y == 0, :] = [1, 0]
        one_hot_y[y == 1, :] = [0, 1]
        y = one_hot_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    labelled_idx = np.random.choice(len(y_train), size=n_label, replace=False)
    label_status = np.zeros(len(y_train))
    label_status[labelled_idx] = 1

    return torch.Tensor(X_train), torch.Tensor(y_train), label_status, torch.Tensor(X_test), torch.Tensor(y_test)


def get_2dgaussian(n_samples=100, n_class=10, n_label=1, noise=0.1, r=1, one_hot=False, to_plot=False):
    theta = (360 / n_class) * (math.pi / 180)
    x_mu = r * np.cos(np.arange(0, n_class) * theta)
    y_mu = r * np.sin(np.arange(0, n_class) * theta)

    X_list = list()
    y_list = list()

    for i in range(n_class):
        center = [x_mu[i], y_mu[i]]
        data = np.random.multivariate_normal(center, [[noise, 0], [0, noise]], n_samples)
        label = np.ones(n_samples) * i
        one_hot_label = onehot(n_class)(i)

        X_list.append(data)
        if one_hot:
            temp = np.repeat(one_hot_label, n_samples).reshape((n_class, -1))
            y_list.append(temp.t())
        else:
            y_list.append(label)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    X = data_in_range(X)

    if to_plot:
        fig, ax = plt.subplots()
        ax.set_title("Input data")
        if one_hot:
            c = np.argmax(y, axis=1)
        else:
            c = y

        ax.scatter(X[:, 0], X[:, 1], c=c, edgecolors='k')

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    labelled_idx = np.random.choice(len(y_train), size=n_label, replace=False)
    label_status = np.zeros(len(y_train))
    label_status[labelled_idx] = 1

    return torch.Tensor(X_train), torch.Tensor(y_train), label_status, torch.Tensor(X_test), torch.Tensor(y_test)


def enumerate_discrete(x, y_dim):

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])
    generated = generated.to(x.device)
    return generated.float()

def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max

def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    return cross_entropy

def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)

def log_gaussian(x, mu, sigma):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param sigma: std of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - torch.log(sigma) - (x - mu)**2 / (2 * (sigma**2))
    return torch.sum(log_pdf, dim=-1)

def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode

class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)

