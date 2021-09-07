import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import Bayes_classifier, Bayes_MLP, Bayes_CNN
from metric import bald, entropy, variation_ratio, mean_std
from tools import get_mnist_data


def train_data_update(model, x_train, label_status, device, n=1, strategy = 'random'):
    # for active learning and random sampling
    p = 1 - label_status   # revert original one means label avaliable
    p = p/p.sum()
    if (strategy == 'random'):
        new_idx = np.random.choice(range(len(label_status)), size = n, replace = False, p = p)
    elif (strategy == 'bald'):
        new_idx = bald(model, x_train, label_status, device, n)
    elif (strategy == 'entropy'):
        new_idx = entropy(model, x_train, label_status, device, n)
    elif (strategy == 'variation_ratio'):
        new_idx = variation_ratio(model, x_train, label_status, device, n)
    elif (strategy == 'mean_std'):
        new_idx = mean_std(model, x_train, label_status, device, n)
    else:
        raise ValueError('Strategy not implemented')

    label_status[new_idx] = 1
    return label_status, new_idx

def transfer_update(label_status, label_seq):
    new_idx = int(label_seq.pop(0))
    label_status[new_idx] = 1
    return label_status, label_seq


def select_model(opt):
    if (opt['model'] == 'CNN'):
        return Bayes_CNN()
    elif (opt['model'] == 'Linear'):
        return Bayes_classifier([784, 10])
    elif (opt['model'] == 'H1'):
        return Bayes_MLP([784, 392, 10])
    elif (opt['model'] == 'H2'):
        return Bayes_MLP([784, [392, 196], 10])
    else:
        raise ValueError('Model not implemented')


# Train
def train(model, optimizer, num_epoch,
          train_loader, test_loader, device,
          train_size, early_epoch, tol,
          verbose = False):
    train_ls = list()
    test_acc = 0

    for epoch in range(num_epoch):
        ls = list()
        data_num = 0

        model.train()
        for x, y in train_loader:
            # forward:
            x = x.to(device)
            y = y.to(device)
            # model output : [num, out_feature, mc_num]
            y_pred = model(x.float()).mean(axis = -1)
            loss = model.kl_divergence/train_size + F.cross_entropy(y_pred, y.long(), reduction='mean')
            ls.append(loss.item())

            # backward:
            optimizer.zero_grad()
            loss.backward()

            # update gradient:
            optimizer.step()

            data_num += len(y)

        train_ls.append(np.mean(ls))
        if verbose:
            print(epoch + 1)
            print(f'\tLoss: {train_ls[-1]:.4f}(train)\t')

        if ((epoch > early_epoch) & (epoch % 10 == 0)):
            mean_1 = np.mean(train_ls[epoch - 60:epoch - 10])
            mean_2 = np.mean(train_ls[epoch - 50:])
            if (mean_1 - mean_2 < tol):
                test_acc = eval_acc(model, test_loader, device)
                if verbose:
                    print(f'\tAcc: {test_acc:.4f}(test)\t')
                break

        if (epoch == num_epoch - 1):
            test_acc = eval_acc(model, test_loader, device)
            if verbose:
                print(f'\tAcc: {test_acc:.4f}(test)\t')
    return (train_ls, test_acc)

def eval_acc(model, test_loader, device):
    correct = 0
    data_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_prob = torch.exp(model(x))
            y_sample = y_prob.mean(axis = -1).detach().cpu().numpy()
            y_pred = y_sample.argmax(axis = 1)
            correct += np.equal(y_pred, y).sum().item()
            data_num += len(y)
    return correct/data_num

def al_procedure(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    # get data
    if (opt['data_set']=='MNIST'):
        x_train, y_train, label_status, x_test, y_test = get_mnist_data(opt['dataset_path'], n_label=1, flatten=False,
                                                                             one_hot=False)
    else:
        raise ValueError('Dataset not implemented')

    acc = list()
    indx_hist = [np.where(label_status == 1)]

    print(opt['AL_strategy'], ' in process...')
    for i in range(opt['AL_cycle']):

        train_loader = DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
                                  shuffle=True, batch_size=opt['batch_size'] )
        test_loader = DataLoader(list(zip(x_test, y_test)), shuffle=True, batch_size=int(len(x_test)))


        m = select_model(opt).to(opt['device'])
        optim = torch.optim.Adam(m.parameters(), lr=opt['lr'])
        train_ls, test_acc = train(m, optim, opt['epochs'], train_loader, test_loader, opt['device'],
                                   train_size = label_status.sum(), early_epoch=opt['early_epoch'],
                                   tol = opt['tol'], verbose=opt['verbose'])
        acc.append(test_acc)
        print('data size {}: acc {}'.format(label_status.sum(), test_acc))
        label_status, new_idx = train_data_update(m, x_train, label_status, opt['device'], strategy = opt['AL_strategy'])
        indx_hist.append(new_idx)

    return (acc,indx_hist)


def transfer_AL(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    def get_name(opt):
        return opt['model_A'] + '_seed' + str(opt['seed']) + '_' + opt['AL_strategy']

    # get data
    if (opt['data_set']=='MNIST'):
        x_train, y_train, label_status, x_test, y_test = get_mnist_data(opt['dataset_path'], n_label=0, flatten=False,
                                                                             one_hot=False)
    else:
        raise ValueError('Dataset not implemented')

    acc = list()

    print('transfer labels from CNN to {}'.format(opt['model']))
    print(opt['AL_strategy'], ' in process...')

    file_name = opt['data_path'] + get_name(opt) + '.npy'
    temp = list()
    label_seq = list()
    with open(file_name, 'rb') as f:
        temp.append(np.load(f))
        label_seq.append(np.load(f, allow_pickle=True))

    label_seq = list(label_seq[0])

    for i in range(len(label_seq)):
        label_status, label_seq = transfer_update(label_status, label_seq)

        train_loader = DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
                                  shuffle=True, batch_size=opt['batch_size'] )
        test_loader = DataLoader(list(zip(x_test, y_test)), shuffle=True, batch_size=int(len(x_test)))


        m = select_model(opt).to(opt['device'])
        optim = torch.optim.Adam(m.parameters(), lr=opt['lr'])
        train_ls, test_acc = train(m, optim, opt['epochs'], train_loader, test_loader, opt['device'],
                                   train_size = label_status.sum(), early_epoch=opt['early_epoch'],
                                   tol = opt['tol'], verbose=opt['verbose'])
        acc.append(test_acc)
        print('data size {}: acc {}'.format(label_status.sum(), test_acc))
        print('{} remaining'.format(len(label_seq)))

    return acc

