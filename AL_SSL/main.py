import torch
from torch import optim
import torch.nn as nn
from utils import *
from model import ConditionalVAE
import numpy as np
import torchvision
from itertools import cycle
from tqdm import tqdm
from torch.utils import data
import timeit


def SupervisedTrain(model,labelled,optimizer,epoch,opt):
    model.train()
    total_loss, accuracy = (0, 0)
    for x,y in tqdm(labelled):
        if opt['data_type']=='binary':
            x = torch.round(x).to(opt['device'])
        else:
            x = rescaling(x).to(opt['device'])
        y = one_hot(y, num_classes=10).to(opt['device'])
        optimizer.zero_grad()

        L = -model(x, y)

        prob_y = model.classify(x)
        if opt['if_regularizer']==True:
            classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
            L+=  opt['alpha'] * classication_loss
        L.backward()
        optimizer.step()
        
        total_loss += L.item()
        accuracy += torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

    m = len(labelled)
    print("Epoch: {}, m = {}".format(epoch, m))
    print("[Train]\t\t L: {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


#def SemiSupervisedTrain(model,labelled,unlabelled,optimizer,epoch,opt,speedy=False):
#    model.train()
#    total_loss, accuracy = (0, 0)
#    unl_acc = 0

#    assert len(unlabelled) > len(labelled)
#    with tqdm(unlabelled, unit="batch") as unlabelled_tqdm:
#        for (x, y), (u, uy) in zip(cycle(labelled), unlabelled_tqdm):
#            unlabelled_tqdm.set_description(f"Epoch {epoch}")
#            current_batch_size = u.size()[0]
#            if opt['data_type']=='binary':
#                x = torch.round(x).to(opt['device'])
#                u = torch.round(u).to(opt['device'])
#            else:
#                x = rescaling(x).to(opt['device'])
#                u = rescaling(u).to(opt['device'])
#            y = one_hot(y, num_classes=10).to(opt['device'])
#            optimizer.zero_grad()

#
#            L = -model(x, y)
#            U = -model(u)

            # Add auxiliary classification loss q(y|x)
#            prob_y = model.classify(x)
#
#            J = L + U
#            # cross entropy
#            if opt['if_regularizer']==True:
#                classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
#                J+= opt['alpha'] * classication_loss
#
#            J.backward()
#            optimizer.step()

#            total_loss += J.item()
#            accuracy += torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

#            if (not speedy):
#                prob_y = model.classify(u)
#                unl_acc += torch.mean((torch.max(prob_y, 1)[1].data.detach().cpu() == uy).float()).item()


#    m = len(unlabelled)
#    print("Epoch: {}".format(epoch))
#    print("[Train/labelled]\t\t J : {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
#    if (not speedy):
#        print("[Train/unlabelled]\t\t J : {:.4f}, accuracy: {:.4f}".format(total_loss / m, unl_acc / m))

#    return total_loss / m, accuracy / m


def SemiSupervisedTrain(model, labelled, unlabelled, optimizer, epoch, opt, speedy=False):
    model.train()
    total_loss, accuracy = (0, 0)
    unl_acc = 0

    assert len(unlabelled) > len(labelled)
    for (x, y), (u, uy) in zip(cycle(labelled), unlabelled):

        if opt['data_type'] == 'binary':
            x = torch.round(x).to(opt['device'])
            u = torch.round(u).to(opt['device'])
        else:
            x = rescaling(x).to(opt['device'])
            u = rescaling(u).to(opt['device'])
        y = one_hot(y, num_classes=10).to(opt['device'])
        optimizer.zero_grad()

        L = -model(x, y)
        U = -model(u)

        # Add auxiliary classification loss q(y|x)
        prob_y = model.classify(x)

        J = L + U
        # cross entropy
        if opt['if_regularizer'] == True:
            classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
            J += opt['alpha'] * classication_loss

        J.backward()
        optimizer.step()

        total_loss += J.item()
        accuracy += torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

        if (not speedy):
            prob_y = model.classify(u)
            unl_acc += torch.mean((torch.max(prob_y, 1)[1].data.detach().cpu() == uy).float()).item()

    m = len(unlabelled)
    print("Epoch: {}".format(epoch))
    print("[Train/labelled]\t\t J : {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
    if (not speedy):
        print("[Train/unlabelled]\t\t J : {:.4f}, accuracy: {:.4f}".format(total_loss / m, unl_acc / m))

    return total_loss / m, accuracy / m



def Test(model,validation,opt):
    model.eval()
    total_loss, accuracy = (0, 0)

    with torch.no_grad():
        for x, y in validation:
            if opt['data_type']=='binary':
                x = torch.round(x).to(opt['device'])
            else:
                x = rescaling(x).to(opt['device'])
            y = one_hot(y, num_classes=10).to(opt['device'])
           
            L = -model(x, y)

            prob_y = model.classify(x)
            if opt['if_regularizer']==True:
                classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
                L+=  opt['alpha'] * classication_loss

            total_loss += L.item()

            _, pred_idx = torch.max(prob_y, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx == lab_idx).float()).item()

        m = len(validation)
        print("[Validation]\t L: {:.4f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))

        return total_loss / m, accuracy / m


def SupervisedMain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,transform=torchvision.transforms.ToTensor())
    test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,transform=torchvision.transforms.ToTensor())

    if opt['data_set']=='binary_MNIST':
        print('binary MNIST')
        opt['data_type']='binary'
    else:
        print('grey MNIST')
        opt['data_type']='grey'

    if (opt['labels_per_class'] == 6000):  # use all data
        labelled = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
        validation = data.DataLoader(test_data, batch_size=opt['validation_batch_size'], shuffle=False)
    else:
        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                               sampler=get_sampler(train_data.targets, opt['labels_per_class']))
        validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                 sampler=get_sampler(test_data.targets))

    model=ConditionalVAE(opt).to(opt['device'])

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    train_loss=[]
    test_loss=[]

    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(1, opt['epochs'] + 1):
        train_l, train_acc = SupervisedTrain(model,labelled,optimizer,epoch,opt)
        test_l, test_acc = Test(model,validation,opt)
        train_loss.append(train_l)

        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    return test_accuracy, model


def SemiSupervisedMain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    if opt['data_set'][-5:]=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,transform=torchvision.transforms.ToTensor())
        
        if opt['data_set']=='binary_MNIST':
            print('binary MNIST')
            opt['data_type']='binary'
        else:
            print('grey MNIST')
            opt['data_type']='grey'

        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                               sampler=get_sampler(train_data.targets,  opt['labels_per_class']))
        unlabelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                                 sampler=get_sampler(train_data.targets))
        validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                 sampler=get_sampler(test_data.targets))

    
    # opt['alpha'] = opt['alpha'] * (len(unlabelled)+len(labelled)) / len(labelled)

    opt['alpha'] = opt['alpha'] * opt['labels_per_class'] * 10

    model=ConditionalVAE(opt).to(opt['device'])

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    train_loss=[]
    test_loss=[]

    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(1, opt['epochs'] + 1):
        train_l, train_acc = SemiSupervisedTrain(model,labelled,unlabelled,optimizer,epoch,opt)
        test_l, test_acc = Test(model,validation,opt)
        train_loss.append(train_l)

        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    return test_accuracy, model


def BenchMarkClassifierMain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    if opt['data_set'][-5:] == 'MNIST':
        train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        if opt['data_set'] == 'binary_MNIST':
            print('binary MNIST')
            opt['data_type'] = 'binary'
        else:
            print('grey MNIST')
            opt['data_type'] = 'grey'

        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                               sampler=get_sampler(train_data.targets, opt['labels_per_class']))
        unlabelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                                 sampler=get_sampler(train_data.targets))
        validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                 sampler=get_sampler(test_data.targets))

    #opt['alpha'] = opt['alpha'] * (len(unlabelled) + len(labelled)) / len(labelled)
    opt['alpha'] = opt['alpha_coef'] * opt['labels_per_class'] * 10

    opt['classifier'] = 'additional'
    model1 = ConditionalVAE(opt).to(opt['device'])
    optimizer1 = optim.Adam(model1.parameters(), lr=opt['lr'])

    opt['classifier'] = 'none'
    model2 = ConditionalVAE(opt).to(opt['device'])
    optimizer2 = optim.Adam(model2.parameters(), lr=opt['lr'])

    train_loss = {}
    train_loss['additional'] = []
    train_loss['none'] = []

    test_loss = {}
    test_loss['additional'] = []
    test_loss['none'] = []

    train_accuracy = {}
    train_accuracy['additional'] = []
    train_accuracy['none'] = []

    test_accuracy = {}
    test_accuracy['additional'] = []
    test_accuracy['none'] = []

    for epoch in range(1, opt['epochs'] + 1):
        print('Additional Classifier: ')
        train_l, train_acc = SemiSupervisedTrain(model1, labelled, unlabelled, optimizer1, epoch, opt)
        test_l, test_acc = Test(model1, validation, opt)
        train_loss['additional'].append(train_l)

        test_loss['additional'].append(test_l)
        train_accuracy['additional'].append(train_acc)
        test_accuracy['additional'].append(test_acc)

        print('None Classifier: ')
        train_l, train_acc = SemiSupervisedTrain(model2, labelled, unlabelled, optimizer2, epoch, opt)
        test_l, test_acc = Test(model2, validation, opt)
        train_loss['none'].append(train_l)

        test_loss['none'].append(test_l)
        train_accuracy['none'].append(train_acc)
        test_accuracy['none'].append(test_acc)

    return test_accuracy, model1, model2


def ActiveLearningMain(opt, inter_save = True):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    def get_drop_last(n, bs):
        if (n % bs == 1):
            return True
        else:
            return False

    if opt['data_set'][-5:] == 'MNIST':
        train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        if opt['data_set'] == 'binary_MNIST':
            print('binary MNIST')
            opt['data_type'] = 'binary'
        else:
            print('grey MNIST')
            opt['data_type'] = 'grey'

        # initial indices:
        sampler = get_sampler(train_data.targets, opt['labels_per_class'])
        label_status = np.zeros(len(train_data))
        label_status[sampler.indices] = 1

        opt['drop_last'] = get_drop_last(label_status.sum(), opt['batch_size'])
        #labelled = torch.utils.data.DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
        #    batch_size=opt['batch_size'], pin_memory=False, drop_last=opt['drop_last'])
        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
                                                 pin_memory=False, sampler = sampler)
        unlabelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                                 sampler=get_sampler(train_data.targets))
        validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                 sampler=get_sampler(test_data.targets))

    model = ConditionalVAE(opt).to(opt['device'])
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    #train_loss = []
    test_loss = []

    #train_accuracy = []
    test_accuracy = []

    label_seq = []
    label_seq.append(sampler.indices)

    indices = sampler.indices

    for AL_cycle in range(0, opt['Al_cycle'] + 1):
        print('AL_cycle: {}, trainning size: {:.0f}'.format(AL_cycle, label_status.sum()))
        opt['alpha'] = opt['alpha_coef'] * label_status.sum()
        start = timeit.default_timer()
        for epoch in range(1, opt['epochs'] + 1):
            train_l, train_acc = SemiSupervisedTrain(model, labelled, unlabelled, optimizer, epoch, opt, speedy=True)

            if (epoch == opt['epochs']):
                test_l, test_acc = Test(model, validation, opt)
                test_loss.append(test_l)
                test_accuracy.append(test_acc)
                end = timeit.default_timer()
                print('train time: ', (end - start) / 60)

            #train_loss.append(train_l)
            #train_accuracy.append(train_acc)

        # update labelling pools for next training:
        label_status, new_idx = AL_update(model, train_data, label_status, opt['device'], n=opt['AL_n'], strategy=opt['AL_strategy'])
        opt['drop_last'] = get_drop_last(label_status.sum(), opt['batch_size'])
        #labelled = torch.utils.data.DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
        #                                       batch_size=opt['batch_size'], pin_memory=False, drop_last=opt['drop_last'])
        label_seq.append(new_idx)
        indices = np.append(indices, new_idx)
        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
                                               pin_memory=False, drop_last=opt['drop_last'],
                                               sampler=SubsetRandomSampler(indices))

        #save intermediate results
        #if (inter_save and ((AL_cycle % 10) == 0)):
        if inter_save:
            with open(opt['save_path'] + 'inter_' + opt['AL_strategy']+'.npy', 'wb') as f:
                np.save(f, test_accuracy)
                np.save(f, label_seq)

        # re - initialise model
        if (AL_cycle != opt['Al_cycle']):
            model = ConditionalVAE(opt).to(opt['device'])
            optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    return test_accuracy, label_seq, model


def LazyALMain(opt, inter_save = True):
    # no initialisation at each AL cycle
    # doesnt work as previous parameters are screwed for current datset
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    def get_drop_last(n, bs):
        if (n % bs == 1): # due to normalisation
            return True
        else:
            return False

    if opt['data_set'][-5:] == 'MNIST':
        train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())

        if opt['data_set'] == 'binary_MNIST':
            print('binary MNIST')
            opt['data_type'] = 'binary'
        else:
            print('grey MNIST')
            opt['data_type'] = 'grey'

        # initial indices:
        sampler = get_sampler(train_data.targets, opt['labels_per_class'])
        label_status = np.zeros(len(train_data))
        label_status[sampler.indices] = 1

        opt['drop_last'] = get_drop_last(label_status.sum(), opt['batch_size'])
        #labelled = torch.utils.data.DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
        #    batch_size=opt['batch_size'], pin_memory=False, drop_last=opt['drop_last'])
        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
                                                 pin_memory=False, sampler = sampler)
        unlabelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], pin_memory=False,
                                                 sampler=get_sampler(train_data.targets))
        validation = torch.utils.data.DataLoader(test_data, batch_size=opt['validation_batch_size'], pin_memory=False,
                                                 sampler=get_sampler(test_data.targets))

    model = ConditionalVAE(opt).to(opt['device'])
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    #train_loss = []
    test_loss = []

    #train_accuracy = []
    test_accuracy = []

    label_seq = []
    label_seq.append(sampler.indices)

    indices = sampler.indices

    for AL_cycle in range(0, opt['Al_cycle'] + 1):
        print('AL_cycle: {}, trainning size: {:.0f}'.format(AL_cycle, label_status.sum()))
        opt['alpha'] = opt['alpha_coef'] * label_status.sum()
        start = timeit.default_timer()

        if (AL_cycle == 0):
            for epoch in range(1, opt['ini_epochs'] + 1):
                train_l, train_acc = SemiSupervisedTrain(model, labelled, unlabelled, optimizer, epoch, opt, speedy=True)

                if (epoch == opt['ini_epochs']):
                    test_l, test_acc = Test(model, validation, opt)
                    test_loss.append(test_l)
                    test_accuracy.append(test_acc)
                    end = timeit.default_timer()
                    print('train time: ', (end - start) / 60)
        else:
            for epoch in range(1, opt['epochs'] + 1):
                train_l, train_acc = SemiSupervisedTrain(model, labelled, unlabelled, optimizer, epoch, opt,
                                                         speedy=True)

                if (epoch == opt['epochs']) and (not np.isnan(train_l)):
                    test_l, test_acc = Test(model, validation, opt)
                    test_loss.append(test_l)
                    test_accuracy.append(test_acc)
                    end = timeit.default_timer()
                    print('train time: ', (end - start) / 60)

            # re train - if lazy update is not good
            if (np.isnan(train_l)):
                print('AL_cycle {} : model re-initialised'.format(AL_cycle))
                model = ConditionalVAE(opt).to(opt['device'])
                optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

                for epoch in range(1, opt['ini_epochs'] + 1):
                    train_l, train_acc = SemiSupervisedTrain(model, labelled, unlabelled, optimizer, epoch, opt,
                                                             speedy=True)
                    if (epoch == opt['ini_epochs']):
                        test_l, test_acc = Test(model, validation, opt)
                        test_loss.append(test_l)
                        test_accuracy.append(test_acc)
                        end = timeit.default_timer()
                        print('train time: ', (end - start) / 60)



            #train_loss.append(train_l)
            #train_accuracy.append(train_acc)

        # update labelling pools for next training:
        label_status, new_idx = AL_update(model, train_data, label_status, opt['device'], n=opt['AL_n'], strategy=opt['AL_strategy'])
        opt['drop_last'] = get_drop_last(label_status.sum(), opt['batch_size'])
        #labelled = torch.utils.data.DataLoader(list(zip(x_train[label_status == 1], y_train[label_status == 1])),
        #                                       batch_size=opt['batch_size'], pin_memory=False, drop_last=opt['drop_last'])
        label_seq.append(new_idx)
        indices = np.append(indices, new_idx)
        labelled = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'],
                                               pin_memory=False, drop_last=opt['drop_last'],
                                               sampler=SubsetRandomSampler(indices))

        #save intermediate results
        #if (inter_save and ((AL_cycle % 10) == 0)):
        if inter_save:
            with open(opt['save_path'] + 'inter_' + opt['AL_strategy']+'.npy', 'wb') as f:
                np.save(f, test_accuracy)
                np.save(f, label_seq)

    return test_accuracy, label_seq, model






    