import torch
from main import *
import sys

opt={}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt['device']= torch.device('cuda')
    opt['if_cuda']=True
else:
    opt['device']= torch.device('cpu')
    opt['if_cuda']=False

opt['labels_per_class']=1
opt['data_set']='binary_MNIST'
opt['dataset_path']='/home/jiuzhang/Final/save/'
opt['save_path']='/home/jiuzhang/Final/save/AL/'
opt['epochs'] = 20
opt['batch_size'] = 16
opt['classifier_gradient']=True
opt['validation_batch_size']=1000
opt['if_regularizer']=True
opt['alpha_coef']=0.1
opt['lr']=1e-3
opt['z_dim']=10
opt['classifier'] = 'none'
opt['Al_cycle'] = 300
opt['AL_n'] = 1

seeds = np.arange(0,5,1)
AL_strategies = ['random', 'bald',  'entropy', 'variation_ratio', 'mean_std']
# select seed and al_strategy:
indx = int(sys.argv[1])

seed_indx = (indx-1) // len(AL_strategies)
AL_indx = (indx-1) % len(AL_strategies)

opt['seed']=seeds[seed_indx]
opt['AL_strategy'] = AL_strategies[AL_indx]

def get_name(opt):
    return opt['data_set']+'_seed'+str(opt['seed'])+'_'+opt['AL_strategy']

print('seed: {}, AL_fun {}'.format(opt['seed'], opt['AL_strategy']))
test_accuracy, label_seq, model=ActiveLearningMain(opt)

with open(opt['save_path']+get_name(opt)+'.npy', 'wb') as f:
    np.save(f, test_accuracy)
    np.save(f, label_seq)

