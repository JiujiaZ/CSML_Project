import sys
from main import transfer_AL
import torch
import numpy as np


opt={}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt['device']= torch.device('cuda')
    opt['if_cuda']=True
else:
    opt['device']= torch.device('cpu')
    opt['if_cuda']=False

opt['data_set']='MNIST'
opt['dataset_path']='/home/jiuzhang/Final/save/'
opt['data_path']='/home/jiuzhang/Final/save/Robustness/'
opt['save_path']='/home/jiuzhang/Final/save/Transfer/'
opt['epochs'] = 1000
opt['early_epoch'] = 300
opt['tol'] = 1e-3
opt['batch_size'] = 256
opt['verbose'] = False
opt['lr']=1e-3
opt['model_A'] = 'CNN' # acquisition model


AL_strategies = ['random', 'bald',  'entropy', 'variation_ratio', 'mean_std']
Models = ['Linear', 'H1', 'H2'] # successor
indx = int(sys.argv[1])

model_indx = (indx-1) % 3
AL_indx = (indx-1) // 3

opt['AL_strategy'] = AL_strategies[AL_indx]
opt['model'] = Models[model_indx]

def get_name(opt):
    return opt['model']+'_seed'+str(seed)+'_'+opt['AL_strategy']

for seed in range(0,5):
    opt['seed']=seed
    print("--------------------------   Start seed: ", seed, "   ------------------------")
    test_accuracy = transfer_AL(opt)

    with open(opt['save_path']+get_name(opt)+'.npy', 'wb') as f:
        np.save(f, test_accuracy)



