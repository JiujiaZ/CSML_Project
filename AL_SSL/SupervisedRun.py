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

opt['data_set']='binary_MNIST'
opt['dataset_path']='/home/jiuzhang/Final/save/'
opt['save_path']='/home/jiuzhang/Final/save/SL/'
opt['epochs'] = 30
opt['classifier_gradient']=False
opt['batch_size'] = 16
opt['validation_batch_size']=1000
opt['if_regularizer']=False
opt['alpha']=0.0
opt['lr']=1e-3
opt['z_dim']=10

labels_per_class = np.arange(1, 31)

indx = int(sys.argv[1])
if (indx == 31):
    opt['labels_per_class'] = 6000
else:
    opt['labels_per_class'] = labels_per_class[indx-1]


def get_name(opt):
    return opt['data_set']+'_seed'+str(seed)+'_sl'+ str(opt['labels_per_class'])

for seed in range(0,5):
    opt['seed']=seed
    print("--------------------------   Start seed: ", seed, "   ------------------------")
    test_accuracy=SupervisedMain(opt)
    np.save(opt['save_path']+get_name(opt),test_accuracy)







    