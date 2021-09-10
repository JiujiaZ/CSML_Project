from main import BenchMarkClassifierMain
import torch
import numpy as np
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
opt['save_path']='/home/jiuzhang/Final/save/Benchmark_Classifier/'
opt['classifier_gradient']= True
opt['validation_batch_size']=1000
opt['if_regularizer']=True
opt['alpha_coef']=0.1
opt['lr']=1e-3
opt['z_dim']=10
opt['classifier'] = 'none'
opt['batch_size'] = 16
opt['epochs'] = 30


def get_name(opt):
    return opt['data_set']+'_seed'+str(seed)+'_nl' + str(opt['labels_per_class'])

classifier_types = ['additional', 'none']
# select 'label_per_classes':
labels_per_class = np.arange(1, 31)
indx = int(sys.argv[1])
opt['labels_per_class'] = labels_per_class[indx-1]


for seed in range(0,5):
    opt['seed']=seed
    print("--------------------------   Start seed: ", seed, "   ------------------------")
    test_accuracy, _, _ = BenchMarkClassifierMain(opt)
    np.save(opt['save_path']+get_name(opt),test_accuracy)
