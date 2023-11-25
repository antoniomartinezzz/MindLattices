import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import random
import glob
from BrainGNN.imports.ABIDEDataset import ABIDEDataset
from BrainGNN.net.braingnn import Network
from BrainGNN.imports.utils import train_val_test_split
from BrainGNN.train_BrainGNN import train, test_acc, test_loss

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--parcellation', type=str, default='OURS', help='CC200, HO or OURS')
parser.add_argument('--mode', type=str, default='test',  help='test, train or demo')
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--repetitions', type=int, default=1, help='How many repetions of 5-fold cross validation training to run')

opt = parser.parse_args()

nclass = 2
K_folds = 5
name = 'ABIDE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/media/SSD2/MindGraphs/ABIDE_data/data_ours_D16_S8/ABIDE_pcp/cpac/filt_noglobal'
nroi = 288
indim = 288

if opt.parcellation == 'CC200':
    path = '/media/SSD2/MindGraphs/ABIDE_data/data_CC200/ABIDE_pcp/cpac/filt_noglobal'
    nroi = 200
    indim = 200
elif opt.parcellation == 'HO':
    path = '/media/SSD2/MindGraphs/ABIDE_data/data_HO/ABIDE_pcp/cpac/filt_noglobal'
    nroi = 111
    indim = 111

#Define dataset and center list for stratified sampling:

dataset = ABIDEDataset(path, name)
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0

id_list = os.listdir(path)
id_list.sort()
id_list = id_list[0:871]

center_list = []

for id in id_list: 
    folder_path = os.path.join(path, id)
    center = glob.glob(os.path.join(folder_path,'*.1D'))[0].split('/')[-1].split('_')[0]
    center_list.append(center)


#### MODE TRAIN ####
#Training BrainGNN with given parcellation:

  
if opt.mode == 'train': 

    #path to save the newly trained models
    save_path = '/media/SSD2/MindGraphs/ABIDE_data/BrainGNN_models_NEW/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for rep in range(opt.repetitions):
        #create path to save new models for current REPETION and Parcellation
        rep_save_path = os.path.join(save_path, opt.parcellation+"_REP_"+str(rep))

        if not os.path.exists(rep_save_path):
            os.makedirs(rep_save_path)

        for fold in range(K_folds): 
            #create path to save new models for current fold

            tr_index, te_index, val_index = train_val_test_split(fold = fold, center_list=center_list) 
            
            train_dataset = dataset[list(tr_index)] 
            val_dataset = dataset[list(val_index)]
            test_dataset = dataset[list(te_index)]

            train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
            val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
            
            ############### Define Graph Deep Learning Network ##########################
            seed_everything(123)
            model = Network(indim,opt.ratio,nclass,nroi).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
            
            #######################################################################################
            ############################   Model Training #########################################
            #######################################################################################
                    
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            best_auc = 0

            for epoch in range(0, opt.n_epochs):
                tr_loss, _, _, w1, w2 = train(model, train_loader, optimizer, scheduler, train_dataset)
                tr_acc, _, _, _, _, _, _, tr_auc_roc, tr_auc_pr = test_acc(train_loader, model)
                val_acc, _, _, _, _, _, _, val_auc_roc, val_auc_pr = test_acc(val_loader, model)
                
                print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Train AUC: {:.7f}'.format(epoch, tr_loss, tr_auc_pr))
                
                if val_auc_roc > best_auc:
                    print("saving best model")
                    best_loss = tr_loss
                    best_auc = val_auc_roc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(rep_save_path,'FOLD'+str(fold)+'.pth'))
            
            #######################################################################################
            ######################### Testing on testing set for current fold #####################
            #######################################################################################

            model.load_state_dict(best_model_wts)
            model.eval()
            test_accuracy, test_sensitivity, test_specificity, test_precision, test_recall, _, _, roc_auc, PR_auc = test_acc(test_loader, model)
            test_l= test_loss(test_loader, model)
            print("CURRENT FOLD TEST METRICS: ")
            print("Test Acc: {:.7f} Test AUC-ROC: {:.7f}, Test AUC:  {:.7f}".format(test_accuracy, roc_auc, PR_auc))
            
    
    "TESTING LATEST RUN OF 5-FOLD CROSS VALIDATION WITH {:.7f} repetions.".format(opt.repetitions)
    acc = []
    auc_roc = []
    auc = []

    trained_path = '/media/SSD2/MindGraphs/ABIDE_data/BrainGNN_models_NEW'
    for rep in range(5):
        rep_path = os.path.join(trained_path, opt.parcellation+"_REP_"+str(rep))
        for fold in range(K_folds): 
            model_path = os.path.join(rep_path,'FOLD'+str(fold)+'.pth')
            _, te_index, _ = train_val_test_split(fold = fold, center_list=center_list) 
            test_dataset = dataset[list(te_index)]
            test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
            model = Network(indim,opt.ratio,nclass,nroi).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            test_accuracy, _, _, _, _, _, _, test_roc_auc, test_PR_auc = test_acc(test_loader, model)
            acc.append(test_accuracy)
            auc_roc.append(test_roc_auc)
            auc.append(test_PR_auc)
    
    print("EVALUATION METRICS for "+opt.parcellation+" parcellation on your BrainGNN (5-fold cross-validation with 5 repetitions: )")
    print('Average accuracy', np.mean(acc), ' STD: ', np.std(acc))
    print('Average AUC ROC', np.mean(auc_roc), ' STD: ', np.std(auc_roc))
    print('Average AUC', np.mean(auc), ' STD: ', np.std(auc))

#### MODE TEST ####

elif opt.mode == 'test':
    acc = []
    auc_roc = []
    auc = []

    pretrained_path = '/media/SSD2/MindGraphs/ABIDE_data/BrainGNN_models'
    for rep in tqdm(range(5)):
        rep_path = os.path.join(pretrained_path, opt.parcellation+"_REP_"+str(rep))
        for fold in range(K_folds): 
            model_path = os.path.join(rep_path,'FOLD'+str(fold)+'.pth')
            _, te_index, _ = train_val_test_split(fold = fold, center_list=center_list) 
            test_dataset = dataset[list(te_index)]
            test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
            model = Network(indim,opt.ratio,nclass,nroi).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            test_accuracy, _, _, _, _, _, _, test_roc_auc, test_PR_auc = test_acc(test_loader, model)
            acc.append(test_accuracy)
            auc_roc.append(test_roc_auc)
            auc.append(test_PR_auc)
    
    print("EVALUATION METRICS for "+opt.parcellation+" parcellation on BrainGNN (5-fold cross-validation with 5 repetitions: )")
    print('Average accuracy', np.mean(acc), ' STD: ', np.std(acc))
    print('Average AUC ROC', np.mean(auc_roc), ' STD: ', np.std(auc_roc))
    print('Average AUC', np.mean(auc), ' STD: ', np.std(auc))

elif opt.mode == 'demo':
    best_model_path = '/media/SSD2/MindGraphs/ABIDE_data/BrainGNN_models/OURS_REP_0/FOLD0.pth'
    subject = [random.randint(0, 871)]
    subject_dataset = dataset[list(subject)]
    subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=True) #create loader with batchsize = 1 and shuffle True to select a single  random subject
    model = Network(indim,opt.ratio,nclass,nroi).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    for i, data in enumerate(subject_loader):
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = int(outputs[0].max(dim=1)[1])
        GT = int(data.y)
        break

    prediction = 'autisic' if pred==1 else 'healthy'
    ground_truth = 'autisic' if GT==1 else 'healthy'
    
    print("Subject {}".format(subject[0]))
    print('Prediction: ' + prediction)
    print('Ground Truth: ' + ground_truth)


    


