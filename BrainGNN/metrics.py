import torch
from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.data import DataLoader
from net.braingnn import Network
import os
import numpy as np
import argparse
from imports.utils import train_val_test_split
from sklearn import metrics
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score



parser = argparse.ArgumentParser()
parser.add_argument('--atlas', type=str, default='CC200', help='atlas')
parser.add_argument('--task', type=str, default='AUTISM', help='atlas')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/media/SSD2/MindGraphs/data_HO/ABIDE_pcp/cpac/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--K_fold', type=int, default=5, help='training which fold')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
opt = parser.parse_args()


def test_acc(loader, model):
    #model.eval()
    correct = 0
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    preds = []
    trues = []
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = outputs[0].max(dim=1)[1]
        preds.append(pred.cpu().detach().numpy())
        trues.append(data.y.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
        confusion_vector = pred/data.y
        TP += torch.sum(confusion_vector == 1).item()
        FN += torch.sum(confusion_vector == 0).item()
        TN += torch.sum(torch.isnan(confusion_vector)).item()
        FP += torch.sum(confusion_vector == float('inf')).item()
    
    ACC = correct / len(loader.dataset)
    SEN = TP / (TP + FN)
    SPE = TN / (TN + FP)
    try:
        PREC = TP / (TP + FP)
        REC = TP / (TP + FN)
    except ZeroDivisionError:
        PREC = 0
        REC = 0

    return ACC, SEN, SPE, PREC, REC, preds, trues

def AUC(loader, model):
    preds = np.array([])
    trues = np.array([])
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = torch.exp(outputs[0])[:,1]
        preds = np.append(preds,pred.cpu().detach().numpy())
        trues = np.append(trues, data.y.cpu().detach().numpy())
    #fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
    auc = metrics.roc_auc_score(trues, preds)
    return auc

def AUC_PREC_REC(loader, model):
    preds = np.array([])
    trues = np.array([])
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = torch.exp(outputs[0])[:,1]
        preds = np.append(preds,pred.cpu().detach().numpy())
        trues = np.append(trues, data.y.cpu().detach().numpy())
    
    precision, recall, thresholds = precision_recall_curve(trues, preds)
    auc = metrics.auc(recall, precision)
    return auc

def test_loss(loader,model):
    print('testing...........')
    #model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5 *loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


path = opt.dataroot
name = 'ABIDE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_fold = opt.K_fold

if opt.atlas == 'CC200':
    if opt.task == 'AUTISM':
        model_list = ['model_CC200_0', 'model_CC200_1', 'model_CC200_2', 'model_CC200_3', 'model_CC200_4']
    elif opt.task == "SEX":
        model_list = ['model_CC200_sex_0', 'model_CC200_sex_1', 'model_CC200_sex_2', 'model_CC200_sex_3', 'model_CC200_sex_4']
elif opt.atlas == 'HO':
    if opt.task == 'AUTISM':
        model_list = ['model_HO_0', 'model_HO_1', 'model_HO_2', 'model_HO_3', 'model_HO_4']
    elif opt.task == "SEX":
        model_list = ['model_HO_sex_0', 'model_HO_sex_1', 'model_HO_sex_2', 'model_HO_sex_3', 'model_HO_sex_4']

acc = []

spe = []

sen = []

pre = []

rec = []

auc_roc = []

auc_prec_rec = []
#model_list = ['model_ASD_5F_CC200_0', 'model_ASD_5F_CC200_1' , 'model_ASD_5F_CC200_2', 'model_ASD_5F_CC200_3', 'model_ASD_5F_CC200_4']
#model_list = ['model_ASD_5F_HO_0', 'model_ASD5F_HO_1' , 'model_ASD5F_HO_2', 'model_ASD5F_HO_3', 'model_ASD_5F_HO_4']

#model_list = ['model_ASD_HO_0', 'model_ASD_HO_1' , 'model_ASD_HO_2', 'model_ASD_HO_3', 'model_ASD_HO_4']
#model_list = ['model_CC200_FOLD4_0', 'model_CC200_FOLD4_1' , 'model_CC200_FOLD4_2', 'model_CC200_FOLD4_3', 'model_CC200_FOLD4_4']
model_list = ['model_ours_D20_S8_0', 'model_ours_D20_S8_1' , 'model_ours_D20_S8_2', 'model_ours_D20_S8_3', 'model_ours_D20_S8_4']

for model_name in tqdm(model_list): 
    folder_path = os.path.join('/home/amartinez/BrainGNN/', model_name)
    model_list = os.listdir(folder_path)
    model_list.sort()


    model_path = os.path.join(folder_path, model_list[-1])

    dataset = ABIDEDataset(path, name)
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0

    id_list = os.listdir(path)
    id_list.sort()
    id_list = id_list[0:871]

    center_list = []
    
    for id in id_list: 
        folder_path_id = os.path.join(path, id)
        center = glob.glob(os.path.join(folder_path_id,'*.1D'))[0].split('/')[-1].split('_')[0]
        center_list.append(center)

    tr_index, te_index, val_index= train_val_test_split(center_list=center_list)
    test_dataset = dataset[list(te_index)]
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

    
    model = Network(opt.indim,opt.ratio,opt.nclass,opt.nroi).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_accuracy, test_sensitivity, test_specificity, test_precision, test_recall, _, _ = test_acc(test_loader, model)
    test_auc = AUC(test_loader, model)
    test_auc_prec_rec = AUC_PREC_REC(test_loader, model)
    
    acc.append(test_accuracy)
    spe.append(test_specificity)
    sen.append(test_sensitivity)
    pre.append(test_precision)
    rec.append(test_recall)
    auc_roc.append(test_auc)
    auc_prec_rec.append(test_auc_prec_rec)

breakpoint()

print('Average accuracy', np.mean(acc), ' STD: ', np.std(acc))
print('Average specificity', np.mean(spe), ' STD: ', np.std(spe))
print('Average sensitivity', np.mean(sen), ' STD: ', np.std(sen))
print('Average precision', np.mean(pre), ' STD: ', np.std(pre))
print('Average recall', np.mean(rec), ' STD: ', np.std(rec))
print('Average AUC ROC', np.mean(auc_roc), ' STD: ', np.std(auc_roc))
print('Average AUC PR', np.mean(auc_prec_rec), ' STD: ', np.std(auc_prec_rec))
