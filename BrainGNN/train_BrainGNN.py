
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from BrainGNN.imports.utils import train_val_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import random


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


torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lamb0 = 1
lamb1 = 1
lamb2 = 0
lamb3 = 1
lamb4 = 1
lamb5 = 1
layer = 2
ratio = 0.5
nclass = 2


def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res

def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def train(model, train_loader, optimizer, scheduler, train_dataset):
    print('train...........')
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    loss_all = 0
    step = 0
    for data in train_loader:
        
        data = data.to(device)
        optimizer.zero_grad()
        #breakpoint()
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        #s1_list.append(s1.view(-1).detach().cpu().numpy())
        #s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)
        #loss_c = loss_fn(output, data.y)
        
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        loss_consist = 0
        for c in range(nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5* loss_consist
        
        step = step + 1
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        
        #loss_all += torch.round(loss*10000000)/10000000 * data.num_graphs
        optimizer.step()

        s1_arr = 0 #np.hstack(s1_list)
        s2_arr = 0 #np.hstack(s2_list)
        
    scheduler.step()
    return loss_all / len(train_dataset), s1_arr, s2_arr ,w1,w2

###################### Network Testing Function#####################################
def test_acc(loader, model):

   #model.eval()
    correct = 0
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    preds = []
    preds_auc = np.array([])
    trues = []
    trues_auc = np.array([])

    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = outputs[0].max(dim=1)[1]
        pred_auc = torch.exp(outputs[0])[:,1]
        preds_auc = np.append(preds_auc, pred_auc.cpu().detach().numpy())
        preds.append(pred.cpu().detach().numpy())
        trues.append(data.y.cpu().detach().numpy())
        trues_auc = np.append(trues_auc, data.y.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
        confusion_vector = pred/data.y
        TP += torch.sum(confusion_vector == 1).item()
        FN += torch.sum(confusion_vector == 0).item()
        TN += torch.sum(torch.isnan(confusion_vector)).item()
        FP += torch.sum(confusion_vector == float('inf')).item()
    
    ACC = correct / len(loader.dataset)
    SEN = TP / (TP + FN)
    SPE = TN / (TN + FP)
    roc_auc = metrics.roc_auc_score(trues_auc, preds_auc)
    precision, recall, thresholds = metrics.precision_recall_curve(trues_auc, preds_auc)
    pr_auc = metrics.auc(recall, precision)

    try:
        PREC = TP / (TP + FP)
        REC = TP / (TP + FN)
    except ZeroDivisionError:
        PREC = 0
        REC = 0
    return ACC, SEN, SPE, PREC, REC, preds, trues, roc_auc, pr_auc

def test_loss(loader,model):
    
    #loss_fn = torch.nn.CrossEntropyLoss()
    print('testing...........')
    #model.eval()
    loss_all = 0
   
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        
        loss_c = F.nll_loss(output, data.y)
        #loss_c = loss_fn(output, data.y)
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,ratio)
        loss_tpk2 = topk_loss(s2,ratio)
        loss_consist = 0
        for c in range(nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

