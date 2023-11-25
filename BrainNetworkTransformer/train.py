from .utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report, auc, precision_recall_curve
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
import logging
from tqdm import tqdm 
import os

class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 dataloaders: List[utils.DataLoader]) -> None:
        self.config = cfg
        self.model = model
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs 
        self.total_steps = 3700
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-6, weight_decay=1.0e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1.0e-6, eta_min=1.0e-7, last_epoch=-1, verbose=False)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(os.path.join(os.path.join("/media/SSD2/MindGraphs/ABIDE_data/BNT_models_NEW/", cfg.log_path), cfg.unique_id))
        #self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()
        
    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            #self.current_step += 1

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            predict = self.model(time_series, node_feature)
            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
        lr_scheduler.step()

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = self.model(time_series, node_feature)

            label = label.float()

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc_roc = roc_auc_score(labels, result)
        precision, recall, thresholds = precision_recall_curve(labels, result)
        pr_auc = auc(recall, precision)
        
        result, labels = np.array(result), np.array(labels)
    
        
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc_roc] + [pr_auc] + list(metric) + recall 

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        #self.current_step = 0
        for epoch in tqdm(range(self.epochs)):
            self.reset_meters()
            self.train_per_epoch(self.optimizer, self.lr_scheduler)
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)
            
            train_result = self.test_per_epoch(self.train_dataloader,
                                              self.train_loss, self.train_accuracy)
            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC_ROC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC_ROC": val_result[0],
                "Val Loss": self.val_loss.avg,
                "Val AUC_PR": val_result[1],
                "Test AUC_PR": test_result[1]
            })

        self.save_result(training_process)