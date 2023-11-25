from omegaconf import OmegaConf
from omegaconf import DictConfig, open_dict
from BrainNetworkTransformer.model.BNT.bnt import BrainNetworkTransformer
from BrainNetworkTransformer.train import Train
from BrainNetworkTransformer.dataset.abide import load_abide_data
from BrainNetworkTransformer.dataset.dataloader import init_stratified_dataloader
from BrainNetworkTransformer.metrics import get_metrics
import argparse
import os
import torch.utils.data as utils
import random
import torch
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--parcellation', type=str, default='OURS', help='CC200, HO or OURS')
parser.add_argument('--mode', type=str, default='test',  help='test, train or demo')
parser.add_argument('--repetitions', type=int, default=5, help='How many repetions of 5-fold cross validation training to run')
parser.add_argument('--epochs', type=int, default=200, help='Epochs to run')

opt = parser.parse_args()

config = OmegaConf.load('BrainNetworkTransformer/config.yaml')
config.repeat_time=opt.repetitions
config.log_path=opt.parcellation
config.training.epochs=opt.epochs

if opt.parcellation == 'CC200':
    config.dataset.path="/media/SSD2/MindGraphs/ABIDE_data/abide_D16_S8.npy"

if opt.parcellation == 'HO':
    config.dataset.path="/media/SSD2/MindGraphs/ABIDE_data/abide_HO.npy"
    
if opt.parcellation == 'OURS':
    config.dataset.path="/media/SSD2/MindGraphs/ABIDE_data/abide_CC200.npy"
    

def model_training(cfg: DictConfig, fold: int, repetition: int):
    with open_dict(cfg):
        cfg.unique_id = str(fold)+"_"+str(repetition)

    datasets = load_abide_data(cfg)
    dataloader = init_stratified_dataloader(cfg, *datasets, fold) 
    model = BrainNetworkTransformer(cfg).cuda()
    training = Train(cfg, model, dataloader)

    training.train()

def main(cfg: DictConfig):
    for f in range(cfg.folds):
        for i in range(cfg.repeat_time):
            model_training(cfg, fold=f, repetition=i)
    mode_path = "/media/SSD2/MindGraphs/ABIDE_data/BNT_models_NEW/"
    get_metrics(mode_path, [cfg.log_path])

if __name__ == '__main__':
    if opt.mode == "train":
        main(config)
        
    if opt.mode == "test":
        paths = os.path.join("/media/SSD2/MindGraphs/ABIDE_data/BNT_models", opt.parcellation)
        list_dir = os.listdir(paths)
        mode_path = os.path.join("/media/SSD2/MindGraphs/ABIDE_data/BNT_models", opt.parcellation)
        get_metrics(mode_path, list_dir)
        
    if opt.mode == "demo":
        best_model_path = '/media/SSD2/MindGraphs/ABIDE_data/BNT_models/CC200/result_CC200_FOLD0/11-23-20-06-34/model.pt'
        subject = [random.randint(0, 846)]

        demo_timeseires, demo_pearson, labels, site = load_abide_data(config)
        test_dataset = utils.TensorDataset(demo_timeseires[subject], demo_pearson[subject], labels[subject])
        test_dataloader = utils.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
        
        model = BrainNetworkTransformer(config).cuda()
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        for time_series, node_feature, label in test_dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = model(time_series, node_feature)
            result = F.softmax(output, dim=1)[:, 1].tolist()
            result = np.array(result)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            pred = int(result[0])
            
            label = label.float()
            GT = int(label.cpu()[0])

        prediction = 'autisic' if pred==1 else 'healthy'
        ground_truth = 'autisic' if GT==1 else 'healthy'
        print("Subject {}".format(subject[0]))
        print('Prediction: ' + prediction)
        print('Ground Truth: ' + ground_truth)
        