import os
import numpy as np

def get_metrics(mode_path:str, log_path: list):
    paths = log_path
    acc = []
    auc_roc = []
    auc_pr = []

    for p in paths:
        p = os.path.join(mode_path, p)
        list_dir = os.listdir(p)
        list_dic = []
        for i in list_dir:
            dataroot = os.path.join(p, i)
            data = np.load(os.path.join(dataroot, "training_process.npy"), allow_pickle=True)
            val_auc = 0
            #breakpoint()
            
            dic = {}
            
            for d in data:
                if d["Val AUC_ROC"] > val_auc and d["Epoch"] > 20:
                    dic = d
                    val_auc = d["Val AUC_ROC"]
            list_dic.append(dic)

        for i in list_dic:
            acc.append(i["Test Accuracy"]/100)
            auc_roc.append(i["Test AUC_ROC"])
            auc_pr.append(i["Test AUC_PR"])

    print('Average accuracy', np.mean(acc), ' STD: ', np.std(acc))
    print('Average AUC_ROC', np.mean(auc_roc), ' STD: ', np.std(auc_roc))
    print('Average AUC_PR', np.mean(auc_pr), ' STD: ', np.std(auc_pr))


    