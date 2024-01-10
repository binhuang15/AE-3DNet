from torch.utils.data import DataLoader
from kfold_train import Dataset_transform_224

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from model import *
import random
import copy
import argparse

test_transforms_224 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

#set random seeds
def seed_torch(seed=23):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

seed_torch(42)
# # evaluating indicator

def calcAUC(truth, probability):
    fpr, tpr, thresholds = roc_curve(truth, probability, pos_label=1)
    thresholds = list(thresholds)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    best_thresholds = thresholds[maxindex]
    tpr = list(tpr)
    fpr = list(fpr)

    avg_train_auc = auc(fpr, tpr)
    return avg_train_auc, best_thresholds,tpr,fpr

#set threshold
def calcACCSENSPE(truth, probability, threshold=0.5):
    probability_binary = np.asarray(copy.deepcopy(probability))
    probability_binary[probability_binary >= threshold] = 1
    probability_binary[probability_binary < threshold] = 0
    probability_binary = probability_binary.tolist()
    confMatrix = confusion_matrix(truth, probability_binary)
    tn = confMatrix[0, 0]
    fp = confMatrix[0, 1]
    tp = confMatrix[1, 1]
    fn = confMatrix[1, 0]
    SEN = tp / (tp + fn)
    SPE = tn / (tn + fp)
    ACC = (tp + tn) / (tp + fp + tn + fn)
    return ACC, SEN, SPE

#calculation of evaluation index values
def metric(y_true, y_pred_1):
    y_pred = np.argmax(y_pred_1, 1)

    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_1[:,1], pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    PPV = tp / (tp + fp)
    NPV = tn / (tn + fn)
    AUC = auc(fpr, tpr)

    print('sensitivity:{:.3f}\tspecificity:{:.3f}\tacc:{:.3f}\tauc:{:.3f}\tppv:{:.3f}\tnpv:{:.3f}'.format(sensitivity, specificity, acc, AUC, PPV, NPV))
    # print('sensitivity:{:.3f}\t1-specificity:{:.3f}\tacc:{:.3f}\tauc:{:.3f}'.format(sensitivity, specificity, acc, AUC))

    return acc, sensitivity, specificity, AUC, fpr, tpr

parse = argparse.ArgumentParser(description='test_params')
parse.add_argument('--data_path', default="/home/zhi/PycharmProjects/Project_hou/HCC_HAIC/data_npy/HCC_HAIC_all_data_20s_seed23/", type=str, help='the path of test data')
parse.add_argument('--model_save_name', '-ms', default="model_for_test", type=str, help='model_save_name')

#main fuction
if __name__ == '__main__':

    test_probability_all = []
    test_truth_all = []
    number_all = []####
    args = parse.parse_args()

    for i in range(5):
        test_data_path = os.path.join(args.data_path, str(i), "test.npy")
        test_list = np.load(test_data_path)

        test_dataset = Dataset_transform_224(test_list, transform=test_transforms_224)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model_ori_path = "./" + args.model_save_name
        model = AE_3DNet(num_classes=2)
        model = model.cuda()

        model_names = os.listdir(model_ori_path)
        model_names.sort()

        print(model_names[i])

        model_name_path = os.path.join(model_ori_path, model_names[i])

        model.load_state_dict(torch.load(model_name_path))

        test_probability = []
        test_truth = []
        number = []

        model.eval()
        for inputs, labels, ID in test_loader:
            input = inputs.cuda()
            label = labels.cuda()

            with torch.no_grad():
                logit = model(input)

                probability = torch.softmax(logit, dim=1)[:, 1]
                # ---

                test_probability.append(probability.data.cpu().numpy())
                test_truth.append(label.data.cpu().numpy())
                number.append(ID[0])  ###

        test_probability = np.concatenate(test_probability)
        test_truth = np.concatenate(test_truth)

        avg_auc, threshold, _, _ = calcAUC(test_truth, test_probability)
        acc, sen, spe = calcACCSENSPE(test_truth, test_probability, threshold)
        print('acc:{:3f},sen:{:3f},spe:{:3f},auc:{:3f}'.format(acc, sen, spe, avg_auc))

        test_probability_all.append(test_probability)
        test_truth_all.append((test_truth))

    test_probability_all = np.concatenate(test_probability_all)
    test_truth_all = np.concatenate(test_truth_all)

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')

    avg_auc, threshold,tpr_all,fpr_all = calcAUC(test_truth_all, test_probability_all)
    acc, sen, spe = calcACCSENSPE(test_truth_all, test_probability_all, threshold)
    print("OverAll - ", 'acc:{:3f},sen:{:3f},spe:{:3f},auc:{:3f}'.format(acc, sen, spe, avg_auc))
    plt.plot(fpr_all, tpr_all, color='coral', lw=2,label='AE-3DNet (AUC = {:.3f})'.format(avg_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity',fontsize=13)
    plt.ylabel('Sensitivity',fontsize=13)
    plt.title('Test result')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_ori_path, 'AUC.png'))
    plt.show()


