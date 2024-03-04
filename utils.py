import csv
import os
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
class AverageMeter(object):
    '''computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count =0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t') #\t为一个tab

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, labels):
    batch_size = labels.size(0)
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    _, labels_ = labels.topk(k=1, dim=1, largest=True)
    # pred = pred.t()
    correct = pred.squeeze().eq(labels_.squeeze().cuda())
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size

def calculate_recall(outputs, labels, opt):
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    _, labels_ = labels.topk(k=1, dim=1, largest=True)
    # pred = pred.t()  # 转置成行
    labels = labels_.squeeze().cuda()
    pred = pred.squeeze()
    if opt.n_classes == 3:
        TP = (((pred.data == 1) & (labels.data == 1)) | (
                    (pred.data == 2) & (labels.data == 2))).cpu().float().sum().data
        # TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FP = (((pred.data == 1) & (labels.data == 0)) | (
                    (pred.data == 2) & (labels.data == 0))|((pred.data == 2) & (labels.data == 1))).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1)) | (
                    (pred.data == 0) & (labels.data == 2))| ((pred.data == 1) & (labels.data == 2)) ).cpu().float().sum().data
        #
        # FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        # p = TP / (TP + FP)  #precision
        # if sum(labels[0].data) == 0: # all sampes are negative
        #     TP = 1
        #     FN = 0
        # label_0 = 0
        # for label in labels[0]: # no negative labels
        #     if label == 0:
        #         label_0 = label_0 + 1
        #     else:
        #         continue
        # if label_0 == 0:
        #     TN = 1
        #     FP = 0
        #
        # FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        # p = TP / (TP + FP)  #precision
        if TP + FN == 0:
            r = 0
        else:
            r = TP / (TP + FN)  # recall
            # r = torch.tensor(r).float()
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
            # p = torch.tensor(p).float()
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
            # f1 = torch.tensor(f1).float()
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
            # sen = torch.tensor(sen).float()
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
        # clf = LogisticRegression(solver="liblinear").fit(np.array(outputs.cpu().float().data), np.array(labels.cpu().float().data))
        # auc = roc_auc_score(np.array(labels.cpu().float().data, clf.predict_proba(np.array(outputs.cpu().float().data)), multi_class='ovr'))
        #        clf = MultiOutputClassifier(clf).fit(np.array(outputs.cpu().float().data), np.array(labels.cpu().float().data))
        #        y_pred = clf.predict_proba(np.array(labels.cpu().float().data))
        #        auc = roc_auc_score(labels.cpu().float().data, outputs.cpu().float().data)
        # F1 = 2 * r * p / (r + p)
        return r, p, f1, sen, sp
    # elif opt.n_classes == 2 and 'HC' in opt.category:
    #     TP = ((pred.data == 1) & (labels.data == 1)).cpu().float().sum().data
    #     #TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
    #     FN = (((pred.data == 0) & (labels.data == 1))).cpu().float().sum().data
    #     #
    #     #FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
    #     #p = TP / (TP + FP)  #precision
    #     r = TP / (TP + FN)  #recall
    # F1 = 2 * r * p / (r + p)
    # return r
    else:
        TP = ((pred.data == 1) & (labels.data == 1)).cpu().float().sum().data
        FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        # TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1))).cpu().float().sum().data
        # if sum(labels[0].data) == 0:
        #     TP = 1
        #     FN = 0
        # label_0 = 0
        # for label in labels[0]:
        #     if label == 0:
        #         label_0 = label_0 + 1
        #     else:
        #         continue
        # if label_0 == 0:
        #     TN = 1
        #     FP = 0
        #
        # FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        # p = TP / (TP + FP)  #precision
        if TP + FN == 0:
            r = 0
        else:
            r = TP / (TP + FN)  # recall
            # r = torch.tensor(r).float()
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
            # p = torch.tensor(p).float()
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
            # f1 = torch.tensor(f1).float()
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
            # sen = torch.tensor(sen).float()
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
            # sp = torch.tensor(sp).float()
        #        auc = roc_auc_score(np.array(labels.cpu().float().data), np.array(outputs.cpu().float().data))
        #        clf = LogisticRegression(solver="liblinear", random_state=0).fit(outputs.cpu().float().data, (labels.view(len(labels), -1)).cpu().float().data)
        #        auc=roc_auc_score(np.array(labels.cpu().float().data), clf.predict_proba(np.array(outputs.cpu().float().data))[:, 1])
        # F1 = 2 * r * p / (r + p)
        return r, p, f1, sen, sp
# def calculate_best_metric(epoch_metric):
#     best_metric = max(epoch_metric[:,1])
#     return
def OsJoin(*args):
    p = os.path.join(*args)
    p = p.replace('\\', '/')
    return p
