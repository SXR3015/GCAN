import csv
import numpy as np
import os
from math import ceil
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts
import pandas as pd
import fnmatch
opt = parse_opts()
root = opt.data_root_path

#health_dir = opt.data_type + '_healthy'
#MCI_dir = opt.data_type + '_MCI'
csv_save_dir = OsJoin('csv/', opt.data_type, opt.category)
test_ratio = 0.1
n_fold = 5

Subs=list()
least_subs_name=r'D:\sxr_bak\features_adni_backup\sfc'

for filename in os.listdir(least_subs_name):
    if 'de' in filename:
        sub = filename.split('zfc_fCovswra_de_')[1].split('_rsfMRI_timeseries_Dosenbach164.mat')[0]
        Subs.append(sub)
    else:
      sub = filename.split('zfc_Covswra_')[1].split('_rsfMRI_timeseries_Dosenbach164.mat')[0]
      # sub_name_delPRE = sub.split('_PRE')[0]
      Subs.append(sub)
sub_info_excel = pd.read_csv(r'E:\sxr\APP\edgedata\184\ADNI2_ADNI3.csv', header=0, usecols=[0, 5])
data_health = []
label_health = []
text_health = []
data_MCI = []
label_MCI = []
text_MCI = []
data_SCD = []
label_SCD = []
text_SCD = []
fea_num = 2

# sub_num=len(Subs)
# data_health=np.empty([sub_num, fea_i])
# data_MCI=np.empty([sub_num, fea_i])
# data_SCD=np.empty([sub_num, fea_i])
# sub_n=0
HC_num = 0
MCI_num = 0
SCD_num = 0
for sub in Subs:
    for feature in os.listdir(root):
        if 'csv' in feature:
            csv_contain=True
            continue
        for sub_fea in os.listdir(os.path.join(root, feature)):
#or str(sub_name_delPRE)+'_PRE' in str(sub_fea)
            if str(sub) in str(sub_fea):
                if 'HC' in sub_fea and 'HC' in opt.category:
                    data_health.append(OsJoin(root, feature, sub_fea))
                    if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                        label_health.append([1,0,0])
                        text_health.append(['HC to SCD in three class situation'])
                    elif opt.n_classes == 2 and opt.category == 'HC_MCI':
                        text_health.append(['HC to MCI in two class situation'])
                        label_health.append([1,0])
                    elif opt.n_classes == 2 and opt.category == 'HC_SCD':
                        text_health.append(['HC to SCD in two class situation'])
                        label_health.append([1, 0])
                    HC_num = HC_num+1
                elif 'MCI' in sub_fea and 'MCI' in opt.category:
                    data_MCI.append(OsJoin(root, feature, sub_fea))
                    if opt.n_classes == 2 and 'HC_MCI' == opt.category:
                        label_MCI.append([0, 1])
                        text_MCI.append(['MCI to HC in two class situation'])
                    elif opt.n_classes == 2 and 'MCI_SCD' == opt.category:
                        label_MCI.append([0, 1])
                        text_MCI.append(['MCI to SCD in two class situation'])
                    elif  opt.n_classes == 3:
                        label_MCI.append([0,1,0])
                        text_MCI.append(['MCI to HC in three class situation'])
                    MCI_num = MCI_num+1
                elif 'SCD' in sub_fea and 'SCD' in opt.category:
                    data_SCD.append(OsJoin(root, feature, sub_fea))
                    SCD_num = SCD_num+1
                    if opt.n_classes == 3:
                      label_SCD.append([0,0,1])
                      text_SCD.append(['SCD to MCI in three class situation'])
                    elif opt.n_classes == 2 and 'MCI_SCD' == opt.category:
                      label_SCD.append([1,0])
                      text_SCD.append(['SCD to MCI in two class situation'])
                    elif opt.n_classes == 2 and 'HC_SCD' == opt.category:
                      label_SCD.append([0,1])
                      text_SCD.append(['SCD to HC in two class situation'])
                else: #data from adni

                    if ('SCD' in sub_fea or 'MCI' in sub_fea or 'HC' in sub_fea):
                            #and ('SCD' not in opt.category or 'MCI' not in opt.category or
                                                                                     # 'HC' not in opt.category):
                       continue#exclude SCD or MCI in clinical dataset whsen classify two class
                    if str(sub) in str(sub_fea):
                        sub_index = list(sub_info_excel['Subject ID']).index(sub)
                        sub_category = sub_info_excel['DX Group'][sub_index]
                        if 'Normal' in sub_category and ('CN' in opt.category or 'HC' in opt.category ):
                                # ('CN' in opt.category or 'HC' in opt.category )
                            data_health.append(OsJoin(root, feature, sub_fea))
                            if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                                label_health.append([1,0,0])
                                text_health.append(['HC to SCD in three class situation'])
                            elif opt.n_classes == 2 and opt.category == 'HC_MCI':
                                label_health.append([1,0])
                                text_health.append(['HC to MCI in two class situation'])
                            HC_num = HC_num + 1
                        elif 'MCI' in sub_category and 'MCI' in opt.category:
                            data_MCI.append(OsJoin(root, feature, sub_fea))
                            if opt.n_classes == 2 and opt.category == 'HC_MCI':
                                label_MCI.append([0,1])
                                text_MCI.append(['MCI to HC in two class situation'])
                            elif opt.n_classes == 2 and 'MCI_SCD' in opt.category:
                                label_MCI.append([0, 1])
                                text_MCI.append(['MCI to SCD in two class situation'])
                            elif opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                                label_MCI.append([0,1,0])
                                text_MCI.append(['MCI to HC in three class situation'])
                            MCI_num = MCI_num + 1
                        elif 'SMC' in sub_category and ('SMC' in opt.category or 'SCD' in opt.category):
                            data_SCD.append(OsJoin(root, feature, sub_fea))
                            SCD_num = SCD_num + 1
                            if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                                label_SCD.append([0,0,1])
                                text_SCD.append(['SCD to MCI in three class situation'])
                            elif opt.n_classes == 2 and 'MCI_SCD' in opt.category:
                                label_SCD.append([1,0])
                                text_SCD.append(['SCD to MCI in two class situation'])
                            elif opt.n_classes == 2 and 'HC_SCD' in opt.category:
                                label_SCD.append([0,1])
                                text_SCD.append(['SCD to HC in two class situation'])
                        else:
                            continue

            else:
                continue
# if csv_contain == True:
#     fea_num = fea_num-1
np.random.seed(opt.manual_seed)
if len(data_health) > 0:
    data_health = np.array(data_health).reshape(int(HC_num/fea_num), fea_num)
    label_health = label_health[0:HC_num * fea_num:fea_num]
    text_health = text_health[0:HC_num * fea_num:fea_num]
    # health_list = np.concatenate((data_health, np.array(label_health).reshape(int(HC_num / fea_num), 1)), axis=1)
    health_list = np.concatenate((data_health, np.array(label_health),np.array(text_health)), axis=1)
    np.random.shuffle(health_list)
    n_test_health = ceil(health_list.shape[0] * test_ratio)
    n_train_val_health = health_list.shape[0] - n_test_health
    train_val_list_health = health_list[0:n_train_val_health, :]
    test_list_health = health_list[n_train_val_health:health_list.shape[0], :]
if len(data_MCI) > 0:
    data_MCI=np.array(data_MCI).reshape(int(MCI_num/fea_num), fea_num)
    label_MCI = label_MCI[0:MCI_num * fea_num:fea_num]
    text_MCI = text_MCI[0:MCI_num * fea_num:fea_num]
    MCI_list = np.concatenate((data_MCI, np.array(label_MCI), np.array(text_MCI)), axis=1)
    np.random.shuffle(MCI_list)
    n_test_MCI = ceil(MCI_list.shape[0] * test_ratio)
    n_train_val_MCI = MCI_list.shape[0] - n_test_MCI
    train_val_list_MCI = MCI_list[0:n_train_val_MCI, :]
    test_list_MCI = MCI_list[n_train_val_MCI:MCI_list.shape[0], :]
if len(data_SCD) > 0:
    data_SCD = np.array(data_SCD).reshape(int(SCD_num/fea_num), fea_num)
    label_SCD = label_SCD[0:SCD_num * fea_num:fea_num]
    text_SCD = text_SCD[0:SCD_num * fea_num:fea_num]
    SCD_list = np.concatenate((data_SCD, np.array(label_SCD), np.array(text_SCD)), axis=1)
    np.random.shuffle(SCD_list)
    n_test_SCD = ceil(SCD_list.shape[0] * test_ratio)
    n_train_val_SCD = SCD_list.shape[0] - n_test_SCD
    train_val_list_SCD = SCD_list[0:n_train_val_SCD, :]
    test_list_SCD = SCD_list[n_train_val_SCD:SCD_list.shape[0], :]
# for feature in os.listdir(root):
#     for filename in os.listdir(os.path.join(root, feature)):
#        # for file in filename:
#             if 'HC' in filename:
#                 #data_health.append(OsJoin(root, filename))
#
#                 label_health.append(0) # 0 for health label
#         #for filename in os.listdir(OsJoin(root, MCI_dir)):
#             elif 'MCI' in filename:
#                 data_MCI.append(OsJoin(root, filename))
#                 label_MCI.append(1) # 1 for MCI label
#             elif 'SCD' in filename:
#                 data_SCD.append(OsJoin(root, filename))
#                 label_SCD.append(-1)  # 1 for MCI label
# health_list = np.array([data_health, label_health]).transpose()
# MCI_list = np.array([data_MCI, label_MCI]).transpose()  # 都是按照名称顺序排列读入
# SCD_list = np.array([data_SCD, label_SCD]).transpose()
  #固定seed后每种数据都按照同一shuffle顺序排列
  # 打乱行顺序


#health_list_rand = np.random.permutation(health_list)
# rand_rows = np.arange(health_list.shape[0])# health_list = health_list[rand_rows]


# # down sampling
# MCI_list = MCI_list[0:health_list.shape[0]]

  # number of test samples

# number of test samples
 # number of trainning samples

# number of trainning samples

kf = KFold(n_splits=opt.n_fold, shuffle=False)
n = 0
names = locals()
if len(data_health) > 0:
    for train_index, val_index in kf.split(train_val_list_health):
        n += 1
        names['train_fold%s_health'%n] = train_val_list_health[train_index]
        names['val_fold%s_health' % n] = train_val_list_health[val_index]
n = 0
if len(data_MCI) > 0:
    for train_index, val_index in kf.split(train_val_list_MCI):
        n += 1
        names['train_fold%s_MCI'%n] = train_val_list_MCI[train_index]
        names['val_fold%s_MCI' % n] = train_val_list_MCI[val_index]
n = 0
if len(data_SCD) > 0:
    for train_index, val_index in kf.split(train_val_list_SCD):
        n += 1
        names['train_fold%s_SCD'%n] = train_val_list_SCD[train_index]
        names['val_fold%s_SCD'%n] = train_val_list_SCD[val_index]
names2 = locals()
for i in range(1, n_fold+1):
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i),  names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI, test_list_SCD))
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SCD) == 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI))
    if len(data_health) > 0 and len(data_MCI) == 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_health, test_list_SCD))
    if len(data_health) == 0 and len(data_MCI) > 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_MCI'%i), names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_MCI, test_list_SCD))
    np.random.seed(opt.manual_seed)
    np.random.shuffle(names2['train_list_fold%s'%i])
    np.random.shuffle(names2['val_list_fold%s'%i])

   # 按行堆叠
np.random.seed(opt.manual_seed)
np.random.shuffle(test_list)

csv_save_path = OsJoin(root, csv_save_dir)
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

for i in range(1, n_fold+1):
    with open(OsJoin(csv_save_path, 'train_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('train_list_fold%s'%i))
    with open(OsJoin(csv_save_path, 'val_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('val_list_fold%s'%i))


with open(OsJoin(csv_save_path, 'test.csv'), 'w', newline='') as f:  # 设置文件对象
    f_csv = csv.writer(f)
    f_csv.writerows(test_list)