import csv
import numpy as np
import os
from math import ceil
import scipy.io as sio
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts
import pandas as pd
import glob
from PIL import Image
import fnmatch
opt = parse_opts()
root = opt.data_root_path
sample_list = []
HC_num = 0
MCI_num = 0
SCD_num = 0
data_health = []
label_health = []
data_MCI = []
label_MCI = []
data_SCD = []
label_SCD = []
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
# jpeg_images = jpeg_images = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True)
sub_info_excel = pd.read_csv(r'E:\sxr\APP\edgedata\184\ADNI2_ADNI3.csv', header=0, usecols=[0, 5])
# if opt.data_type == 'JPEG':
#     for sub_dir in os.listdir(root):
#         if 'FC' in sub_dir:
#             continue
#             for sub_dir in os.listdir(os.path.join(root, 'FC')):
#                 sample_list.append(os.path.join(root, 'FC', sub_dir))
#         else:
#             if 'csv' in sub_dir:
#                 continue
#             jpeg_sub_dir = os.listdir(os.path.join(root,sub_dir,'images'))
#             for jpegs in jpeg_sub_dir:
#                 img_pil = Image.open(os.path.join(root,sub_dir,'images',jpegs))
#                 img_arr = np.array(img_pil)
#                 if len(img_arr.shape) == 2:
#                     continue
#                 sample_list.append(os.path.join(root,sub_dir,'images',jpegs))
if opt.data_type == 'DFC':
    for sub_dir in os.listdir(os.path.join(root,'DFC')):
        # sample_list.append(os.path.join(root, 'FC_split_tmp', sub_dir))
        if 'HC' not in sub_dir and 'MCI' not in sub_dir and 'SCD' not in sub_dir:
            sub_name = sub_dir.split('_rsfMRI')[0]
            # if 'de' in sub_name:
            sub_name = sub_name.split('_Covswra_')[-1]
        # if 'SCD' in sub_dir:
        #     print('stop')
        if 'HC' in sub_dir and 'HC' in opt.category:
            data_health.append(os.path.join(root, 'DFC', sub_dir))
            if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                label_health.append([1,0,0])
            elif opt.n_classes == 2 and opt.category == 'HC_MCI':
                label_health.append([1,0])
            elif opt.n_classes == 2 and opt.category == 'HC_SCD':
                label_health.append([1,0])
            HC_num = HC_num+1
        elif 'MCI' in sub_dir and 'MCI' in opt.category:
            data_MCI.append(os.path.join(root, 'DFC', sub_dir))
            if opt.n_classes == 2 and 'HC_MCI' == opt.category:
                label_MCI.append([0, 1])
            elif opt.n_classes == 2 and 'MCI_SCD' == opt.category:
                label_MCI.append([0, 1])
            elif  opt.n_classes == 3:
                label_MCI.append([0,1,0])
            MCI_num = MCI_num+1
        elif 'SCD' in sub_dir and 'SCD' in opt.category:
            data_SCD.append(os.path.join(root, 'DFC', sub_dir))
            SCD_num = SCD_num+1
            if opt.n_classes == 3:
              label_SCD.append([0,0,1])
            elif opt.n_classes == 2 and 'MCI_SCD' == opt.category:
              label_SCD.append([1,0])
            elif opt.n_classes == 2 and 'HC_SCD' == opt.category:
              label_SCD.append([0,1])
        else: #data from adni

            if ('SCD' in sub_dir or 'MCI' in sub_dir or 'HC' in sub_dir) and ('SCD' not in opt.category or 'MCI' not in opt.category or
                                                                             'HC' not in opt.category):
               continue
                    #exclude SCD or MCI in clinical dataset whsen classify two class
            # if str(sub) in str(sub_dir):
            sub_index = list(sub_info_excel['Subject ID']).index(sub_name)
            sub_category = sub_info_excel['DX Group'][sub_index]
            if 'Normal' in sub_category and ('CN' in opt.category or 'HC' in opt.category ):
                    # ('CN' in opt.category or 'HC' in opt.category )
                data_health.append(os.path.join(root, 'DFC', sub_dir))
                if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                    label_health.append([1,0,0])
                elif opt.n_classes == 2 and opt.category == 'HC_MCI':
                    label_health.append([1,0])
                elif opt.n_classes == 2 and opt.category == 'HC_SCD':
                    label_health.append([1, 0])
                HC_num = HC_num + 1
            elif 'MCI' in sub_category and 'MCI' in opt.category:
                data_MCI.append(os.path.join(root, 'DFC', sub_dir))
                if opt.n_classes == 2 and opt.category == 'HC_MCI':
                    label_MCI.append([0,1])
                elif opt.n_classes == 2 and 'MCI_SCD' in opt.category:
                    label_MCI.append([0, 1])
                elif opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                    label_MCI.append([0,1,0])
                MCI_num = MCI_num + 1
            elif 'SMC' in sub_category and ('SMC' in opt.category or 'SCD' in opt.category):
                data_SCD.append(os.path.join(root, 'DFC', sub_dir))
                SCD_num = SCD_num + 1
                if opt.n_classes == 3 and opt.category == 'HC_MCI_SCD':
                    label_SCD.append([0,0,1])
                elif opt.n_classes == 2 and 'MCI_SCD' in opt.category:
                    label_SCD.append([1,0])
                elif opt.n_classes == 2 and 'HC_SCD' in opt.category:
                    label_SCD.append([0,1])
            else:
                continue
#health_dir = opt.data_type + '_healthy'
#MCI_dir = opt.data_type + '_MCI'
csv_save_dir = OsJoin('csv/', opt.data_type, opt.category)
def mean_group(group_list):
    try:
        mat_arr = sio.loadmat(group_list[0][0])
        mat_arr = np.array((mat_arr['z_matrix_sub']))[:160, :160]
    except KeyError:
        mat_arr = sio.loadmat(group_list[0][0])
        mat_arr = np.array((mat_arr['dfc_tmp']))[:160, :160]
    data_group = np.zeros((mat_arr.shape[0], mat_arr.shape[1]))
    for path in group_list:
        try :
            mat_arr = sio.loadmat(path[0])
            mat_arr = np.array((mat_arr['z_matrix_sub']))[:160,:160]
        except KeyError:
            mat_arr = sio.loadmat(path[0])
            mat_arr = np.array((mat_arr['dfc_tmp']))[:160,:160]
            mat_arr = np.mean(mat_arr,axis=2)
        data_group = data_group + mat_arr
    data_group = data_group / group_list.shape[0]
    return data_group
csv_save_path = OsJoin(root, csv_save_dir)

test_ratio = 0.1
n_fold = opt.n_fold

# sub_num=len(Subs)
# data_health=np.empty([sub_num, fea_i])
# data_MCI=np.empty([sub_num, fea_i])
# data_SMC=np.empty([sub_num, fea_i])
# sub_n=0

# sub_info_excel = pd.read_csv(r'E:\sxr\APP\edgedata\184\ADNI2_ADNI3.csv', header=0, usecols=[0, 5])
# if len(sample_list) > 0:
#     np.random.shuffle(sample_list)
#     n_test = ceil(len(sample_list) * test_ratio)
#     n_train_val = len(sample_list) - n_test
#     train_val_list = sample_list[0:n_train_val]
#     test_list = sample_list[n_train_val:]

if len(data_health) > 0:
    # data_health = np.array(data_health).reshape(int(HC_num/fea_num), fea_num)
    # label_health = label_health[0:HC_num * fea_num:fea_num]
    # text_health = text_health[0:HC_num * fea_num:fea_num]
    # health_list = np.concatenate((data_health, np.array(label_health).reshape(int(HC_num / fea_num), 1)), axis=1)
    health_list = np.concatenate((np.array(data_health)[...,np.newaxis], np.stack((label_health), axis=0)), axis=1)
    np.random.shuffle(health_list)
    n_test_health = ceil(health_list.shape[0] * test_ratio)
    n_train_val_health = health_list.shape[0] - n_test_health
    train_val_list_health = health_list[0:n_train_val_health, :]
    group_HC = mean_group(train_val_list_health)
    HC_dict = {"HC_group": group_HC, "label": "HC_group"}
    sio.savemat(os.path.join(csv_save_path, 'HC_group.mat'), HC_dict)
    test_list_health = health_list[n_train_val_health:health_list.shape[0], :]
if len(data_MCI) > 0:
    MCI_list = np.concatenate((np.array(data_MCI)[...,np.newaxis], np.array(label_MCI)), axis=1)
    np.random.shuffle(MCI_list)
    n_test_MCI = ceil(MCI_list.shape[0] * test_ratio)
    n_train_val_MCI = MCI_list.shape[0] - n_test_MCI
    train_val_list_MCI = MCI_list[0:n_train_val_MCI, :]
    group_MCI = mean_group(train_val_list_MCI)
    MCI_dict = {"MCI_group": group_MCI, "label": "MCI_group"}
    sio.savemat(os.path.join(csv_save_path, 'MCI_group.mat'), MCI_dict)
    test_list_MCI = MCI_list[n_train_val_MCI:MCI_list.shape[0], :]
if len(data_SCD) > 0:
    SCD_list = np.concatenate((np.array(data_SCD)[...,np.newaxis], np.array(label_SCD)), axis=1)
    np.random.shuffle(SCD_list)
    n_test_SCD = ceil(SCD_list.shape[0] * test_ratio)
    n_train_val_SCD = SCD_list.shape[0] - n_test_SCD
    train_val_list_SCD = SCD_list[0:n_train_val_SCD, :]
    group_SCD = mean_group(train_val_list_SCD)
    SCD_dict = {"SCD_group":  group_SCD, "label": "SCD_group"}
    sio.savemat(os.path.join(csv_save_path, 'SCD_group.mat'), SCD_dict)
    test_list_SCD = SCD_list[n_train_val_SCD:SCD_list.shape[0], :]

kf = KFold(n_splits=opt.n_fold, shuffle=False)
n = 0
names = locals()


# if len(sample_list) > 0:
#     for train_index, val_index in kf.split(train_val_list):
#         n += 1
#         names['train_fold%s'%n] = np.array(train_val_list)[train_index]
#         names['val_fold%s'%n] = np.array(train_val_list)[val_index]
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
# for i in range(1, n_fold+1):
#     names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s'%i)))
#     names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s'%i)))
#     test_list = np.vstack((test_list))
#
#     np.random.seed(opt.manual_seed)
#     np.random.shuffle(names2['train_list_fold%s'%i])
#     np.random.shuffle(names2['val_list_fold%s'%i])

   # 按行堆叠
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