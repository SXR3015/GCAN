import os.path

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import csv
import os
import scipy.io as sio
from opts import parse_opts
from utils import OsJoin
from PIL import Image
import codecs
import pickle
opt = parse_opts()
data_type = opt.data_type
csv_dir = OsJoin(opt.data_root_path, 'csv', data_type, opt.category)
def jpeg_loader(path):
    img_pil = Image.open(path)
    # img_pil = img_pil.resize((160,160))
    img_arr = np.array(img_pil)
    # img_arr= np.transpose(img_arr, (2, 0, 1))
    if len(img_arr.shape)==3:
        img_arr = img_arr[:,:,0]
    else:
        img_arr = img_arr
    # img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    img_arr_cleaned = np.nan_to_num(img_arr)  # Replace NaN with zero and infinity with large finite numbers.

    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    img_pil = torch.from_numpy(img_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil)))
    # img_pil = img_pil / max_
    return img_pil
def nii_loader(path):
    img_pil = nib.load(path)
    img_arr = np.array(img_pil.get_fdata())
    if len(img_arr.shape) > 3:
        img_arr = np.sum(img_arr, axis=3)
    # img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    img_arr_cleaned = np.nan_to_num(img_arr)  # Replace NaN with zero and infinity with large finite numbers.

    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    img_pil = torch.from_numpy(img_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil )))
    # img_pil = img_pil / max_
    return img_pil
def dfc_loader(path):

    mat_arr = sio.loadmat(path)
    mat_arr = np.array((mat_arr['dfc_tmp']))
    mean_ = np.mean(mat_arr[:,:,:120], axis=2)
    mean_0_80 = np.mean(mat_arr[:,:,:80], axis=2)
    mean_80_ = np.mean(mat_arr[:, :, 80:], axis=2)
    # mean_80_120 = np.mean(mat_arr[:, :, 80:120], axis=2)
    mean_ =mean_[np.newaxis,:,:]
    # mean_0_40 = mean_0_40[np.newaxis, :, :]
    mean_0_80 = mean_0_80[np.newaxis, :, :]
    mean_80_  = mean_80_[np.newaxis, :, :]
    # mat_arr =np.transpose(mat_arr[:160,:160,:120],(2,0,1))
    mat_arr = np.concatenate((mean_0_80[:,:160,:160],mean_80_[:,:160,:160],mean_[:,:160,:160]),axis=0)
    # mat_arr = mean_[:,:160,:160]
    mat_arr_cleaned = np.nan_to_num(mat_arr)
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    mat_arr_cleaned [mat_arr_cleaned > 1.5] = 1.5
    noise = np.random.normal(0,1,(mat_arr.shape[1],mat_arr.shape[2]))
    noise = noise[np.newaxis,:,:]
    noise =np.tile(noise,(mat_arr_cleaned.shape[0],1,1))
    noise_image = noise + mat_arr_cleaned
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    # max_ = np.max(np.max(mat_arr))
    # max_ = torch.max(torch.max(torch.max(mat_arr)))
    # min_ = torch.min(torch.min(torch.min(mat_arr)))
    # mat_arr = (mat_arr-min_)/ max_
    # mat_arr = torch.where(torch.isinf(mat_arr), torch.full_like(mat_arr , 0), mat_arr)
    # mat_arr = torch.where(torch.isnan(mat_arr), torch.full_like(mat_arr, 0), mat_arr)
    # random_int = np.random.randint(1, 100)
    # if mat_arr.shape[2]>120:
    #     mat_arr = mat_arr[:,:,:120]
    # mat_arr_ = torch.sum(mat_arr, dim=2)/mat_arr.shape[2]
    # mat_arr_ = mat_arr[:,:,random_int]
    # random_int[mat_arr_ < 0]=0
    # mat_arr_  = torch.where(torch.isinf(mat_arr_ ), torch.full_like(mat_arr_ , 1), mat_arr_)
    return noise_image, mat_arr

def sfc_loader(path):
    mat_arr = sio.loadmat(path)
    try :
        mat_arr = np.array((mat_arr['z_matrix_sub']))[:160,:160]
    except KeyError:
        mat_arr = np.array((mat_arr['dfc_tmp']))[:160,:160]
    mat_arr  = mat_arr
    mat_arr_cleaned = np.nan_to_num(mat_arr)
    mat_arr_cleaned [mat_arr_cleaned > 1.5] = 1.5
    noise = np.random.normal(0,1,mat_arr.shape)
    noise_image = noise + mat_arr_cleaned
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    # mat_arr = mat_arr
    # min_ = torch.min(torch.min(mat_arr))
    # max_ = torch.max(torch.max(mat_arr))
    # mat_arr= (mat_arr-max_)/(max_-min_)

    # max_ = torch.max(torch.max(mat_arr))
    # mat_arr = mat_arr / max_
    # mat_arr = torch.unsqueeze(mat_arr, 2)
    return noise_image, mat_arr
def load_target_FC(labels):
    root = opt.data_root_path
    csv_save_dir = OsJoin('csv/', opt.data_type, opt.category)
    csv_save_path = OsJoin(root, csv_save_dir)
    # labels = labels.type(torch.FloatTensor)
    label = np.argmax(labels,axis=0)
    if 'HC' in  opt.category:
       HC_group = sio.loadmat(os.path.join(csv_save_path, 'HC_group.mat'))
       HC_group_arr = np.array((HC_group['HC_group']))[:160, :160]
    if 'MCI' in  opt.category:
       MCI_group = sio.loadmat(os.path.join(csv_save_path, 'MCI_group.mat'))
       MCI_group_arr = np.array((MCI_group['MCI_group']))[:160, :160]
    if 'SCD' in  opt.category:
       SCD_group = sio.loadmat(os.path.join(csv_save_path, 'SCD_group.mat'))
       SCD_group_arr = np.array((SCD_group['SCD_group']))[:160, :160]
    target_FC = []

    if opt.category == 'HC_MCI_SCD' and opt.n_classes ==3:
        if label == 1:
            target_FC = MCI_group_arr
            # target_FC.append(MCI_group_arr)
        elif label == 2:
            target_FC =  HC_group_arr
            # target_FC.append(HC_group_arr)
        elif label == 0:
            target_FC = SCD_group_arr
            # target_FC.append(SCD_group_arr)
    elif opt.category == 'HC_MCI' and opt.n_classes ==2:
        if label == 1:
            target_FC = HC_group_arr
            # target_FC.append(HC_group_arr)
        elif label == 0:
            target_FC = MCI_group_arr
            # target_FC.append(MCI_group_arr)
    elif opt.category == 'HC_SCD' and opt.n_classes ==2:
        if label == 1:
            target_FC = HC_group_arr
            # target_FC.append(HC_group_arr)
        elif label == 0:
            target_FC = SCD_group_arr
            # target_FC.append(SCD_group_arr)
    elif opt.category == 'MCI_SCD' and opt.n_classes ==2:
        if label == 1:
            target_FC = SCD_group_arr
            # target_FC.append(SCD_group_arr)
        elif label == 0:
            target_FC = MCI_group_arr
            # target_FC.append(MCI_group_arr)
    mat_arr_cleaned = np.nan_to_num(target_FC)
    mat_arr_cleaned [mat_arr_cleaned > 1.5] = 1.5
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    return mat_arr
def default_loader(path):
    data = np.loadtxt(path)
    # with open(path, 'rb') as f:
    #     data = f.read()
    # with open(path, 'rb') as f:
    #     datadict = pickle.load(f, encoding='latin1')
    #     x = datadict['data']
    # img_pil = nib.load(path)
    # img_arr = np.array(img_pil.get_fdata())
    # if len(img_arr.shape)>3:
    #     img_arr=np.sum(img_arr,axis=3)
    #img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    data_arr_cleaned = np.nan_to_num(data)
    # max_ = torch.max(torch.max(data_arr_cleaned))# Replace NaN with zero and infinity with large finite numbers.
    # data_arr_cleaned = data_arr_cleaned / max_
    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    # img_pil = torch.from_numpy(data_arr_cleaned)
    img_pil = torch.from_numpy(data_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil )))
    # img_pil = img_pil / max_
    img_pil = torch.unsqueeze(img_pil, 2)
    return img_pil


class TrainSet(Dataset):

    def __init__(self, fold_id, loader = default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            #for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_train = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_train_sfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_train_1 = [row[1] for row in reader]
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_train_2 = [row[2] for row in reader]
        # # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        # #     reader = csv.reader(csvfile)
        # #     file_train_fc = ([OsJoin(opt.data_root_path, row[3]) for row in reader])
        # # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        # #     reader = csv.reader(csvfile)
        # #     file_train_fb = ([OsJoin(opt.data_root_path, row[4]) for row in reader])
        # # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        # #     reader = csv.reader(csvfile)
        # #     file_train_reho = ([OsJoin(opt.data_root_path, row[5]) for row in reader])
        #
        # file_train = np.array([file_train_dfc, file_train_sfc])
        #     #file_train=[file_train file_train_tmp]
        #     #row=(row[0:5] for row in reader)
        #     #file_train_tmp=([OsJoin(opt.data_root_path, row[1]) for row in reader])
        label_train = np.array([label_train_1, label_train_2])
        # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     text = [row[4] for row in reader]
        if opt.n_classes ==3:
            with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_train_3 = [row[3] for row in reader]
            # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            #     reader = csv.reader(csvfile)
            #     text = [row[3] for row in reader]
            label_train = np.array([label_train_1, label_train_2, label_train_3])
        self.image = file_train
        self.label = label_train
        # self.text = text
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
        self.jpeg_loader = jpeg_loader
        self.load_target_FC =load_target_FC
    def __getitem__(self, index):
        fn = self.image[index]
        img_arr=[]
        # for fea in fn:
            # if "nii" in fea:
            #   img_arr_tmp = self.loader(fea)
            #   img_arr.append(img_arr_tmp)
        if "dfc" in fn:
          noise, real = self.dfc_loader(fn)
          img_arr.append(noise)
          img_arr.append(real)
        elif "sfc" in fn or "zfc" in fn:
          noise, real = self.sfc_loader(fn)
          img_arr.append(noise)
          img_arr.append(real)
        elif "Alff" in fn:
           img_arr_tmp = self.nii_loader(fn)
           img_arr.append(img_arr_tmp)
        elif 'JPEG'in fn:
            img_arr_tmp = self.jpeg_loader(fn)
            img_arr.append(img_arr_tmp)
        else:
            print('None data')
        #img = self.loader(fn)
        label = self.label.transpose()[index]
        label= np.array([int(label_str) for label_str in label])
        target_FC = load_target_FC(label)
        # text = self.text[index]
        return img_arr,label, target_FC

    def __len__(self):
        return len(self.image)

class ValidSet(Dataset):
    def __init__(self, fold_id, loader=default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            # for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_val = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        # with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_val_sfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_val_1 = [row[1] for row in reader]
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_val_2 = [row[2] for row in reader]
        # with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     text = [row[4] for row in reader]
        # file_val = np.array([file_val_dfc, file_val_sfc])
        label_val = np.array([label_val_1, label_val_2])
        # # file_val=[file_val file_val_tmp]
        # # row=(row[0:5] for row in reader)
        # # file_val_tmp=([OsJoin(opt.data_root_path, row[1]) for row in reader])
        if opt.n_classes==3:
            with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_val_3 = [row[3] for row in reader]
            # with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            #     reader = csv.reader(csvfile)
            #     text = [row[5] for row in reader]
            label_val = np.array([label_val_1, label_val_2, label_val_3])
        self.image = file_val
        self.label = label_val
        # self.text = text
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
        self.jpeg_loader = jpeg_loader
        self.load_target_FC =load_target_FC
    def __getitem__(self, index):
        fn = self.image[index]
        img_arr = []
        # for fea in fn:

            # for fea in fn:
            # if "nii" in fea:
            #   img_arr_tmp = self.loader(fea)
            #   img_arr.append(img_arr_tmp)
        if "dfc" in fn:
                noise, real = self.dfc_loader(fn)
                img_arr.append(noise)
                img_arr.append(real)
        elif "sfc" in fn or "zfc" in fn:
                noise, real = self.sfc_loader(fn)
                img_arr.append(noise)
                img_arr.append(real)
        elif "Alff" in fn:
                img_arr_tmp = self.nii_loader(fn)
                img_arr.append(img_arr_tmp)
        elif 'JPEG' in fn:
                img_arr_tmp = self.jpeg_loader(fn)
                img_arr.append(img_arr_tmp)
        else:
                print('None data')
        # img = self.loader(fn)
        label = self.label.transpose()[index]
        # text = self.text[index]
        label= np.array([int(label_str) for label_str in label])
        target_FC = load_target_FC(label)
        return img_arr,label,target_FC

    def __len__(self):
        return len(self.image)#data_set

class TestSet(Dataset):

    def __init__(self,loader=default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        file_test = []
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test_1 = [row[1] for row in reader]
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test_2 = [row[2] for row in reader]
        label_test = np.array([label_test_1, label_test_2])
        # with open(csv_dir + '/test.csv', 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_test_sfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        # with open(csv_dir + '/test.csv', 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     test_label_1 = [row[2] for row in reader]
        # with open(csv_dir + '/test.csv', 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     test_label_2 = [row[3] for row in reader]
        # with open(csv_dir + '/test_fold%s.csv', 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     text = [row[4] for row in reader]

        if opt.n_classes ==3:
            with open(csv_dir + '/test.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_test_3 = [row[3] for row in reader]
            # with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            #     reader = csv.reader(csvfile)
            #     text = [row[5] for row in reader]
            label_test = np.array([label_test_1, label_test_2, label_test_3])
        self.image = file_test
        self.label = label_test
        # self.text = text
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
        self.load_target_FC =load_target_FC
    def __getitem__(self, index):
        fn = self.image[index]
        img_arr = []
        if "dfc" in fn:
            noise, real = self.dfc_loader(fn)
            img_arr.append(noise)
            img_arr.append(real)
        elif "sfc" in fn or "zfc" in fn:
            noise, real = self.sfc_loader(fn)
            img_arr.append(noise)
            img_arr.append(real)
        elif "Alff" in fn:
            img_arr_tmp = self.nii_loader(fn)
            img_arr.append(img_arr_tmp)
        elif 'JPEG' in fn:
            img_arr_tmp = self.jpeg_loader(fn)
            img_arr.append(img_arr_tmp)
        else:
            print('None data')
        # img = self.loader(fn)
        label = self.label.transpose()[index]
        # text = self.text[index]
        label= np.array([int(label_str) for label_str in label])
        target_FC = self.load_target_FC(label)
        return img_arr, label,target_FC

    def __len__(self):
        return len(self.image)
