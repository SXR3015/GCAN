import os
import scipy.io as sio
import numpy as np
path = 'D:\sxr_bak\images\\train\FC_split';
path_save = 'D:\sxr_bak\images\\train\FC_split_tmp'
for file in os.listdir(path):
    if 'zfc' in file:
        zfc_mat = sio.loadmat(os.path.join(path,file))
        file_name = file.split('.mat')[0]
        zfc_mat = np.array((zfc_mat['z_matrix_sub']))
        zfc_dict = {"z_matrix_sub":zfc_mat , "label":"z_matrix_sub"}
        sio.savemat(os.path.join(path_save, file_name + '.mat'),zfc_dict)
        print(file_name)
        # continue
    else:
        dfc_mat = sio.loadmat(os.path.join(path,file))
        file_name = file.split('.mat')[0]
        dfc_mat = np.array((dfc_mat['dfc_tmp']))
        for i in range(0, dfc_mat.shape[2]):
            dfc_mat_i = dfc_mat[:,:,i]
            dfc_dict = {"dfc_tmp":dfc_mat_i, "label":"dfc_tmp"}
            sio.savemat(os.path.join(path_save,file_name+'_'+str(i)+'.mat'), dfc_dict)
            print(file_name)