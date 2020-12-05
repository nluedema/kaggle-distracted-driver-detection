import pandas as pd
import numpy as np
import os
import shutil
import pathlib

path_data = pathlib.Path(__file__).resolve().parent
path_img_list = os.path.join(path_data,'driver_imgs_list.csv')
img_list = pd.read_csv(path_img_list)

# the train/val split is performed on drivers, so that only unseen drivers are in
# the validation split.
val_subjects = ['p066','p050','p049','p024','p016']
val_mask = np.isin(img_list['subject'],val_subjects)
img_list_val = img_list[val_mask]
img_list_train = img_list[~val_mask]

train_perc = len(img_list_train)/len(img_list)
val_perc = len(img_list_val)/len(img_list)

print('Creating train/val split')
print(f'train/val split: {round(train_perc,3)}/{round(val_perc,3)}')

path_dst = os.path.join(path_data, 'own_split')
pathlib.Path(path_dst).mkdir()
print(f'Create new directory {path_dst}')

# create directories
for dir_1 in ['train','val']:
    path_dir_1 = os.path.join(path_dst,dir_1)
    pathlib.Path(path_dir_1).mkdir()
    for dir_2 in img_list_val['classname'].unique():
        path_dir_2 = os.path.join(path_dir_1,dir_2)
        pathlib.Path(path_dir_2).mkdir()

# copy images to the new directories
path_src_dir_1 = os.path.join(path_data,'imgs','train')
for dir_1, img_list_df in zip(['train','val'],[img_list_train,img_list_val]):
    path_dst_dir_1 = os.path.join(path_dst,dir_1)

    counter = 1
    size = len(img_list_df)
    print(f'Copying images to {path_dst_dir_1} ...')

    for row in img_list_df.itertuples():
        classname = row.__getattribute__('classname')
        img = row.__getattribute__('img')
        
        path_src_dir_2 = os.path.join(path_src_dir_1,classname,img)
        path_dst_dir_2 = os.path.join(path_dst_dir_1,classname,img)
        
        shutil.copy2(src=path_src_dir_2,dst=path_dst_dir_2)

        print('', end = '\r')
        print(f'{counter}/{size}', end = '')
        counter += 1
    
    print('')
