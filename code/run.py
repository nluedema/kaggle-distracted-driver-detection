import pathlib
import os
import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pandas as pd

from model import create_model
import tasks

base_dir = pathlib.Path(__file__).resolve().parents[1]
experiments_dir = os.path.join(base_dir,'experiments')
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

config_list = []
for modelname in ['resnet34','resnet50']:
    file_dir = os.path.join(experiments_dir,f'{timestamp}_{modelname}')

    config_list.append({
        'modelname':modelname,
        'filename':modelname,
        'file_dir':file_dir,
        'device':'cuda:1',
        'batch_size':64,
        'n_epochs': 10,
        'loss_function': nn.CrossEntropyLoss()
    })

# see https://pytorch.org/docs/stable/torchvision/models.html
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=(224,224),scale=(0.75,1)),
    transforms.ColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.3, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

val_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

path_train = os.path.join(base_dir,'data','own_split','train')
path_val = os.path.join(base_dir,'data','own_split','val')
path_train_full = os.path.join(base_dir,'data','imgs','train')
path_test = os.path.join(base_dir,'data','imgs','test')

dataset_train = tasks.create_dataset(path_train, train_transforms)
dataset_val = tasks.create_dataset(path_val, val_transforms)
dataset_train_full = tasks.create_dataset(path_train_full, train_transforms)
dataset_test = tasks.TestDataSet(path_test, val_transforms)

for config in config_list:
    print(f'Starting with {config["modelname"]}')
    model = create_model(config['modelname'])
    model = model.to(config['device'])

    ### PHASE 1 - tune on train/val split the last layers of the model
    config['datasets_dict'] = {
        'train' : dataset_train,
        'val' : dataset_val
    }
    
    # only tune the last conv block of the resnet and the new classifier
    params_to_optimize = []
    for m in list(model.children())[-2:]:
        params_to_optimize.extend(list(m.parameters()))
    
    config['optimizer'] = torch.optim.Adamax(params=params_to_optimize, lr=0.002)
    model = tasks.train(model,config)

    ### PHASE 2 - tune on whole train set the whole model
    config['datasets_dict'] = {
        'train' : dataset_train_full,
    }

    # change filename
    config['filename'] = config['filename'] + '_finetune' 
    
    # tune the whole model
    params_to_optimize = list(model.parameters())

    # reduce learning rate
    config['optimizer'] = torch.optim.Adamax(params=params_to_optimize, lr=0.0001)

    # reduce number of epochs
    config['n_epochs'] = 5

    model = tasks.train(model,config)

    ### PHASE 3 - create predictions
    model = nn.Sequential(*list(model.children()),nn.Softmax(dim=1))
    res, list_img_names = tasks.predict(model,dataset_test,config)

    df_res = pd.DataFrame(
        res, index = list_img_names,
        columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    )
    df_res.to_csv(
        os.path.join(config['file_dir'],f'submission_{config["filename"]}.csv'),
        index_label = 'img'
    )
