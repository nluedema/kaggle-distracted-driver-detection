import time
import copy
import pickle
import pathlib
import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

class TestDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_name = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, img_name)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image,img_name)

def create_dataset(path, transforms):
    dataset = datasets.ImageFolder(root=path,transform=transforms)
    
    return dataset

def train(model, config):
    since = time.time()
    
    path = config['file_dir']
    filename = config['filename']
    device = config['device']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    datasets_dict = config['datasets_dict']
    loss_function = config['loss_function']
    optimizer = config['optimizer']
    if 'optimizer' in config.keys():
        lr_scheduler = config['optimizer']
    else:
        lr_scheduler = None

    if not pathlib.Path(path).is_dir():
        pathlib.Path(path).mkdir()

    accuracy_dict = {
        'train': [],
        'val': []
    }
    loss_dict = {
        'train': [],
        'val': []
    }

    f = open(os.path.join(path,f'log_{filename}'), 'w')

    if 'train' in datasets_dict.keys():
        phases = ['train']
    else:
        ValueError('A train dataset has to be provided.')

    if 'val' in datasets_dict.keys():
        phases.append('val')
    else:
        print('No validation set provided, skipping validation.')
        f.write('No validation set provided, skipping validation.\n')

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs-1}')
        print('-' * 10)
        f.write(f'Epoch {epoch}/{n_epochs-1}\n')
        f.write('-' * 10)
        f.write('\n')

        for phase in phases:
            dataset = datasets_dict[phase]
            if phase == 'train':
                model.train()
                loader = DataLoader(
                    dataset,batch_size=batch_size,shuffle=True, num_workers=4
                )
            else:
                model.eval()
                loader = DataLoader(
                    dataset,batch_size=batch_size,shuffle=False, num_workers=4
                )

            running_loss = 0.0
            running_corrects = 0

            # iterate over data.
            for batch_idx, (inputs, target) in enumerate(loader):
                # remove previous line
                print('                                          ', end='\r')
                # print new line
                print(f'Batch {batch_idx} / {len(loader)-1}',end='\r')

                inputs = inputs.to(device)
                target = target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)
            
            if phase == 'train' and lr_scheduler != None:
                lr_scheduler.step()

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)

            loss_dict[phase].append(epoch_loss)
            accuracy_dict[phase].append(epoch_acc)

            print('{} loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            f.write('{} loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val': 
                if epoch == 0:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()
        f.write('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if 'val' in phases:
        print('Best val loss: {:4f}'.format(best_loss))
        f.write('Best val loss: {:4f}\n'.format(best_loss))
    else:
        print('No validaton set provided, saving last model as best model.')
        f.write('No validaton set provided, saving last model as best model.\n')
        best_model_wts = model.state_dict()
    
    print()

    f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    with open(os.path.join(path,f'model_{filename}.pkl'), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path,f'loss_{filename}.pkl'), 'wb') as output:
        pickle.dump(loss_dict, output, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path,f'acc_{filename}.pkl'), 'wb') as output:
        pickle.dump(accuracy_dict, output, pickle.HIGHEST_PROTOCOL)
    
    return model

def predict(model, dataset, config):
    since = time.time()
    
    device = config['device']
    batch_size = config['batch_size']

    res = torch.empty((len(dataset),10),dtype=torch.float,device=device)
    list_img_names = []
    
    model.eval()
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False, num_workers=4)

    start = 0
    
    # iterate over data.
    for batch_idx, (inputs, img_names) in enumerate(loader):
        print(f'Batch {batch_idx} / {len(loader)-1}',end='\r')

        inputs = inputs.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        
        res[start:(start+batch_size)] = outputs
        start = start + batch_size
        list_img_names.extend(list(img_names))

    res = res.cpu().detach().numpy()

    time_elapsed = time.time() - since
    print('Prediction complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return res,list_img_names
