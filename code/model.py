import torch
import torch.nn as nn
import torchvision.models as models

# see https://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def create_model(modelname):
    params_to_optimize = []
    
    if modelname == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif modelname == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        ValueError(f'{modelname} is not implemented.')
        
    num_ftrs = model.fc.in_features
    new_classifier = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(2 * num_ftrs),
        nn.Dropout(),
        nn.Linear(2 * num_ftrs,512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(),
        nn.Linear(512,10)
    )

    # remove last two layers (linear and adaptive avg pool)
    model_list = list(model.children())[:-2]
    model = nn.Sequential(*model_list)
    # replace adaptive avg pool with adapative max + avg pool and flatten
    # replace linear layer with 2 blocks of batch norm, dropout, linear
    model.classifier = new_classifier

    return model

