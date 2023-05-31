import copy
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from custom_resnet_features import ResNet_features, Bottleneck

model_urls = {
    'resnet50Nat': '/home/robin/models/resnet50_iNaturalist.pth'
}

model_dir = '../pretrained_models'

import sys
sys.path.append('../')

def resnet50_features(pretrained=False, inat=False, dropout=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], dropout=dropout)
    if pretrained:
        if inat:
            model_dict = torch.load(model_urls['resnet50Nat'])
            new_model = copy.deepcopy(model_dict)
            for k in model_dict.keys():
                if k.startswith('module.backbone.cb_block'):
                    splitted = k.split('cb_block')
                    new_model['layer4.2' + splitted[-1]] = model_dict[k]
                    del new_model[k]
                elif k.startswith('module.backbone.rb_block'):
                    del new_model[k]
                elif k.startswith('module.backbone.'):
                    splitted = k.split('backbone.')
                    new_model[splitted[-1]] = model_dict[k]
                    del new_model[k]
                elif k.startswith('module.classifier'):
                    del new_model[k]
            model.load_state_dict(new_model, strict=True)
        else:
            my_dict = model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir)
            my_dict.pop('fc.weight')
            my_dict.pop('fc.bias')
            model.load_state_dict(my_dict, strict=False)
    return model


class Resnet50iNaturalist(nn.Module):
    '''
    The purpose of this class is to create resnet50 pretrained on iNaturalist.

    The code is based on ProtoPool code.

    Please note that ProtoPool code does not use the last layer of resnet50.
    The last layer is implemented in self.last_layer below.

    '''

    def __init__(self, num_classes: int = 100, dropout=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        self.features = resnet50_features(pretrained=True, inat=True, dropout=dropout)
        
        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.features.block.expansion, num_classes))   
        

    def forward(self, x):
        features = self.features(x)
        x = self.last_layer(features)
        return x, features
    