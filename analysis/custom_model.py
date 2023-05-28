
import torch.nn as nn

import sys
sys.path.append('../')

from resnet_features import resnet50_features


base_architecture_to_features = {'resnet50': resnet50_features}

class Resnet50iNaturalist(nn.Module):
    '''
    The purpose of this class is to create resnet50 pretrained on iNaturalist.

    The code is based on ProtoPool code.

    Please note that ProtoPool code does not use the last layer of resnet50.
    The last layer is implemented in self.last_layer below.

    '''

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        self.features = base_architecture_to_features['resnet50'](pretrained=True, inat=True)

        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.features.block.expansion, num_classes))   
        

    def forward(self, x):
        x = self.features(x)
        x = self.last_layer(x)
        return x
    