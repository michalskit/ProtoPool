
import torch
import torch.nn as nn

import sys
sys.path.append('../')

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


class ModelWithoutPrototypicalLayerAKAresnet(nn.Module):
    def __init__(self, num_classes: int = 100, arch: str = 'resnet50', pretrained: bool = True,
                 add_on_layers_type: str = 'log', inat: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.arch = arch
        self.pretrained = pretrained
        self.inat = inat

        if self.inat:
            self.features = base_architecture_to_features['resnet50'](pretrained=pretrained, inat=True)
        else:
            self.features = base_architecture_to_features[self.arch](pretrained=pretrained)

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            raise NotImplementedError
        else:
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, 
                          out_channels= 512 * self.features.block.expansion,
                          kernel_size=1),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            ]

            self.add_on_layers = nn.Sequential(*add_on_layers)
        
        # initial weights
        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.features.block.expansion, num_classes))   
        

    def forward(self, x):
        x = self.features(x)
        x = self.add_on_layers(x)
        x = self.last_layer(x)
        return x
    