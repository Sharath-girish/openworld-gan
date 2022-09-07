import torch
import torchvision
import torch.nn as nn
from torchvision import models

class MulticlassNet(nn.Module):
    def __init__(self, num_classes=10, add_fc=True):
        super(MulticlassNet, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.feat_dim = 2048
        self.fc_dim = 128 if add_fc else 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
            ) if add_fc else nn.Identity()
        self.classifier = nn.Linear(self.fc_dim, num_classes)
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, add_fc=True):
        super(MobileNetV2, self).__init__()
        self.features = models.mobilenet_v2(pretrained=True).features
        self.feat_dim = 1280
        self.fc_dim = 1280 if add_fc else 128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
            ) if add_fc else nn.Identity()
        self.classifier = nn.Linear(self.fc_dim, num_classes)
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

network_mapping = {'multiclassnet': MulticlassNet, 'mobilenetv2': MobileNetV2}

def get_network(name):
    return network_mapping[name]