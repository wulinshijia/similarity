#encoding:utf8
import torch
import torchvision
import torch.nn as nn

import head
from update import GeM
from model import resnet50

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        if args.backbone == 'resnet50':
            net = torchvision.models.resnet50(pretrained=True)
            features = list(net.children())[:-2]
        else:
            net = torchvision.models.mobilenet_v2(pretrained=True)
            features = list(net.children())[:-1]
        #net = resnet50()
        self.features = nn.Sequential(*features)
        self.pool = GeM()
        self.whiten = nn.Linear(args.backbone_dim, args.backbone_dim, bias=True)
        
        
    def forward(self, x):
        
        embeddings = self.features(x)
        embeddings = self.pool(embeddings).squeeze(-1).squeeze(-1)
        embeddings = self.whiten(embeddings)
        
        
        return embeddings
    
    