#encoding:utf8

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

import config
import dataset
import head
import network
from update import GeM
from model import resnet50

import os
import random
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.FileHandler("train.txt")
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("using {} device.".format(device))
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)    
    
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])} 
    
    img_dir = os.path.join(args.data_root, args.train_data_path)
    train_dataset = dataset.Imagesdataset(img_dir, transform=data_transform['train'], imsize=512)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=1, 
                                         shuffle=False, 
                                         num_workers=0
                                        )    
    
    #weight_path = './resnet50-pre.pth'
    #state = torch.load(weight_path)
    model = network.Network(args)
    #model.load_state_dict(state['state_dict'])
    model.to(device)
    
    heads = head.ArcMarginProduct(in_features=args.backbone_dim, out_features=args.class_num)
    #heads = head.build_head(head_type=args.head, 
                            #embedding_size=512, 
                            #class_num=9, 
                            #m=args.m, 
                            #t_alpha=args.t_alpha, 
                            #h=args.h, 
                            #s=args.s)  
                            
    
    
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    
    epochs = args.epochs
    save_path = './mobilenetv2.pth' 
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            cos_thetas = heads(embeddings, labels)
            #cos_thetas = heads(embeddings, norms, labels)
            loss = loss_function(cos_thetas, labels)
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            logger.info("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss))
        
    torch.save(model, save_path)  
  
    logger.info('Finished Training')

if __name__ == '__main__':
    
    args = config.get_args()
    
    main(args)