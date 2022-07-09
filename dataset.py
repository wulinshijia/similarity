#encoding:utf8

import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.datasets as datasets

class ImageProcess():
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def process(self):
        imgs = list()
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                img_path = os.path.join(root + os.sep, file)
                image = Image.open(img_path)
                imgs.append(img_path)

        return imgs
    
class Imagesdataset(datasets.ImageFolder):

    def __init__(self, img_dir, transform=None, imsize=None, loader=datasets.folder.default_loader):
        super(Imagesdataset, self).__init__(img_dir, transform=transform, loader=loader)

        self.imsize = imsize
        self.transform = transform

    def __getitem__(self, index):

        path, target = self.samples[index]
        img = pil_loader(path)

        img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img