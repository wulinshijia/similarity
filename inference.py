#encoding:utf8
import torch
import numpy as np
from PIL import Image,ImageChops,ImageDraw

import dataset
from model import resnet50
import random

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.FileHandler("inference.txt")
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

def to_input(img):
    np_img = np.array(img)
    img = ((np_img/ 255.) - 0.5) / 0.5
    tensor = torch.tensor([img.transpose(2, 0, 1)]).float()
    return tensor

def L2_norm(x):
    norm = torch.norm(x, 2, 1, True) 
    output = torch.div(x, norm)    
    return output

def query(query_path, imagetoid, idtoimage, similarity_scores, path_name):
    if query_path == 'my_images/database_test/data\\target.jpeg':
        number = 10
    else:
        number = 5

    idx = imagetoid[query_path]
    values, indices = torch.sort(similarity_scores[idx], descending=True)
    logger.info(path_name)
    logger.info('input img: {}'.format(query_path))
    for i in range(number):
        logger.info('query{} img: {} score:{}'.format(i, idtoimage[indices[i].item()], values[i]))         

def compare_images(path_one, path_two, diff_save_location):

    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
 
    diff = ImageChops.difference(image_one, image_two)

    if diff.getbbox() is None:
        print("We are the same!")
    else:
        diff.save(diff_save_location)
        diff.show()

def compete_pix(img0, img1, i, j):
    pix_img0 = img0.getpixel((i, j))
    pix_img1 = img1.getpixel((i, j))
    x=-1
    y=-1

    threshold = 20
    if abs(pix_img0[0] - pix_img1[0]) < threshold and abs(pix_img0[1] - pix_img1[1]) < threshold and abs(
        pix_img0[2] - pix_img1[2]) < threshold:
        x=i
        y=j        
        return x,y
    else:

        return x,y


if __name__ == '__main__':
    
    img_dir = 'my_images/database_test/data'
    images = dataset.ImageProcess(img_dir).process() 
    imagetoid = {j:i for i,j in enumerate(images)}
    idtoimage = {i:j for i,j in enumerate(images)}    
    
    query_path = random.choice(images)
    model_weight_path = ["./resnet50.pth", "./mobilenetv2.pth"]
    n = len(model_weight_path)
    for path_id in range(n):
        path_name = model_weight_path[path_id][2:-4]
        path = model_weight_path[path_id]
        net = torch.load(path)
        net.eval()
        
        features = []
        for j in images:
            img = Image.open(j)
            img = imresize(img, 512)
            in_input = to_input(img)
            
            feature = net(in_input)
            feature = L2_norm(feature)
            
            features.append(feature)
        #features = torch.cat(features)
        #norm_features = features/torch.sqrt(torch.sum(torch.square(features), dim=1, keepdim=True))
        #similarity_scores = norm_features @ norm_features.T
        similarity_scores = torch.cat(features) @ torch.cat(features).T 
        
        query(query_path, imagetoid, idtoimage, similarity_scores, path_name)
    
    path_one = 'my_images/database_test/data\\22.jpeg'
    path_two ='my_images/database_test/data\\target.jpeg'
    diff_save_location = '不相同.png'
    compare_images(path_one, path_two, diff_save_location)
    
    img0 = Image.open(path_one)
    img1 = Image.open(path_two) 
    img2 = img1.copy()
    draw=ImageDraw.Draw(img2)  
    for i in range(img0.size[0]):
        for idx,j in enumerate(range(img0.size[1])):    
            x, y = compete_pix(img0, img1, i, j)
            if x and y and idx%5 == 0:
                
                draw.point((x,y),fill=(255,0,0))
    img2.save('相同点.png')
    img2.show()
    print('相同点.png:红色连线的地方')
    
    
    