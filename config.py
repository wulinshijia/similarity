#encoding:utf8
import argparse

def get_args():

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--data_root', type=str, default='my_images')
    parent_parser.add_argument(
        '--train_data_path', type=str, default='images') 
    parent_parser.add_argument(
        '--epochs', default=20, type=int, metavar='N', help='number of total epochs to run') 
    parent_parser.add_argument(
        '--seed', type=int, default=42, help='seed for initializing training.')
    parent_parser.add_argument(
        '--backbone', type=str, default='mobilenetv2')     
    parent_parser.add_argument(
        '--backbone_dim', type=int, default=1280)    
    parent_parser.add_argument(
        '--class_num', type=int, default=9)     
    parent_parser.add_argument('--head', default='adaface',
                        type=str, choices=('adaface'))
    parent_parser.add_argument('--m', default=0.1, type=float)
    parent_parser.add_argument('--h', default=0.333, type=float) 
    parent_parser.add_argument('--s', type=float, default=64.0) 
    parent_parser.add_argument('--t_alpha', default=0.01, type=float) 
    
    args = parent_parser.parse_args()
    
    return args