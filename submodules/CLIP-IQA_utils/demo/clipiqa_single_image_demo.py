# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np

import plotly.graph_objects as go
import plotly.offline as pyo

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_cobgs.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--base_path', default='/home/hit_zjx/project/CLIP-IQA/nvos/', help='base path to input image folders')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def calculate_average_attributes(model, method_path):
    attributes_list = []
    for scene_folder in os.listdir(method_path):
        scene_path = os.path.join(method_path, scene_folder)
        if os.path.isdir(scene_path):
            for img_file in os.listdir(scene_path):
                img_path = os.path.join(scene_path, img_file)
                output, attributes = restoration_inference(model, img_path, return_attributes=True)
                attributes_list.append(attributes.float().detach().cpu().numpy()[0])

    average_attributes = np.mean(attributes_list, axis=0)
    return average_attributes

def main():
    args = parse_args()
    model = init_model(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    methods = ['flashsplat', 'our', 'sa3d', 'sagd']
    average_attributes_dict = {}

    for method in methods:
        method_path = os.path.join(args.base_path, method)
        average_attributes = calculate_average_attributes(model, method_path)
        average_attributes_dict[method] = average_attributes
        print(f'Average attributes for {method}: {average_attributes}')

if __name__ == '__main__':
    main()