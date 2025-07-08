from __future__ import print_function
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.utils.data
import matplotlib.pyplot as plt
import warnings

from torchvision import datasets, transforms
from model.model_architectures import construct_model
from utils.reconstructed import convert_relu_to_sigmoid
from utils.config import config
from model.generator import trainer
import os
import json
warnings.filterwarnings("ignore")

import torchvision.transforms.functional as TF
from datetime import datetime

dataset = datasets.CIFAR10(
        root='./dataset/cifar10-data/', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

config['device'] = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
config['architecture_search'] = True
config['classes'] = 10
config['model_search_space_path'] = "model_search_space.txt"

# Initialize the manual seed and model
arch = 'ResNet18'
model, _ = construct_model(arch, seed=42, num_classes=config['classes'], num_channels=3)
model = model.to(config['device'])
convert_relu_to_sigmoid(model) # Following GION

start_time = datetime.now()
print(f'start_time: {start_time}')

# model
size_batch = 4
config["nz"] = 128
config['total_img'] = size_batch
config['b_size'] = size_batch
config['num_epochs'] = 30000
config['rep_freq'] = config['num_epochs']/250
num_exps = 8
config['num_exps'] = num_exps
config["dst"] = "cifar10"
num_epochs = config['num_epochs']
exp_name = f'cifar10_num_epochs_{num_epochs}_size_batch_{size_batch}_num_exps_{num_exps}'
if config['architecture_search']:
    exp_name = f'{exp_name}_architecture_search'

# defense_strategy = "noise"
# defense_strategy = "clipping"
# defense_strategy = "compression"
# defense_strategy = "representation"
defense_strategy = None
noise_param = 0.1
clipping_param = 4
compression_param = 90
representation_param = 80
config['defense_strategy'] = defense_strategy
config['noise_param'] = noise_param
config['clipping_param'] = clipping_param
config['compression_param'] = compression_param
config['representation_param'] = representation_param
if config['defense_strategy'] is not None:
    exp_name = f'{exp_name}_{defense_strategy}'

if config['architecture_search']:
    method = 'GI-NAS'
else:
    method = 'GION'
attack =  {'method':f'{method}','lr':0.001}
config['lr']=attack['lr']
trainer = trainer(config,attack,model,dataset)        

reconstructed_image,avg_score = trainer.attack_training()
ssim_list = []
fsim_list = []
psnr_list = []
lpips_list = []
for num_exp in range(num_exps):
    idx = str(num_exp)
    ssim_list.append(avg_score[idx][method]['ssim']['score'][-1])
    fsim_list.append(avg_score[idx][method]['fsim']['score'][-1])
    psnr_list.append(avg_score[idx][method]['psnr']['score'][-1])
    lpips_list.append(avg_score[idx][method]['lpips']['score'][-1])
ssim_avg = sum(ssim_list) / len(ssim_list)
fsim_avg = sum(fsim_list) / len(fsim_list)
psnr_avg = sum(psnr_list) / len(psnr_list)
lpips_avg = sum(lpips_list) / len(lpips_list)
print(f'ssim_avg: {ssim_avg}, fsim_avg: {fsim_avg}, psnr_avg: {psnr_avg}, lpips_avg: {lpips_avg}')
avg_score['total_avg'] = {}
avg_score['total_avg']['ssim_avg'] = ssim_avg
avg_score['total_avg']['fsim_avg'] = fsim_avg
avg_score['total_avg']['psnr_avg'] = psnr_avg
avg_score['total_avg']['lpips_avg'] = lpips_avg

with open(f'{exp_name}.json', 'w') as json_file:
    json.dump(avg_score, json_file)

for num_exp in range(config['num_exps']):
    result_folder_name = f'./result_cifar10/{exp_name}/batch-{num_exp}'
    imgs = reconstructed_image[str(num_exp)]["image"][method]['ssim']["timeline"][0][-1]
    os.makedirs(result_folder_name, exist_ok=True)
    for idx in range(imgs.shape[0]):
        img = TF.to_pil_image(imgs[idx])
        img.save(f'{result_folder_name}/{idx}.jpg')

end_time = datetime.now()
print(f'end_time: {end_time}')

print(f'total time consumed: {end_time - start_time}')