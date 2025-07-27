import argparse
import os
import torch
import json
import time
import subprocess

import re

from vllm import LLM, SamplingParams

def parse_option(): ## test codes series
    parser = argparse.ArgumentParser() ### use absolute path globally  
    parser.add_argument('--model_name', type = str)
    parser.add_argument('--dataset_name', type = str, default = "dev_bird_0627_10b") # original dataset name
    parser.add_argument('--db_path', type = str, default = "../data/bird/dev_20240627/dev_databases") # the test base
    parser.add_argument('--model_family', type=str, default='codellama') # model family devides the preprocessing template
    parser.add_argument('--select_ckpt', type=int, default=0) # if specifies single ckpt to sample
    opt = parser.parse_args()

    return opt

opt = parse_option()

sample_strategy = ['default', 'greedy'] 

# infer args
args = {
    'i': opt.dataset_name, # dataset name 
    'd': opt.db_path, # database path 
    'n': 4, # num of gpus
    'g': '0,1,2,3', # cuda devices
    'm': opt.model_name, # model folder
    's': 70, # training step 
    't': 'default',
    'f': opt.model_family
}

def fetch_checkpoint_folders(path_prefix, model_folder):
    checkpoint_folders = []
    for root, dirs, files in os.walk(path_prefix + model_folder):
        for dir in dirs:
            if 'checkpoint' in dir:
                checkpoint_folders.append(dir)
    return checkpoint_folders

def parse_checkpoint_suffix(ckpts: list) -> dict:
    ckpt_dict = {}
    for ckpt in ckpts:
        if 'checkpoint' in ckpt:
            ckpt_dict[ckpt] = int(ckpt.split('-')[1])
    # sort by step
    ckpt_dict = dict(sorted(ckpt_dict.items(), key=lambda item: item[1]))
    return ckpt_dict

def parse(configuration: dict) :
    '''parse a config dict into command line arguments'''
    args = []
    for key, value in configuration.items():
        if value is True:
            args.append(f"-{key}")
        elif value is False:
            pass
        else:
            args.append(f"-{key}")
            args.append(str(value))
    return args

path_prefix = '../' # where root directory model save
model_folder = opt.model_name
ckpt_folders = fetch_checkpoint_folders(path_prefix, model_folder)
ckpt_suffix_dict = parse_checkpoint_suffix(ckpt_folders)

for strategy in sample_strategy :
    args['t'] = strategy
    for ckpt in ckpt_folders :
        if opt.select_ckpt != 0 and ckpt_suffix_dict[ckpt] != opt.select_ckpt : # selected ckpt and this is not the desired one
            continue
        args['s'] = ckpt_suffix_dict[ckpt]
        saving_name= model_folder + '-checkpoint-' + str(args['s'])
        output_dir = f'../results/{saving_name}_sampling_{strategy}_{opt.dataset_name}.json'
        print(' '.join(['bash', 'multi-device_sample.sh']+parse(args)))
        if os.path.exists(output_dir) == True or strategy == 'default' and os.path.exists(f'../results/{saving_name}_sampling_{opt.dataset_name}.json') :
            print('== sample outcome already exist ==') # no need to sample again
            continue
        subprocess.run(' '.join(['bash', 'multi-device_sample.sh']+parse(args)), shell=True)
