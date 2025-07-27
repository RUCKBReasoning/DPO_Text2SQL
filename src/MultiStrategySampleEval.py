import argparse
import os
import torch
import json
import time
import subprocess
import argparse
import re

from datetime import datetime

# record start time
start_time = datetime.now()

def parse_option(): ## test codes series
    parser = argparse.ArgumentParser() ### use absolute path globally  
    parser.add_argument('--llm_name', type = str) 
    opt = parser.parse_args()

    return opt

def parse(configuration: dict) :
    '''parse a config dict into command line arguments'''
    args = []
    for key, value in configuration.items():
        args.append(f"-{key}")
        args.append(str(value))
    return args


args = parse_option()

eval_model_family = [
    (opt.llm_name, 'default', True),
]

eval_metrics = ['greedy', 'pass', 'maj'] # metrics use to measure model perfomance

# 1. do sample for all models - an continous time to use gpus
sample_args = {
    '-model_name': '',
    '-model_family': '',
    '-dataset_name': 'dev_bird_0627_10b', # specifies dataset
}
# parser.add_argument('--model_name', type = str, default = "codellama_7b_auto_test")
# parser.add_argument('--dataset_name', type = str, default = "dev_bird_0627_10b") # original dataset name
# parser.add_argument('--db_path', type = str) # the test base 
# parser.add_argument('--model_family', type=str, default='codellama') # model family devides the preprocessing template
for model_name, model_family, is_cot in eval_model_family :
    sample_args['-model_name'] = model_name
    sample_args['-model_family'] = model_family
    print(' '.join(['python', '-u', 'SingleModelSample.py']+parse(sample_args)))
    subprocess.run(' '.join(['python', '-u', 'SingleModelSample.py']+parse(sample_args)), shell=True)

# 2. do eval for every models across all metrics
eval_args = {
    '-model_name': '',
    '-is_cot': '',
    '-metric': ''
}
# parser.add_argument('--model_name', type=str, default='codellama_7b_auto_test') # the evaled model
# parser.add_argument('--dataset_split', type=str, default='dev') # the dataset split to be evaled
# parser.add_argument('--is_cot', type=bool, default=True) # if the model is a cot model
# parser.add_argument('--metric', type=str, default='greedy') # supported eval metrics in ['maj', 'pass', 'greedy']
for model_name, model_family, is_cot in eval_model_family :
    eval_args['-model_name'] = model_name
    eval_args['-is_cot'] = str(is_cot)
    print(f'== now starts the eval of {model_name} ==')
    for metric in eval_metrics :
        eval_args['-metric'] = metric
        print(' '.join(['python', '-u', 'EvalSingleModel.py']+parse(eval_args)))
        subprocess.run(' '.join(['python', '-u', 'EvalSingleModel.py']+parse(eval_args)), shell=True) 

# record end time
end_time = datetime.now()

# elapsed_time
elapsed_time = end_time - start_time

# standardize
hours, remainder = divmod(elapsed_time.seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"== total time elapsed: {hours}h {minutes}m {seconds}s")