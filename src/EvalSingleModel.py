import yaml
import subprocess
import sys
import json
import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='codellama_7b_auto_test') # the evaled model
    parser.add_argument('--dataset_split', type=str, default='dev') # the dataset split to be evaled
    parser.add_argument('--is_cot', type=str, default='True') # if the model is a cot model
    parser.add_argument('--metric', type=str, default='greedy') # supported eval metrics in ['maj', 'pass', 'greedy']

    opt = parser.parse_args()
    return opt

opt = parse_option()
path_prefix = '.' # where root directory model save
eval_prefix = '.' 
#print(opt)
# eval args
eval_args = {
    '-gold': '../data/bird/dev_20240627/dev.json', # gold results
    '-db_path': '../data/bird/dev_20240627/dev_databases', # database path
    '-is_cot': opt.is_cot 
}

dataset_name = 'dev_bird_0627_10b'

if opt.dataset_split == 'train' : # change path
    eval_args['-gold'] = '../data/bird/train/train.json'
    eval_args['-db_path'] = './data/bird/train/train_databases'
    dataset_name = 'train_bird'

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
        #if value is True:
        #    args.append(f"-{key}")
        #elif value is False:
        #    pass
        #else:
        args.append(f"-{key}")
        args.append(str(value))
    return args

# change to the base directory
#new_directory = '..'
#os.chdir(new_directory)
#subprocess.run('pwd')

model_folder = opt.model_name

# sample on each checkpoint
ckpt_folders = fetch_checkpoint_folders(path_prefix, model_folder)
ckpt_suffix_dict = parse_checkpoint_suffix(ckpt_folders)
    
# eval major voting results: maj -> eval_single
if opt.metric == 'maj' :

    # major voting on each checkpoint
    for ckpt in ckpt_folders :
        pred = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_sampling_default_{dataset_name}.json'
        if os.path.exists(pred) == False : # history sample output saving suffix
            pred = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_sampling_{dataset_name}.json'
        output = eval_prefix + f'{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_voting_{dataset_name}.json'
        #if os.path.exists(output) == True : # have history major voting outcomes
        #    continue
        eval_args['-pred'] = pred
        eval_args['-output'] = output
        print(' '.join(['python', '-u', 'evaluate_bird_ex_sampling_maj.py']+parse(eval_args)))
        subprocess.run(' '.join(['python', '-u', 'evaluate_bird_ex_sampling_maj.py']+parse(eval_args)), shell=True)

    # evaluate major voting results    
    model_log = {}
    for ckpt in ckpt_folders:
        pred = eval_prefix + f'{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_voting_{dataset_name}.json'
        output = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_maj@16_{dataset_name}_results.json'
        if os.path.exists(output) == False or True: # need eval
            eval_args['-pred'] = pred
            eval_args['-output'] = output
            print(' '.join(['python', '-u', 'evaluate_bird_ex.py']+parse(eval_args)))
            subprocess.run(' '.join(['python', '-u', 'evaluate_bird_ex.py']+parse(eval_args)), shell=True)

        # read the log file record it
        result = json.load(open(output)) # [{'correctness': 0}, ...]
        total = len(result)
        correct = sum([r['correctness'] for r in result])
        # accuarcy save 3 decimal places
        acc = round(correct/total, 3)
        model_log[ckpt] = acc

    # print the evaluation results
    print(f'== eval metric: {opt.metric} ==')
    print(model_folder)
    print(model_log)
    # save to model dir
    with open(f'{path_prefix}/{model_folder}/maj_ckpts.json', 'w') as f:
        json.dump(model_log, f, indent=4)

# eval pass@16 results:     
if opt.metric == 'pass' :

    model_log = {}

    # evaluate sampling results
    for ckpt in ckpt_folders :
        pred = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_sampling_default_{dataset_name}.json'
        if os.path.exists(pred) == False : # history sample output saving suffix
            pred = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_sampling_{dataset_name}.json'
        output = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_eval_sampling_{dataset_name}.json'
        if os.path.exists(output) == False or True: # need eval
            eval_args['-pred'] = pred
            eval_args['-output'] = output
            print(' '.join(['python', '-u', 'evaluate_bird_ex_sampling.py']+parse(eval_args)))
            subprocess.run(' '.join(['python', '-u', 'evaluate_bird_ex_sampling.py']+parse(eval_args)), shell=True)
        
        # read the log file record it
        result = json.load(open(output)) # [{'ex_scores': [...]}, ..]
        total = len(result)
        correct = sum([sum(r['ex_scores'])/len(r['ex_scores']) for r in result])
        # accuarcy save 3 decimal places
        acc = round(correct/total, 3)
        model_log[ckpt] = acc

    # print the evaluation results
    print(f'== eval metric: {opt.metric} ==')
    print(model_folder)
    print(model_log)
    # save to model dir
    with open(f'{path_prefix}/{model_folder}/pass_ckpts.json', 'w') as f:
        json.dump(model_log, f, indent=4)

# eval greedy results
if opt.metric == 'greedy' :

    model_log = {}

    # evaluate sampling results
    for ckpt in ckpt_folders :
        pred = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_sampling_greedy_{dataset_name}.json'
        output = eval_prefix + f'./results/{model_folder}-checkpoint-{ckpt_suffix_dict[ckpt]}_eval_sampling_greedy_{dataset_name}.json'
        eval_args['-pred'] = pred
        eval_args['-output'] = output
        if os.path.exists(output) == False or True: # need eval
            print(' '.join(['python', '-u', 'evaluate_bird_ex_sampling.py']+parse(eval_args)))
            subprocess.run(' '.join(['python', '-u', 'evaluate_bird_ex_sampling.py']+parse(eval_args)), shell=True)
        
        # read the log file record it
        result = json.load(open(output)) # [{'ex_scores': [...]}, ..]
        total = len(result)
        correct = sum([sum(r['ex_scores']) for r in result]) # only 1 sample outcome for each item
        # accuarcy save 3 decimal places
        acc = round(correct/total, 3)
        model_log[ckpt] = acc

    # print the evaluation results
    print(f'== eval metric: {opt.metric} ==')
    print(model_folder)
    print(model_log)
    # save to model dir
    with open(f'{path_prefix}/{model_folder}/greedy_ckpts.json', 'w') as f:
        json.dump(model_log, f, indent=4)
        