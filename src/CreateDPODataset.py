import sys
import sqlite3
import json
import argparse
import os
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import re

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type = str, default = 'cl_7b')
    parser.add_argument('--sft_dataset', type = str, default = 'train_bird_ins')
    parser.add_argument('--ckpt', type = int, default = 294)
    parser.add_argument('--sample', type = str, default = "../results/cl_7b_temp_chat_train_bird_ins-checkpoint-294_sampling_default_train_bird.json") # the sample outcomes
    parser.add_argument('--eval', type = str, default = "../results/cl_7b_temp_chat_train_bird_ins-checkpoint-294_eval_sampling_default_train_bird.json") # evaluation of sampled outcomes
    parser.add_argument('--sft_data', type = str, default = "../data/train_bird.json") # the original data with processed database prompt, to use its sql
    parser.add_argument('--chat_data', type = str, default = "../data/bird_train_cot_v1_sorted_chat.json") # the conversation format sft data, different from original dataset, we use the prompt
    parser.add_argument('--db_path', type = str, default = "../data/bird/train/train_databases") # database path, for query plan comparison
    parser.add_argument('--output', type = str, default = "../LLaMA-Factory/data/dpo_cl_7b_ori.json") # where to store the created dataset
    parser.add_argument('--register', type=bool, default=True) # whether to register the dataset to LlamaFactory
    parser.add_argument('--dataset_name', type=str, default="dpo_cl_7b_ori") # the dataset name for LlamaFactory to index 
    parser.add_argument('--budget', type=int) # the cap of sample budget 
    opt = parser.parse_args()
    return opt

def compare_query_plan(std_sql, pred_sql, db_id) :

    '''identify if two sqls are logically identical'''

    conn = sqlite3.connect(f'{opt.db_path}/{db_id}/{db_id}.sqlite')
    cursor = conn.cursor()
    try :
        cursor.execute('EXPLAIN ' + std_sql)
        std_query_plan = cursor.fetchall()
    except : # the ground truth may be faulty
        return True
    try :
        cursor.execute('EXPLAIN ' + pred_sql)
        pred_query_plan = cursor.fetchall()
        if std_query_plan == pred_query_plan :
            return True
        else :
            return False
    except :
        return True

def select_pairs(item, idx) :
    '''select a paired data of each instance for dpo'''
    np.random.seed(0)
    db_id = item['db_id']
    std_sql = item['ground_truth']
    
    #if hasattr(opt, 'budget') :
    #    item['pred_sqls'] = item['pred_sqls'][:opt.budget]
    #    item['ex_scores'] = item['ex_scores'][:opt.budget]

    pass_sqls = [sql for sql, ex in zip(item['pred_sqls'], item['ex_scores']) if ex == 1 ]
    fail_sqls = [sql for sql, ex in zip(item['pred_sqls'], item['ex_scores']) if ex == 0 ]
    if len(pass_sqls) == 0 or len(fail_sqls) == 0 : #  Cannot find a pair
        return None
    candidate_chosen = [sql for sql in pass_sqls if compare_query_plan(std_sql, sql, db_id) == False]
    if candidate_chosen == [] :
        return None
    chosen_index = -1
    reject_index = -1
    
    while True :
        chosen = np.random.choice(candidate_chosen)
        reject = np.random.choice(fail_sqls)
        for i, outcome in enumerate(sampling_outcomes[idx]) :
            if outcome.find(chosen) != -1 :
                chosen_index = i
            if outcome.find(reject) != -1 :
                reject_index = i
        if chosen_index == -1 or reject_index == -1 :
            continue
        else :
            break
    #print(chosen)
    #print('----------------')
    #print(reject)
    #print('================') 
    chosen_cot = sampling_outcomes[idx][chosen_index].strip()
    reject_cot = sampling_outcomes[idx][reject_index].strip()
    #print(chosen_cot)
    #print('----------------')
    #print(reject_cot)
    return chosen_cot, reject_cot

if __name__ == "__main__": 

    DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1'] # detect debugging mode 
    if DEBUG :
        print( '== operate in verbose mode ==' )

    opt = parse_option()

    # construct names from model prefix 
    opt.sample = f"../results/sft_model_sample.json"
    opt.eval = f"../results/eval_sft_model_sample.json"
    opt.dataset_name = f'dpo_syn_cot'
    opt.output = f'../LLaMA-Factory/data/{opt.dataset_name}.json'
    
    prompts = json.load(open(opt.sft_data))
    if DEBUG :
        print( 'dataset:' )
        print( prompts[1327], len(prompts) )

    data = json.load(open(opt.eval)) # the evaluation outcome 
    if DEBUG :
        print( 'eval result:' )
        print( data[10], len(data) )

    original_data = json.load(open(opt.chat_data)) # the chain-of-thought prompt, aligned with question id
    
    sampling_outcomes = json.load(open(opt.sample))
    if DEBUG :
        print( 'sample outcome: ')
        print( sampling_outcomes[7213][3] )

    collection = []
    for i, item in tqdm(enumerate(data)) :
        pair = select_pairs(item, item['question_id'])
        if pair == None :
            continue 
        instance = {} # we uniformly use the share gpt format
        instance['question_id'] = item['question_id']
        instance['messages'] = [original_data[i]['messages'][0]] # copy the prompt from templated sft data
        instance['chosen'] = { 'role': 'assistant', 'content': pair[0] }
        instance['rejected'] = { 'role': 'assistant', 'content': pair[1] }
        collection.append(instance)
    
    # print statistics concerning the data
    filename = opt.sample.split('/')[-1] 
    print(f'== DPO data construction for {filename} is done ==')
    pass_16 = len([1 for d in data if sum(d['ex_scores']) > 0 ]) / len(data)
    its = len(collection)
    print(f'pass@16: {pass_16}')
    print(f'    its: {its}')

    json.dump(collection, open(opt.output, 'w'), indent=2) # save the dataset to targeted location

    #exit()
    if opt.register != True :
        print('== no registration, work done ==')
        exit()

    # to register the created dataset to LlamaFactory
    datainfo_path = '../LLaMA-Factory/data/dataset_info.json'
    datasets = json.load(open(datainfo_path, 'r'))
    sharegpt_template = {
        "file_name": opt.output.split('/')[-1],
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
        "messages": "messages",
        "chosen": "chosen",
        "rejected": "rejected"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
        }
    }
    datasets[opt.dataset_name] = sharegpt_template
    json.dump(datasets, open(datainfo_path, 'w'), indent=2) # update the dataset info file
    print(f'== registration finished, the dataset named {opt.dataset_name} ==')
    
    