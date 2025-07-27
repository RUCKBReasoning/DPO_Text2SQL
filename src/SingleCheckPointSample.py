import argparse
import os
import torch
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from vllm import LLM, SamplingParams

def parse_option(): ## test codes series
    parser = argparse.ArgumentParser() ### use absolute path globally  
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--dataset_path', type = str, default = "./data/dev_bird.json") # original input from dataset 
    parser.add_argument('--db_path', type = str, default = "./bird/dev/dev_databases") # the test base 
    parser.add_argument('--sampling_output', type = str, default='test_sampling.json') # 
    parser.add_argument('--sample_strategy', type = str, default='default') # ['default', 'greedy']
    parser.add_argument('--sample_budget', type=int, default=16) # the sampling budget, default 16 across all stages  

    parser.add_argument('--table_num', type = int, default = 7) # configuraiton of retrivers 
    parser.add_argument('--column_num', type = int, default = 20)
    parser.add_argument('--max_tokens', type = int, default = 4096) # context windows
    parser.add_argument('--max_new_tokens', type = int, default = 2048) # default 2048 for CoT models, 256 for non-CoT models

    #parser.add_argument('--cot', type=bool, default=True) # whether the model is Chain-of-Thought model, determines length and sql extraction
    parser.add_argument('--model_family', type=str, default='deepseekcoder') # model family devides the preprocessing template

    opt = parser.parse_args()

    return opt

def extract_sql_from_cot(text, is_cot = True):
    # Extract all SQL code blocks from the generated text and return the last one
    
    if is_cot == False : # non-cot output no need to extract answers
        return text 
    matches = re.findall(r"```(?i:sql)*\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last matched SQL
    matches = re.findall(r"`(.*?)`", text, re.DOTALL) # Inline code 
    if matches:
        return matches[-1].strip()  # Return the last matched SQL
    return text # else is not cot model

if __name__ == "__main__":
    
    opt = parse_option()
    print('== start sample on ' + '/'.join(opt.llm_path.split('/')[-2:]) + ' ==')
    print(opt)

    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    model_to_template = { # maps model name to its corresponding templates in vllm inference
        'codes': 'codes',
        'codellama': 'deepseekcoder',
        'llama': 'llama',
        'deepseekcoder': 'deepseekcoder',
        'deepseek': 'deepseek',
        'qwencoder': 'qwencoder',
        'qwen': 'qwencoder',
        'default': 'default'
    }

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        model_to_template[opt.model_family], # provide support for different template
        8192 - max_new_tokens, # truncate input if no room for new tokens
        "eval"
    )

    dataloader = DataLoader(eval_set, batch_size = 1)
    model = LLM(
        model = opt.llm_path,
        #load_format = 'safetensors',
        #dtype = 'bfloat16',
        max_model_len = max_tokens,
        gpu_memory_utilization = 0.9,
        #max_num_seqs = 1,
        tensor_parallel_size = 1,
        swap_space = 64,
        enforce_eager=True,  
        disable_custom_all_reduce=True,
        #max_num_batched_tokens = 4096,
        enable_chunked_prefill = False
    )

    start_time = time.time()
    predicted_sqls = []
    
    prompt_token_ids = []
    for data in dataloader: 
        prompt_token_ids.append(data['input_ids'][0].tolist())
    #prompt_token_ids = prompt_token_ids[:10]

    print(tokenizer.decode(prompt_token_ids[0])) # test if template is loaded correctly
    #print(prompt_token_ids[0])

    t = 0.0 if opt.sample_strategy == 'greedy' else 1.0 # if greedy, temperature is set to zero
    sample_budget = opt.sample_budget 
    if opt.sample_strategy == 'greedy' : # if takes greedy strategy, only one shot is needed 
        sample_budget = 1
    sampling_params = SamplingParams(
        temperature = t, 
        n = sample_budget, # sample budget
        top_k = 32, 
        #use_beam_search = False,
        #early_stopping = False,
        max_tokens = max_tokens,
        #stop_token_ids = [tokenizer.eos_token_id, 32021],
        stop = ['<|EOT|>', '<|eot_id|>'], # stop conditions include this string(deepseekcoder) or model's end of sentence special token(other models)
        #include_stop_str_in_output = True # the stop token(s) are not included in the final output
    )
    
    outputs = model.generate(
        #prompt_token_ids,
        sampling_params = sampling_params,
        prompt_token_ids = prompt_token_ids
    ) 

    all_predicted_sqls = []
    for output in tqdm(outputs) :
        generated_sqls = [ o.text for o in output.outputs ]
        all_predicted_sqls.append(generated_sqls)
        
    with open(opt.sampling_output, "w", encoding="utf-8") as f:
        f.write(json.dumps(all_predicted_sqls, indent=2, ensure_ascii=False))

    end_time = time.time()
    print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        opt.llm_path, 
        end_time - start_time,
        len(raw_dataset),
        (end_time - start_time) / len(raw_dataset)
        )
    )
