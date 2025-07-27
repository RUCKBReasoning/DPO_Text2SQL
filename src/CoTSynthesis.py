import json
import tqdm
import requests
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type = str)
    parser.add_argument('--sample_budget', type = int, default = 16)
    opt = parser.parse_args()
    return opt

arg = parse_option()

model = 'gpt-4o-mini-2024-07-18'
# system message template
system_message = open('../data/system.txt', 'r').read()
# load trainset
trainset = json.loads(open('../data/train_bird.json').read())

def text2sql_item(item, model='gpt-4-0125-preview', n=1, t=0.0) :
    url = 'https://api.openai.com/v1/chat/completions'
    api_key = arg.api_key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    prompt = item['input'] + f'"Reference Solution": "{item["output"]}"\n'
    params = {
        'model': model,
        'messages': [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': prompt}],
        'max_tokens': 2048, 
        'n': n,
        'temperature': t
    }
    response = requests.post(url, headers=headers, data=json.dumps(params))
    print(response)
    texts = [ json.dumps(text.get('message').get('content')) for text in response.json().get('choices')]
    
    output = item.copy()
    output['generated'] = texts
    return output

# request in parallel
collection = []
failed_task_id = []
executor = ThreadPoolExecutor(max_workers=20)
all_task = {executor.submit(text2sql_item, trainset[i], model, arg.sample_budget, 1.0) : i for i in range(len(trainset))}
for future in as_completed(all_task) :
    task_id = all_task[future] # fetch task id from dict
    try :
        res = future.result(timeout=300)
        res['question_id'] = task_id
        # print the task id, completion, and answer
        print('====================')
        print('task_id:', task_id)
        print('count:', len(res['generated']))
        # save the completion and answer to collection
        collection.append(res)
    except :
        failed_task_id.append(task_id)
executor.shutdown()
print('failed: ', failed_task_id)

# shepard the data
tailored = []
for col in tqdm.tqdm(collection) :
    gened = col['generated'].copy()
    d = col.copy()
    del d['generated']
    for sql in gened :
        d['output'] = json.loads(sql)
        tailored.append(d.copy())

tailored.sort(key=lambda x: x['question_id'])

# conversation format
collections = []
for i in range(len(tailored)): 
    col = {'messages': [] }
    user = tailored[i]['input']
    assistant = tailored[i]['output']
    col['messages'].append({'role': 'user', 'content': user})
    col['messages'].append({'role': 'assistant', 'content': assistant})
    #col['question_id'] = cot_data[i]['question_id']
    collections.append(col)
print(len(collections))

# save the data
json.dump(collections, open('../LLaMA-Factory/data/syn_cot_bird.json', 'w'), indent=2)