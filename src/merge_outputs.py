# usage: python merge_outputs.py --input_dir ./data/splits --output bird_predict_dev.json --predix dev_bird --num_splits 8

import argparse
import json
import os

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, default = "./data/splits")
    parser.add_argument('--output', type = str, default='bird_predict_dev.json')
    parser.add_argument('--prefix', type = str, default='dev_bird')
    parser.add_argument('--num_splits', type = int, default = 8)
    parser.add_argument('--type', type = str, default ='infer')

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_option()
    print('Start Merging Results...')
    print(opt)

    if opt.type == 'infer' :
        merged_data = {}
        for i in range(opt.num_splits):
            with open(os.path.join(opt.input_dir, f'{opt.prefix}_part{i}.json'), 'r') as f:
                data = json.load(f)
                for j in range(len(data)) :
                    merged_data[str(len(merged_data))] = data[str(j)]
    elif opt.type == 'sample' :
        merged_data = []
        for i in range(opt.num_splits):
            with open(os.path.join(opt.input_dir, f'{opt.prefix}_part{i}.json'), 'r') as f:
                data = json.load(f)
                merged_data += data

    print(f'Merged data size: {len(merged_data)}')
    
    with open(opt.output, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print('Merging completed.')