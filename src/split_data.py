# usage: python split_data.py --input ./data/dev_bird.json --output_dir ./data/splits --num_splits 8

import argparse
import json
import os

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, default = "./data/dev_bird.json")
    parser.add_argument('--output_dir', type = str, default='./data/splits')
    parser.add_argument('--num_splits', type = int, default = 8)

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_option()
    print('Start Spliting Dataset...')
    print(opt)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    with open(opt.input, 'r') as f:
        data = json.load(f)

    split_size = (len(data) // opt.num_splits) + 1 
    split_data = [data[i*split_size:(i+1)*split_size] for i in range(opt.num_splits)]

    for i, split in enumerate(split_data):
        print(f'Split {i} size: {len(split)}')

    # prefix
    prefix = opt.input.split('/')[-1].split('.')[0]

    for i, split in enumerate(split_data):
        with open(os.path.join(opt.output_dir, prefix + f'_part{i}.json'), 'w') as f:
            json.dump(split, f, indent=2)

    print('Splitting completed.') 