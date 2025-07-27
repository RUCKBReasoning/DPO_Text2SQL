import json
import torch
from datasets import Dataset
from torch.utils.data import Dataset


def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = tokenizer(prefix_seq , truncation = False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation = False)["input_ids"][1:] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs(prefix_seq, tokenizer, max_prefix_length, template = 'deepseekcoder'):

    format_seq = prefix_seq
    sys_seq = ''

    # the ds-coder chat format
    if template == 'deepseekcoder' :
        sys_seq = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n"
        format_seq = f'### Instruction:\n{prefix_seq}\n### Response:'

    # the qwen-coder chat format
    if template == 'qwencoder' :
        sys_seq = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
        format_seq = f'<|im_start|>user\n{prefix_seq}<|im_end|>\n<|im_start|>assistant\n'

        ## multi-turn conversation
        format_seq = f'{prefix_seq}<|im_start|>assistant\n'

    # the codes chat format
    if template == 'codes' :
        sys_seq = ''
        format_seq = f'<|endoftext|>Human:\n{prefix_seq}\nAssistant:\n'

    # the deepseek-llm chat format
    if template == 'deepseek' :
        sys_seq = ''
        format_seq = f'User: {prefix_seq}\n\nAssistant:' 

    # the llama chat format
    if template == 'llama' :
        sys_seq = ''
        format_seq = f'<|start_header_id|>user<|end_header_id|>\n\n{prefix_seq}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'    

    input_ids = tokenizer(sys_seq+format_seq, truncation = False)["input_ids"]

    #print(sys_seq+format_seq)
    #print(input_ids)
    #print(tokenizer.decode(input_ids))

    #input_ids = tokenizer(prefix_seq , truncation = False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }

class SFTSQLGenerationDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, template, max_tokens, mode):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.template = template
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = data["input"]

        if self.mode == "train":
            target_seq = data["output"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode == "eval":
            return prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens, self.template)

    def __len__(self):
        return len(self.dataset)
