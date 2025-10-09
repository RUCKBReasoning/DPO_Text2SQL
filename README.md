# DPO Text-to-SQL

This repository is the official implementation of ACL 2025 Main Long paper: *Uncovering the Impact of Chain-of-Thought Reasoning for Direct Preference Optimization: Lessons from Text-to-SQL*, providing a step-by-step tutorial to reproduct the main experiment of the paper for a 7b-scale base model with 4xA100 GPUs (~1.5d). 

### News

(25.2.17) Synthetic CoT data (Sec 3.2) and preprocessed database prompt for Bird dataset (Appendix E) can be downloaded via [Google Drive](https://drive.google.com/file/d/1l0JeJ6hqaM6py4r2Vacl7GsYI06FtjXT/view?usp=share_link) for reproduction. 

[25.3.11] Prefercence data collected from Syn CoT SFT Qwen2.5-7B-Instruct under default configuation is now available at [Google Drive](https://drive.google.com/file/d/12sotBI7OlNfJ2fp2hNa_7Zy7RrAE5PXe/view?usp=sharing). 

[25.5.16] 🎉🥂 Our paper has been accepted to ACL 2025 main conference, see you in Vienna! 

### Step 0: Environment Setup

Our project uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training (with [flash-attention](https://github.com/Dao-AILab/flash-attention) enabled) and [vllm](https://github.com/vllm-project/vllm) for inference. The [Train Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip) and [Dev Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) of the [Bird](https://bird-bench.github.io) dataset need to be downloaded and extracted into the `/data/bird/` directory. The [data (Google Drive)](https://drive.google.com/file/d/1gwG_p9pvcqcktgHFEYdJaOwYFmiRrDtb/view?usp=sharing) we uploaded includes inputs processed from the database schema as described in Section 3. `train_bird.json` , `dev_bird_0627_10b.json` and `conversation.json` should be extracted and placed in the `/data/` directory. Also, the base model folder should be placed in the project root directory.

**Install LLaMA-Factory and auxiliary packages**

```bash
conda create -n llama_factory python=3.11 -y
conda activate llama_factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory 
git fetch origin 50b44d3c6da7c9cb24a60fedec171fb1de3e764a
git switch --detach 50b44d3c6da7c9cb24a60fedec171fb1de3e764a
pip install -e ".[torch,metrics]"
pip install flash-attn --no-build-isolation
pip install transformers==4.47.1
pip install trl==0.8.6
pip install deepspeed==0.15.0
pip install tensorboardX
```

> [!IMPORTANT]
>
> If you encounter problems when installing flash-attention, please try to download [lastest release](https://github.com/Dao-AILab/flash-attention/releases) according to your environment and manually install by `pip install`. For example, if your cuda version is `12.1.1`, with torch `2.2`, and python `3.9`, you can download `flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`.

**Install vllm v0.5.5**

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
export VLLM_VERSION=0.5.5
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
pip install func_timeout
```

> [!NOTE]
>
> We use a separate virtual environment for vllm since the dependencies may conflict.

### Step 1: Chain-of-Thought Synthesis

The first step of the pipeline is to use `GPT4o-mini` to synthesis chain-of-thought solution for Train Set of Bird (~0.5h). Run to following script in root directory: 

```bash
cd src
python -u CoTSynthesis.py --sample_budget 16 --api_key OPENAI_API_KEY 
```

 `sample_budget` specifies how many chain-of-thought solution paths to synthesize for each data point in the training set. The generated data will be placed in the `/LLaMA-Factory/data/` directory. You can also download it from our data attachment and unzip it in the same directory.

Finally, for subsequent training, you need to register the data into the LLaMA-Factory dataset list. To do this, insert the following content into `/LLaMA-Factory/data/dataset_info.json` to complete the registration:

```json
"syn_cot_bird": {
    "file_name": "syn_cot_bird.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
```

### Step 2: Supervised Fine-tuning

Run the following script in the project directory to perform supervised fine-tuning using 4 GPUs (~1 day). You need to replace `BASE_MODEL_NAME` and `SFT_MODEL_NAME` with the folder name of the base model and the trained model, respectively. Note that our inference and evaluation scripts assume models are placed in the project root folder. 

```bash
conda activate llama_factory
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
	--model_name_or_path BASE_MODEL_NAME \
	--stage sft \
	--do_train \
	--finetuning_type full \
	--deepspeed examples/deepspeed/ds_z3_config.json \
	--dataset syn_cot_bird \
	--template default \
	--cutoff_len 4096 \
	--overwrite_cache \
	--preprocessing_num_workers 16 \
	--output_dir SFT_MODEL_NAME \
	--logging_steps 5 \
	--report_to tensorboard \
	--run_name sft \
	--save_steps 1 \
	--plot_loss \
	--save_strategy epoch \
	--save_only_model \
	--overwrite_output_dir \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--learning_rate 1e-05 \
	--num_train_epochs 4 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.05 \
	--bf16 \
	--use_fast_tokenizer \
	--flash_attn fa2
```

After completing the training, evaluate the model by running the following script. Use 4 GPUs for parallel inference and perform parallel evaluation on the Bird development set, obtaining performance metrics under `greedy`, `pass@16`, and `maj@16` (~1.5 h).

```bash
cd src
python -u ./src/MultiStrategySampleEval.py --llm_name SFT_MODEL_NAME
```

In addition to being displayed on the terminal, the evaluation results are also stored in the model directory as files. For example, with `maj@16`, you can print the test results using `cat maj_ckpt.json` and select the best checkpoint based on this to serve as the reference model for the DPO phase.

### Step 3: Direct Preference Optimization

First, use the following command to perform 4-GPU parallel sampling on the Bird train set using the reference model. (~0.5h)

```bash
cd src
bash multi-device_sample.sh -i train_bird -d ../data/bird/train/train_databases -n 4 -m SFT_MODEL_NAME -s REFERENCE_CHECKPOINT -t default -f default -g 0,1,2,3
```

Next, use the following commands to evaluate the sampling results in parallel, then automatically construct the DPO dataset and register it. (~2h)

```bash
mv {SFT_MODEL_NAME}-checkpoint-{REFERENCE_CHECKPOINT}_default_train_bird.json sft_model_sample.json
python -u evaluate_bird_ex_sampling.py --pred ../results/sft_model_sample.json --gold ../data/bird/train/train.json --db_path ../data/bird/train/train_databases --output ../results/eval_sft_model_sample.json
python -u CreateDPODataset.py --sft_dataset syn_cot_bird --chat_data ../data/conversation.json --ckpt REFERENCE_CHECKPOINT --sft_data ../data/train_bird.json
```

> [!IMPORTANT]
>
> The database in the Bird Train Set is large, causing the SQL execution results to occupy a significant amount of memory. An excessive number of parallel cores or a too long timeout can easily lead to memory overflow, causing the server to crash. The number of parallel cores and timeout can be adjusted in the `execute_sqls_parallel` function of `./src/evaluate_bird_ex_sampling.py` using the `num_cpus` and `timeout` parameters. The default values are 20 cores and 20 seconds, and it is not recommended to exceed these values.

Next, perform 4-GPU DPO training using the following script (~5h).

```bash
conda activate llama_factory
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
	--model_name_or_path SFT_MODEL_PATH \
	--stage dpo \
	--do_train \
	--finetuning_type full \
	--pref_beta 0.1 \
	--deepspeed examples/deepspeed/ds_z3_config.json \
	--dataset dpo_syn_cot \
	--template default \
	--cutoff_len 4096 \
	--overwrite_cache \
	--preprocessing_num_workers 16 \
	--output_dir DPO_MODEL_PATH \
	--logging_steps 5 \
	--save_strategy epoch \
	--report_to tensorboard \
	--save_steps 1 \
	--save_only_model \
	--plot_loss \
	--overwrite_output_dir \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--learning_rate 1e-06 \
	--num_train_epochs 8 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.05 \
	--bf16 \
	--use_fast_tokenizer \
	--flash_attn fa2
```

Finally, evaluate the model by running the following script. Use 4 GPUs for parallel inference and perform parallel evaluation on the Bird development set, obtaining performance metrics under `greedy`, `pass@16`, and `maj@16` (~3 h).

```bash
python -u MultiStrategySampleEval.py --llm_path DPO_MODEL_NAME
```

---

If This project helps you, please cite the following paper properly. 🥳 

```la
@inproceedings{liu-etal-2025-uncovering,
    title = "Uncovering the Impact of Chain-of-Thought Reasoning for Direct Preference Optimization: Lessons from Text-to-{SQL}",
    author = "Liu, Hanbing  and
      Li, Haoyang  and
      Zhang, Xiaokang  and
      Chen, Ruotong  and
      Xu, Haiyong  and
      Tian, Tian  and
      Qi, Qi  and
      Zhang, Jing",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1031/",
    pages = "21223--21261",
    ISBN = "979-8-89176-251-0"
}
```





