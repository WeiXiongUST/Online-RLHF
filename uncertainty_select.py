import json
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from accelerate import Accelerator
import os

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_path: Optional[str] = field(
        default="uf_split0_responses_K8.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the output file"},
    )
    model_path: Optional[str] = field(
        default="RLHFlow/LLaMA3-SFT-v2",
        metadata={"help": "the model for embedding"}
    )

accelerator = Accelerator()
device = accelerator.device
#print(device)
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset_path = script_args.dataset_path
output_dir = script_args.output_dir
model_path = script_args.model_path

# ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
# ds = load_dataset("json", data_files=ds_dir, split="train")
#ds = ds.select(range(100))
local_rank = Accelerator().local_process_index

# data_size = len(ds["prompt"])

# share = int(data_size / world_size) + 1
# ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

idx = output_dir.split(".")[-2][-1]
if idx == "1":
    data = []
    preference_data = []
    
    with open(dataset_path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    for sample in data:
        max_idx = sample['rewards'].index(max(sample['rewards']))
        min_idx = sample['rewards'].index(min(sample['rewards']))
        if sample['rewards'][max_idx] - sample['rewards'][min_idx] < 0.1:
            continue
        chosen = sample['responses'][max_idx]
        rejected = sample['responses'][min_idx]
        
        chosen_reward = sample['rewards'][max_idx]
        rejected_reward = sample['rewards'][min_idx]
        
        tmp = {
            "prompt": sample['prompt'],
            "chosen": chosen,
            "rejected": rejected,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward
        }
        preference_data.append(tmp)
        
    with open(output_dir, "w", encoding="utf8") as f:
        for i in range(len(preference_data)):
            json.dump(preference_data[i], f, ensure_ascii=False)
            f.write('\n')
else:
    prev_preference_data = []
    idx = int(idx)
    output_prefix = output_dir.replace(".json","")[:-1]
    for i in range(1,10):
        if i < idx:
            with open(f"{output_prefix}{str(i)}.json",'r') as f:
                for line in f:
                    prev_preference_data.append(json.loads(line))
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, torch_dtype=torch.bfloat16
    ).to(local_rank)
    model.eval()
    
    feature_data = {
        "chosen_feature": [],
        "rejected_feature": []
    }
    ## processing the previous pair (construct the matrix)
    data_size = len(prev_preference_data)

    share = int(data_size / world_size) + 1
    batch_data = prev_preference_data[local_rank * share:min((local_rank + 1) * share, len(prev_preference_data))]
    
    for sample in tqdm(batch_data[:]):
        chosen_text = sample['prompt'] + sample['chosen'] + tokenizer.eos_token
        rej_text = sample['prompt'] + sample['rejected'] + tokenizer.eos_token
        # 编码
        chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(device)
        rej_inputs = tokenizer(rej_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            chosen_base_outputs = model.base_model(**chosen_inputs, return_dict=True)
            chosen_last_hidden_state = chosen_base_outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            chosen_pooled = chosen_last_hidden_state[:, -1, :]  # 取 [CLS] 位置的表示

            rej_base_outputs = model.base_model(**rej_inputs, return_dict=True)
            rej_last_hidden_state = rej_base_outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            rej_pooled = rej_last_hidden_state[:, -1, :]  # 取 [CLS] 位置的表示

        chosen_feature_vector = chosen_pooled.squeeze(0).float().cpu().tolist()
        rej_feature_vector = rej_pooled.squeeze(0).float().cpu().tolist()
        
        # feature_data.append({
        #     'chosen_feature': chosen_feature_vector,
        #     'rejected_feature': rej_feature_vector
        # })
        feature_data['chosen_feature'].append(chosen_feature_vector)
        feature_data['rejected_feature'].append(rej_feature_vector)
    
    train_dataset = Dataset.from_dict(feature_data)
    #print(len(feature_data['chosen_feature']))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": feature_data,
    }

    import torch.distributed as dist

    dist.all_gather_object(all_process_list, data_to_send)
    gathered_data = []

    #print(len(all_process_list[0]["data"]))
    #print(len(all_process_list[0]["data"]['chosen_feature']))
    for i in range(world_size):
        #tmp_data = [tmp for tmp in all_process_list[i]["data"]]
        gathered_data.append(all_process_list[i]["data"])
    
    #print(len(gathered_data))
    #print(gathered_data[0])
    #print(gathered_data[0]['chosen_feature'][:2])
    all_train_dataset = concatenate_datasets([Dataset.from_dict(i) for i in gathered_data])
    #print(len(all_train_dataset))
    current_data = []
    with open(dataset_path,'r') as f:
        for line in f:
            current_data.append(json.loads(line))
        
    def compute_diff(example):
        return {"diff": np.array(example["chosen_feature"]) - np.array(example["rejected_feature"])}

    all_data = all_train_dataset.map(compute_diff)

    # 转换为 numpy 数组
    diff_matrix = np.stack(all_data["diff"])  # shape: [N, 4096]

    # 计算协方差矩阵 A
    A = np.cov(diff_matrix, rowvar=False)  # shape: [4096, 4096]
    
    #arr_eps = [1.0, 0.1, 0.5, 0.01]
    epsilon = 0.1
    tmp_A = A + epsilon * np.eye(A.shape[0])

    # 对新测试向量 x 计算 x^T A^{-1} x
    A_inv = np.linalg.inv(tmp_A)

    def compute_mahalanobis_batch(x_list, A_inv):
        return [x.T @ A_inv @ x for x in x_list]
    
    data_size = len(current_data)

    share = int(data_size / world_size) + 1
    batch_current_data = current_data[local_rank * share:min((local_rank + 1) * share, len(current_data))]
    
    preference_data = []
    tmp_feature_data = {
        "chosen_feature": [],
        "rejected_feature": []
    }
    store_chosen_answers = []
    store_rejected_answers = []
    for sample in tqdm(batch_current_data[:]):
        tmp = [(sample['responses'][i],sample['rewards'][i]) for i in range(len(sample['responses']))]
        #print(tmp)
        #print(len(tmp))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        #print(len(tmp))
        half = int(len(tmp)/2)
        #print(half)
        chosen_answers = tmp[:half]
        rejected_answers = tmp[half:]
        
        store_chosen_answers.append(chosen_answers)
        store_rejected_answers.append(rejected_answers)
        
        # tmp_feature_data = {
        #     "chosen_feature": [],
        #     "rejected_feature": []
        # }
        for i in range(len(chosen_answers)):
            for j in range(len(rejected_answers)):
                chosen_text = sample['prompt'] + chosen_answers[i][0] + tokenizer.eos_token
                rej_text = sample['prompt'] + rejected_answers[j][0] + tokenizer.eos_token
                # 编码
                chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(device)
                rej_inputs = tokenizer(rej_text, return_tensors="pt").to(device)
        
                with torch.no_grad():
                    chosen_base_outputs = model.base_model(**chosen_inputs, return_dict=True)
                    chosen_last_hidden_state = chosen_base_outputs.last_hidden_state  # [1, seq_len, hidden_dim]
                    chosen_pooled = chosen_last_hidden_state[:, -1, :]  # 取 [CLS] 位置的表示

                    rej_base_outputs = model.base_model(**rej_inputs, return_dict=True)
                    rej_last_hidden_state = rej_base_outputs.last_hidden_state  # [1, seq_len, hidden_dim]
                    rej_pooled = rej_last_hidden_state[:, -1, :]  # 取 [CLS] 位置的表示

                chosen_feature_vector = chosen_pooled.squeeze(0).float().cpu().tolist()
                rej_feature_vector = rej_pooled.squeeze(0).float().cpu().tolist()

                tmp_feature_data['chosen_feature'].append(chosen_feature_vector)
                tmp_feature_data['rejected_feature'].append(rej_feature_vector)
                
    tmp_eval_dataset = Dataset.from_dict(tmp_feature_data)
    tmp_eval_dataset = tmp_eval_dataset.map(compute_diff)
        
    x_test = np.array(tmp_eval_dataset['diff'])
    results = compute_mahalanobis_batch(x_test, A_inv)
    
    for count,i in tqdm(enumerate(range(0,len(results),16))):
        tmp_results = results[i:i+16]

        max_idx = tmp_results.index(max(tmp_results))
        chosen_idx = int(max_idx/len(chosen_answers))
        rejected_idx = int(max_idx % len(rejected_answers))
        
        tmp = {
            "prompt": batch_current_data[count]['prompt'],
            "chosen": store_chosen_answers[count][chosen_idx][0],
            "rejected": store_rejected_answers[count][rejected_idx][0],
            "chosen_reward": store_chosen_answers[count][chosen_idx][1],
            "rejected_reward": store_rejected_answers[count][rejected_idx][1]
        }
        preference_data.append(tmp)
    
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": preference_data,
    }

    import torch.distributed as dist

    dist.all_gather_object(all_process_list, data_to_send)
    gathered_data = []

    for i in range(world_size):
        #tmp_data = [tmp for tmp in all_process_list[i]["data"]]
        gathered_data = gathered_data + all_process_list[i]["data"]
    
    if local_rank == 0:    
        with open(output_dir, "w", encoding="utf8") as f:
            for i in range(len(gathered_data)):
                json.dump(gathered_data[i], f, ensure_ascii=False)
                f.write('\n')
