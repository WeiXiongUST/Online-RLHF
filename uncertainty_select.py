import json
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset_path = script_args.dataset_path
output_dir = script_args.output_dir
model_path = script_args.model_path

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
    ).to(0)
    model.eval()
    
    feature_data = {
        "chosen_feature": [],
        "rejected_feature": []
    }
    ## processing the previous pair (construct the matrix)
    for sample in tqdm(prev_preference_data[:100]):
        chosen_text = sample['prompt'] + sample['chosen'] + tokenizer.eos_token
        rej_text = sample['prompt'] + sample['rejected'] + tokenizer.eos_token
        # 编码
        chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(0)
        rej_inputs = tokenizer(rej_text, return_tensors="pt").to(0)
        
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
    
    current_data = []
    with open(dataset_path,'r') as f:
        for line in f:
            current_data.append(json.loads(line))
        
    def compute_diff(example):
        return {"diff": np.array(example["chosen_feature"]) - np.array(example["rejected_feature"])}

    all_data = train_dataset.map(compute_diff)

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
    
    preference_data = []
    for sample in tqdm(current_data[:100]):
        tmp = [(sample['responses'][i],sample['rewards'][i]) for i in range(len(sample['responses']))]
        #print(tmp)
        #print(len(tmp))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        #print(len(tmp))
        half = int(len(tmp)/2)
        #print(half)
        chosen_answers = tmp[:half]
        rejected_answers = tmp[half:]
        
        tmp_feature_data = {
            "chosen_feature": [],
            "rejected_feature": []
        }
        for i in range(len(chosen_answers)):
            for j in range(len(rejected_answers)):
                chosen_text = sample['prompt'] + chosen_answers[i][0] + tokenizer.eos_token
                rej_text = sample['prompt'] + rejected_answers[j][0] + tokenizer.eos_token
                # 编码
                chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(0)
                rej_inputs = tokenizer(rej_text, return_tensors="pt").to(0)
        
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

        max_idx = results.index(max(results))
        chosen_idx = int(max_idx/len(chosen_answers))
        rejected_idx = int(max_idx % len(rejected_answers))
        
        tmp = {
            "prompt": sample['prompt'],
            "chosen": chosen_answers[chosen_idx][0],
            "rejected": rejected_answers[rejected_idx][0],
            "chosen_reward": chosen_answers[chosen_idx][1],
            "rejected_reward": rejected_answers[rejected_idx][1]
        }
        preference_data.append(tmp)
        
    with open(output_dir, "w", encoding="utf8") as f:
        for i in range(len(preference_data)):
            json.dump(preference_data[i], f, ensure_ascii=False)
            f.write('\n')
