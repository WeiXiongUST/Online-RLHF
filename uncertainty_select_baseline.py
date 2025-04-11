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
# if idx == "1":
if True:
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
