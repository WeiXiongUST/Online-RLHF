from datasets import Dataset, concatenate_datasets
import random
import json

# 载入 JSON 文件
def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# 载入两个数据集
#llama3_v2_train_first_20k.json
dataset1 = load_json_dataset("/home/wx13/latent/llama3_v2_dpo_iter1_test.json")
dataset2 = load_json_dataset("pre_trained_second_20k.json")

dataset1.push_to_hub("raftstudy/dpo_exp_llama3_v2_dpoiter1_test")
# 合并数据集
#combined_dataset = concatenate_datasets([dataset1, dataset2])

# 打乱顺序
#shuffled_dataset = combined_dataset.shuffle(seed=42)

# 验证一下
#print(shuffled_dataset[0])  # 查看第一个样本

#shuffled_dataset.push_to_hub("raftstudy/pretrained_rm_train_feature40k")
