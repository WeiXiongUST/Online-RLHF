import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
# 设置模型和设备
model_name = "RLHFlow/Llama3-v2-iterative-DPO-iter2"
device = torch.device("cuda:8")  # 或 "cpu"

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, torch_dtype=torch.bfloat16
).to(device)
model.eval()
ds = load_dataset("raftstudy/ultrafeedback_pairs", split='train').select(range(20000))

all_data = []
t = 0
for sample in ds:
    t += 1
    if t % 1000 == 0:
        print(t)
    chosen_text = tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=False, add_special_tokens=False)
    rej_text = tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=False, add_special_tokens=False)
    #chosen_text = sample['chosen'][0]['content'] + " " + sample['chosen'][1]['content'] 
    #rej_text = sample['rejected'][0]['content'] + " " + sample['rejected'][1]['content'] 
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

    # 收集样本
    all_data.append({
        "chosen": sample['chosen'],
        "rejected": sample['rejected'],
        'chosen_feature': chosen_feature_vector,
        'rejected_feature': rej_feature_vector
    })

# 保存为 JSON 文件
with open("llama3_v2_train_test.json", "w") as f:
    json.dump(all_data, f, indent=2)

print("保存完毕，共保存了", len(all_data), "个样本。")
