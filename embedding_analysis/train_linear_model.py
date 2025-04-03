import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datasets import load_dataset

# Dataset 类
class PairwiseDataset(Dataset):
    def __init__(self, json_path):
        #with open(json_path, 'r') as f:
        #    data = json.load(f)
        data = load_dataset(json_path, split='train')
        #print(data[0])
        self.chosen = torch.tensor([item["chosen_feature"] for item in data], dtype=torch.float32)
        self.rejected = torch.tensor([item["rejected_feature"] for item in data], dtype=torch.float32)
        self.labels = torch.ones(len(data), dtype=torch.float32)  # 所有 chosen 都是正例

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.chosen[idx], self.rejected[idx], self.labels[idx]

# 简单线性模型
class LinearScoringModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, chosen, rejected):
        score_chosen = self.linear(chosen).squeeze(-1)
        score_rejected = self.linear(rejected).squeeze(-1)
        score_diff = score_chosen - score_rejected
        prob = torch.sigmoid(score_diff)
        return prob

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for chosen, rejected, labels in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        preds = model(chosen, rejected)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for chosen, rejected, labels in tqdm(dataloader, desc="Evaluating"):
            preds = model(chosen, rejected)
            all_preds.append((preds > 0.5).float())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    return acc

def main():
    train_path = 'raftstudy/dpo_exp_llama3_v2_dpo_iter1_train_20k'
    #'raftstudy/dpo_exp_llama3_v2_sftmodel_train_20k'
    val_path = 'raftstudy/dpo_exp_llama3_v2_sftmodel_test'
    #'raftstudy/dpo_exp_llama3_v2_dpoiter1_test'
    #'raftstudy/dpo_exp_llama3_v2_sftmodel_test'
    #train_path = 'raftstudy/pretrained_rm_train_feature40k'
    #val_path = 'raftstudy/welltrained_rm_test_feature'
    
    model_path = 'linear_model.pt'

    # 数据加载
    train_set = PairwiseDataset(train_path)
    val_set = PairwiseDataset(val_path)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # 模型定义
    input_dim = 4096
    model = LinearScoringModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    acc_arr = []
    los_arr = []
    # 训练过程
    for epoch in range(10):
        print(f"Epoch {epoch}")
        loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, val_loader)
        acc_arr.append(acc)
        los_arr.append(loss)
        print(f"Loss: {loss:.4f}, Val Acc: {acc:.4f}")

    # 模型保存
    print(acc_arr, los_arr)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
