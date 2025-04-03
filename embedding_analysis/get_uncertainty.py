from datasets import load_dataset, concatenate_datasets
import numpy as np

# 加载你的两个 HuggingFace 数据集
dataset1 = load_dataset("raftstudy/dpo_exp_llama3_v2_dpo_iter1_train_20k")["train"]
dataset2 = load_dataset("raftstudy/dpo_exp_llama3_v2_dpoiter1_test")["train"]

# 合并数据集
#all_data = concatenate_datasets([dataset1, dataset2])
all_data  = dataset1
# 提取差向量
def compute_diff(example):
    return {"diff": np.array(example["chosen_feature"]) - np.array(example["rejected_feature"])}

all_data = all_data.map(compute_diff)

# 转换为 numpy 数组
diff_matrix = np.stack(all_data["diff"])  # shape: [N, 4096]

# 计算协方差矩阵 A
A = np.cov(diff_matrix, rowvar=False)  # shape: [4096, 4096]
dataset2 = dataset2.map(compute_diff)
# 可选：加一点 regularization 防止奇异
arr_eps = [1.0, 0.1, 0.5, 0.01]
for epsilon in arr_eps:
#epsilon = 1.0
    tmp_A = A + epsilon * np.eye(A.shape[0])

    # 对新测试向量 x 计算 x^T A^{-1} x
    A_inv = np.linalg.inv(tmp_A)

    def compute_mahalanobis_batch(x_list, A_inv):
        return [x.T @ A_inv @ x for x in x_list]
    
    x_test = np.array(dataset2['diff'])
    results = compute_mahalanobis_batch(x_test, A_inv)
    import numpy as np

    # 假设你已经得到了 results 是一个 List[float] 或 np.ndarray of shape (N,)
    np.save("dpo_latent/dpoiter1_mahalanobis_results" + str(epsilon) + '.npy', results)
