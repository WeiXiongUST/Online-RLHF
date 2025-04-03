import numpy as np
a = np.load("dpo_latent/sftmodel_mahalanobis_results0.01.npy")
b = np.load("dpo_latent/dpoiter1_mahalanobis_results0.01.npy")

a = np.sqrt(a)
b = np.sqrt(b)
print("SFT Mean:", np.mean(a), "Std:", np.std(a))
print("Well-trained Mean:", np.mean(b), "Std:", np.std(b))
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

corr, pval = pearsonr(a, b)
print(f"Pearson correlation: {corr:.4f}, p-value: {pval:.4e}")


import matplotlib.pyplot as plt
import seaborn as sns

sns.regplot(x=a, y=b, scatter_kws={"alpha":0.5}, line_kws={"color": "red"})
plt.xlabel("SFT Mahalanobis Distance")
plt.ylabel("Well-trained Mahalanobis Distance")
plt.title("Mahalanobis Distance Correlation Lambda 0.01")
plt.grid(True)
#plt.show()
plt.savefig("dpo_latent/sft_dpoiter1_mahalanbis_dis_corr_lambda0_01")

from scipy.stats import spearmanr

rank_corr, _ = spearmanr(a, b)
print(f"Spearman rank correlation: {rank_corr:.4f}")
