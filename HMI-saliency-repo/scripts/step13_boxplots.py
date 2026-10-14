# 第13步：生成论文用箱线图（AUC、NSS、CC 分布）
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")

models = ["SR", "Itti", "GBVS"]
metrics = ["AUC", "NSS", "CC"]
titles = ["AUC", "NSS", "CC"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric, title in zip(axes, metrics, titles):
    data = [df[df["模型"] == m][metric].values for m in models]
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True, widths=0.5)
    colors = ['#8ECAE6', '#E63946', '#2A9D8F']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(title, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.axhline(y=0.5 if metric == 'AUC' else 0, color='gray', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(r"C:\Users\sjzhy\Desktop\HMI_study\Figure2_boxplots.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 已保存：Figure2_boxplots.jpg")
