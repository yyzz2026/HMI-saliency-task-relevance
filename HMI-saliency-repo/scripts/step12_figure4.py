import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")

models = ["SR", "Itti", "GBVS"]
auc_means = [df[df["模型"] == m]["AUC"].mean() for m in models]
hit_rates = [df[df["模型"] == m]["命中"].mean() * 100 for m in models]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AUC 对比
axes[0].bar(models, auc_means, color=['#8ECAE6', '#E63946', '#2A9D8F'], edgecolor='black', width=0.5)
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Random (AUC=0.5)')
axes[0].set_ylabel('AUC', fontsize=12)
axes[0].set_title('AUC by Model', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1.0)
axes[0].legend()
for i, v in enumerate(auc_means):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

# 命中率对比
axes[1].bar(models, hit_rates, color=['#8ECAE6', '#E63946', '#2A9D8F'], edgecolor='black', width=0.5)
axes[1].axhline(y=85, color='red', linestyle='--', label='Acceptable threshold (85%)')
axes[1].set_ylabel('Hit Rate (%)', fontsize=12)
axes[1].set_title('Hit Rate by Model', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].legend()
for i, v in enumerate(hit_rates):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(r"C:\Users\sjzhy\Desktop\HMI_study\Figure4_model_comparison.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 已保存：Figure4_model_comparison.jpg")
