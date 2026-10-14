import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")

# 根据文件名判断类别
def get_category(filename):
    if "main" in filename: return "Main menu"
    elif "nav" in filename: return "Navigation"
    elif "climate" in filename: return "Climate"
    elif "media" in filename: return "Media"
    elif "settings" in filename: return "Settings"
    return "Unknown"

df["类别"] = df["图片"].apply(get_category)

models = ["SR", "Itti", "GBVS"]
categories = ["Main menu", "Navigation", "Climate", "Media", "Settings"]

auc_means = {m: [df[(df["模型"]==m) & (df["类别"]==c)]["AUC"].mean() for c in categories] for m in models}
hit_means = {m: [df[(df["模型"]==m) & (df["类别"]==c)]["命中"].mean()*100 for c in categories] for m in models}

x = np.arange(len(categories))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, m in enumerate(models):
    axes[0].bar(x + i*width, auc_means[m], width, label=m)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(categories, rotation=15)
axes[0].set_ylabel("Mean AUC")
axes[0].set_title("AUC by category")
axes[0].legend()
axes[0].set_ylim(0, 1)

for i, m in enumerate(models):
    axes[1].bar(x + i*width, hit_means[m], width, label=m)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(categories, rotation=15)
axes[1].set_ylabel("Hit rate (%)")
axes[1].set_title("Hit rate by category")
axes[1].legend()
axes[1].set_ylim(0, 100)
axes[1].axhline(y=85, color='red', linestyle='--', label='85% threshold')

plt.tight_layout()
plt.savefig(r"C:\Users\sjzhy\Desktop\HMI_study\Figure4_category_bars.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 已保存：Figure4_category_bars.jpg")
