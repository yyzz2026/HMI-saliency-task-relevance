import pandas as pd
from scipy.stats import wilcoxon

# 读取详细结果
df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")

# 将数据透视为：每行一个场景，列为不同模型的 AUC
pivot_auc = df.pivot_table(index=["场景编号", "图片"], columns="模型", values="AUC")

models = ["SR", "Itti", "GBVS"]
print("模型间 AUC 配对比较（Wilcoxon 符号秩检验）\n")

for i in range(len(models)):
    for j in range(i+1, len(models)):
        m1, m2 = models[i], models[j]
        if m1 in pivot_auc.columns and m2 in pivot_auc.columns:
            stat, p = wilcoxon(pivot_auc[m1], pivot_auc[m2])
            print(f"{m1} vs {m2}: 统计量={stat:.2f}, p值={p:.4f}")
        else:
            print(f"{m1} 或 {m2} 数据缺失")

# 对 NSS、CC、命中率也可做类似检验，此处略。
