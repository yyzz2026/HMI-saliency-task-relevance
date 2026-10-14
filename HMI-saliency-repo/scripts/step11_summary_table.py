import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")

models = ["SR", "Itti", "GBVS"]
print("Table 3: Model Performance Summary (mean ± SD)\n")
print(f"{'Model':<8} {'AUC':<20} {'NSS':<20} {'CC':<20} {'Hit Rate':<10}")
print("-" * 80)

for model in models:
    sub = df[df["模型"] == model]
    auc_mean, auc_std = sub["AUC"].mean(), sub["AUC"].std()
    nss_mean, nss_std = sub["NSS"].mean(), sub["NSS"].std()
    cc_mean, cc_std = sub["CC"].mean(), sub["CC"].std()
    hit_rate = sub["命中"].mean() * 100
    print(f"{model:<8} {auc_mean:.4f} ± {auc_std:.4f}    {nss_mean:.4f} ± {nss_std:.4f}    {cc_mean:.4f} ± {cc_std:.4f}    {hit_rate:.1f}%")

# 输出到 CSV 供论文使用
summary = []
for model in models:
    sub = df[df["模型"] == model]
    summary.append({
        "Model": model,
        "AUC_mean": round(sub["AUC"].mean(), 4),
        "AUC_std": round(sub["AUC"].std(), 4),
        "NSS_mean": round(sub["NSS"].mean(), 4),
        "NSS_std": round(sub["NSS"].std(), 4),
        "CC_mean": round(sub["CC"].mean(), 4),
        "CC_std": round(sub["CC"].std(), 4),
        "HitRate": round(sub["命中"].mean() * 100, 1)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(r"C:\Users\sjzhy\Desktop\HMI_study\Table3_model_summary.csv", index=False, encoding="utf-8-sig")
print("\n已保存到 Table3_model_summary.csv")
