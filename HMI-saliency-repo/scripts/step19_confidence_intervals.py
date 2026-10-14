import pandas as pd
import numpy as np
from scipy import stats
# 读取详细结果
df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")
models = ["SR", "Itti", "GBVS"]
metrics = ["AUC", "NSS", "CC"]
def bootstrap_ci(data, n_boot=1000, ci=95):
    """计算 bootstrap 95% 置信区间"""
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    return lower, upper
np.random.seed(42)
print("="*70)
print("Table 1 更新：均值 ± 标准差 与 95% 置信区间")
print("="*70)
print(f"{'Model':<8} {'Metric':<6} {'Mean ± SD':<20} {'95% CI':<20}")
print("-"*70)
results = {}
for model in models:
    sub = df[df["模型"] == model]
    results[model] = {}
    for metric in metrics:
        data = sub[metric].values
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci_low, ci_high = bootstrap_ci(data)
        results[model][metric] = {
            "mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high
        }
        print(f"{model:<8} {metric:<6} {mean:.4f} ± {std:.4f}      [{ci_low:.4f}, {ci_high:.4f}]")
print("\n" + "="*70)
print("Wilcoxon 配对检验效应量（rank-biserial correlation）")
print("="*70)
# 将数据透视为每行一个场景，列为不同模型的指标
pivot_auc = df.pivot_table(index=["场景编号", "图片"], columns="模型", values="AUC")
pivot_nss = df.pivot_table(index=["场景编号", "图片"], columns="模型", values="NSS")
pivot_cc = df.pivot_table(index=["场景编号", "图片"], columns="模型", values="CC")
def rank_biserial(x, y):
    """计算配对 Wilcoxon 检验的效应量 r = Z / sqrt(N) 或基于秩差的 r"""
    # 使用 scipy 的 Wilcoxon 统计量计算 r
    stat, p = stats.wilcoxon(x, y)
    n = len(x)
    # 效应量 r = 1 - 2*W / (n*(n+1)/2) 其中 W 是正秩和
    # 更常用：r = Z / sqrt(N)
    # 计算 Z 值
    # 对于大样本，Z = (W - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24)
    # 这里使用更简单的方法：rank-biserial correlation
    diffs = x - y
    ranks = stats.rankdata(np.abs(diffs))
    pos_ranks = ranks[diffs > 0].sum()
    neg_ranks = ranks[diffs < 0].sum()
    r = (pos_ranks - neg_ranks) / (pos_ranks + neg_ranks)
    return r, p
pairs = [("SR", "Itti"), ("SR", "GBVS"), ("Itti", "GBVS")]
for metric_name, pivot in [("AUC", pivot_auc), ("NSS", pivot_nss), ("CC", pivot_cc)]:
    print(f"\n--- {metric_name} ---")
    for m1, m2 in pairs:
        if m1 in pivot.columns and m2 in pivot.columns:
            r, p = rank_biserial(pivot[m1].values, pivot[m2].values)
            print(f"{m1} vs {m2}: r = {r:.3f}, p = {p:.4f}")
        else:
            print(f"{m1} vs {m2}: 数据缺失")
# 保存结果到CSV
summary_rows = []
for model in models:
    for metric in metrics:
        r = results[model][metric]
        summary_rows.append({
            "Model": model,
            "Metric": metric,
            "Mean": round(r["mean"], 4),
            "SD": round(r["std"], 4),
            "CI_low": round(r["ci_low"], 4),
            "CI_high": round(r["ci_high"], 4)
        })
pd.DataFrame(summary_rows).to_csv(r"C:\Users\sjzhy\Desktop\HMI_study\Table1_with_CI.csv", index=False, encoding="utf-8-sig")
print("\nTable1_with_CI.csv 已保存。")
