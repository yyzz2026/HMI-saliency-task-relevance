import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 自动从 CSV 统计（替代硬编码，数据更新后可重算） ==========
df = pd.read_csv(r"C:\Users\sjzhy\Desktop\HMI_study\model_metrics_detailed.csv", encoding="utf-8-sig")
thresholds = ["NSS ≥ 0.5\n(lenient)", "NSS ≥ 1.0\n(adopted)", "NSS ≥ 1.5\n(strict)"]
thr_vals = [0.5, 1.0, 1.5]

def nss_hit_rates(model):
    sub = df[df["模型"] == model]
    n = len(sub)
    return [100.0 * (sub["NSS"] >= t).sum() / n for t in thr_vals]

sr = nss_hit_rates("SR")
itti = nss_hit_rates("Itti")
gbvs = nss_hit_rates("GBVS")

print("Auto-computed hit rates (NSS>=thr scenario share):")
print("  SR  ", [f"{v:.1f}%" for v in sr])
print("  Itti", [f"{v:.1f}%" for v in itti])
print("  GBVS", [f"{v:.1f}%" for v in gbvs])

x = np.arange(len(thresholds))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width, sr, width, label='SR', color='#8ECAE6', edgecolor='black')
bars2 = ax.bar(x, itti, width, label='Itti-Koch', color='#E63946', edgecolor='black')
bars3 = ax.bar(x + width, gbvs, width, label='GBVS', color='#2A9D8F', edgecolor='black')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.axhline(y=85, color='red', linestyle='--', linewidth=1.5, label='85% screening threshold')
ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.set_ylabel("Hit rate (%)", fontsize=12)
ax.set_title("Hit rate under different NSS thresholds", fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(r"C:\Users\sjzhy\Desktop\HMI_study\Figure5_threshold.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 已保存：Figure5_threshold.jpg")
