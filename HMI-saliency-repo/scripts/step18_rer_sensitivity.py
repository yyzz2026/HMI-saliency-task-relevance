import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

base = r"C:\Users\sjzhy\Desktop\HMI_study"

# ========== 自动从数据重算三种权重条件的 RER（与 Figure 6 同源） ==========
scen = list(csv.DictReader(open(os.path.join(base,'scenarios.csv'), encoding='utf-8-sig')))
coord = {}
for a in csv.DictReader(open(os.path.join(base,'annotations.csv'), encoding='gbk')):
    coord[(a['文件名'], a['区域名称'])] = (int(float(a['x1'])),int(float(a['y1'])),int(float(a['x2'])),int(float(a['y2'])))

def build_ann(path, enc):
    d = {}
    for a in csv.DictReader(open(path, encoding=enc)):
        d[(a['文件名'], a['区域名称'])] = float(a['权重'])
    return d

w_corr = build_ann(os.path.join(base,'annotations.csv'),'gbk')
w_uni  = build_ann(os.path.join(base,'annotations_B.csv'),'utf-8-sig')   # 均匀 1.0
w_rev  = build_ann(os.path.join(base,'annotations_A.csv'),'utf-8-sig')   # 反向(倒数)

def rer_for(wmap):
    vals = []
    for sc in scen:
        fn = sc['图片文件名']; t = sc['目标区域名称']
        if (fn, t) not in wmap: continue
        x1,y1,x2,y2 = coord[(fn,t)]
        b = os.path.splitext(fn)[0]
        sal = cv2.imread(os.path.join(base,'output', b+'_saliency.jpg'), 0)
        if sal is None: continue
        h, wi = sal.shape
        x1=max(0,min(x1,wi-1)); x2=max(0,min(x2,wi-1)); y1=max(0,min(y1,h-1)); y2=max(0,min(y2,h-1))
        gm = sal.mean()
        if gm == 0: continue
        tm = sal[y1:y2+1, x1:x2+1].mean()
        vals.append(wmap[(fn,t)] * tm / gm)
    return np.array(vals)

a_corr = rer_for(w_corr)
a_uni  = rer_for(w_uni)
a_rev  = rer_for(w_rev)
conditions = ["Correct", "Uniform", "Reversed"]
rer_means = [a_corr.mean(), a_uni.mean(), a_rev.mean()]
rer_std   = [a_corr.std(), a_uni.std(), a_rev.std()]
print("Correct  RER mean={:.3f} std={:.3f} n={}".format(rer_means[0], rer_std[0], len(a_corr)))
print("Uniform  RER mean={:.3f} std={:.3f} n={}".format(rer_means[1], rer_std[1], len(a_uni)))
print("Reversed RER mean={:.3f} std={:.3f} n={}".format(rer_means[2], rer_std[2], len(a_rev)))

# ========== 绘图 ==========
colors = ['#2A9D8F', '#E9C46A', '#E76F51']
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(conditions, rer_means, yerr=rer_std, capsize=8, color=colors, edgecolor='black', width=0.5)
ax.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (RER = 1.0)')
ax.set_ylabel("RER (mean ± SD)", fontsize=12)
ax.set_title("Sensitivity analysis: weight conditions", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
for bar, val in zip(bars, rer_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f'{val:.2f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\Users\sjzhy\Desktop\HMI_study\Figure7_rer_sensitivity.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 已保存：Figure7_rer_sensitivity.jpg")
