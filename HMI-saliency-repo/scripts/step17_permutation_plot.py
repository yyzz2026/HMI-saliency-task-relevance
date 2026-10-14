import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径设置 ==========
base = r"C:\Users\sjzhy\Desktop\HMI_study"
saliency_dir = os.path.join(base, "output")          # SR 显著性图
scenarios_file = os.path.join(base, "scenarios.csv")
annotations_file = os.path.join(base, "annotations.csv")
output_fig = os.path.join(base, "Figure6_permutation.jpg")
# ==============================

# 读取 scenarios.csv (UTF-8)
scenarios = []
with open(scenarios_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scenarios.append(row)

# 读取 annotations.csv (GBK)
annotations = []
with open(annotations_file, "r", encoding="gbk") as f:
    reader = csv.DictReader(f)
    for row in reader:
        annotations.append(row)

# 构建字典：key = (文件名, 区域名称) -> (x1, y1, x2, y2, weight)
ann_dict = {}
# 同时构建：文件名 -> 该图片所有区域的权重列表
weights_by_file = {}
for a in annotations:
    fn = a["文件名"]
    key = (fn, a["区域名称"])
    x1 = int(float(a["x1"]))
    y1 = int(float(a["y1"]))
    x2 = int(float(a["x2"]))
    y2 = int(float(a["y2"]))
    w = float(a["权重"])
    ann_dict[key] = (x1, y1, x2, y2, w)
    weights_by_file.setdefault(fn, []).append(w)

# 对于每个场景，计算 global_mean 和 target_mean
# 结果存储：list of dict {file, target, global_mean, target_mean, correct_weight}
scene_data = []
print("正在读取显著性图并计算基础均值...")
for sc in scenarios:
    fn = sc["图片文件名"]
    target_name = sc["目标区域名称"]
    key = (fn, target_name)
    if key not in ann_dict:
        print(f"警告：未找到标注 {key}，跳过")
        continue
    x1, y1, x2, y2, w_correct = ann_dict[key]
    base_name = os.path.splitext(fn)[0]
    sal_path = os.path.join(saliency_dir, base_name + "_saliency.jpg")
    sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
    if sal is None:
        print(f"无法读取 {sal_path}，跳过")
        continue
    h, w_img = sal.shape
    x1 = max(0, min(x1, w_img-1))
    x2 = max(0, min(x2, w_img-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    global_mean = sal.mean()
    target_mean = sal[y1:y2+1, x1:x2+1].mean()
    if global_mean == 0:
        continue
    scene_data.append({
        "file": fn,
        "target": target_name,
        "global_mean": global_mean,
        "target_mean": target_mean,
        "correct_weight": w_correct
    })
print(f"有效场景数：{len(scene_data)}")

# 计算正确配对下的 RER
rer_correct_list = []
for sd in scene_data:
    rer = sd["correct_weight"] * sd["target_mean"] / sd["global_mean"]
    rer_correct_list.append(rer)
observed_mean_rer = np.mean(rer_correct_list)
print(f"正确配对的平均 RER = {observed_mean_rer:.4f}")

# 置换检验
n_perm = 1000
null_means = []
np.random.seed(42)
for _ in range(n_perm):
    perm_rer = []
    for sd in scene_data:
        fn = sd["file"]
        # 从该图片的所有区域权重中随机抽取一个
        w_list = weights_by_file[fn]
        w_rand = np.random.choice(w_list)
        rer = w_rand * sd["target_mean"] / sd["global_mean"]
        perm_rer.append(rer)
    null_means.append(np.mean(perm_rer))
null_means = np.array(null_means)

# 计算 p 值：随机均值 >= 观测均值的比例
p_value = np.mean(null_means >= observed_mean_rer)
print(f"置换检验 p 值 = {p_value:.4f}")
print(f"Null 分布 95th 百分位 = {np.percentile(null_means, 95):.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(null_means, bins=30, color='#8ECAE6', edgecolor='black', alpha=0.7, label='Null distribution (1000 permutations)')
ax.axvline(observed_mean_rer, color='red', linestyle='--', linewidth=2, label=f'Observed RER = {observed_mean_rer:.2f}')
ax.axvline(np.percentile(null_means, 95), color='gray', linestyle=':', linewidth=2, label='95th percentile')
ax.set_xlabel("Mean RER", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Permutation test: correct pairing vs. random pairing", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 6 已保存：{output_fig}")

# 可选：保存 null 分布数据
np.savetxt(os.path.join(base, "permutation_null.csv"), null_means, delimiter=",", header="RER", comments="")
print("Null 分布已保存到 permutation_null.csv")
