# 第14步：生成 Figure 3（典型场景可视化对比）
# 命中场景 S06（main_06 导航信息模块）、未命中场景 S21（media_03 音乐播放区）
# 依据统一口径（SR + 正确权重 NSS>=1.0）：S06 命中(NSS=1.306)、S21 未命中(NSS=0.904)
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径设置 ==========
base = r"C:\Users\sjzhy\Desktop\HMI_study"
images_dir = os.path.join(base, "images")
sr_dir = os.path.join(base, "output")
itti_dir = os.path.join(base, "output_Itti")
gbvs_dir = os.path.join(base, "output_GBVS")
weighted_dir = os.path.join(base, "weighted_output")
scenarios_file = os.path.join(base, "scenarios.csv")
annotations_file = os.path.join(base, "annotations.csv")
# ==============================

# 选择两个场景编号；状态按统一口径标注
scenario_ids = ["S06", "S21"]  # 命中场景、未命中场景
status_map = {"S06": "Hit", "S21": "Miss"}

# 读取 scenarios.csv (UTF-8)
scenarios = {}
with open(scenarios_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scenarios[row["场景编号"]] = {
            "file": row["图片文件名"],
            "target": row["目标区域名称"]
        }

# 读取 annotations.csv (GBK)
ann = {}
with open(annotations_file, "r", encoding="gbk") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["文件名"], row["区域名称"])
        ann[key] = (int(float(row["x1"])), int(float(row["y1"])),
                    int(float(row["x2"])), int(float(row["y2"])))

# 处理每个场景
for sc_id in scenario_ids:
    if sc_id not in scenarios:
        print(f"场景 {sc_id} 未找到")
        continue
    fn = scenarios[sc_id]["file"]
    target_name = scenarios[sc_id]["target"]
    key = (fn, target_name)
    if key not in ann:
        print(f"警告：未找到标注 {key}，跳过场景 {sc_id}")
        continue
    x1, y1, x2, y2 = ann[key]

    # 读取原图
    img_path = os.path.join(images_dir, fn)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取原图 {fn}")
        continue
    h, w = img.shape[:2]

    # 生成任务掩膜
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    mask[y1:y2 + 1, x1:x2 + 1] = 1

    # 读取显著性图
    base_name = os.path.splitext(fn)[0]
    sr = cv2.imread(os.path.join(sr_dir, base_name + "_saliency.jpg"), 0)
    itti = cv2.imread(os.path.join(itti_dir, base_name + "_saliency_itti.jpg"), 0)
    gbvs = cv2.imread(os.path.join(gbvs_dir, base_name + "_saliency_gbvs.jpg"), 0)
    weighted = cv2.imread(os.path.join(weighted_dir, base_name + "_weighted.jpg"), 0)

    # 检查读取
    for name, im in [("SR", sr), ("Itti", itti), ("GBVS", gbvs), ("Weighted", weighted)]:
        if im is None:
            print(f"警告：{fn} 的 {name} 图像未找到")
    if any(im is None for im in [sr, itti, gbvs, weighted]):
        continue

    # 调整尺寸以匹配原图
    if sr.shape != (h, w):
        sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_LINEAR)
    if itti.shape != (h, w):
        itti = cv2.resize(itti, (w, h), interpolation=cv2.INTER_LINEAR)
    if gbvs.shape != (h, w):
        gbvs = cv2.resize(gbvs, (w, h), interpolation=cv2.INTER_LINEAR)
    if weighted.shape != (h, w):
        weighted = cv2.resize(weighted, (w, h), interpolation=cv2.INTER_LINEAR)

    # 绘制 2x3 子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Scenario {sc_id} — {status_map[sc_id]} (target: {target_name})",
                 fontsize=14, fontweight='bold')
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Task mask")
    axes[0, 2].imshow(sr, cmap='hot')
    axes[0, 2].set_title("SR")
    axes[1, 0].imshow(itti, cmap='hot')
    axes[1, 0].set_title("Itti-Koch")
    axes[1, 1].imshow(gbvs, cmap='hot')
    axes[1, 1].set_title("GBVS")
    axes[1, 2].imshow(weighted, cmap='hot')
    axes[1, 2].set_title("SR weighted")
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(base, f"Figure3_{sc_id}.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{output_path}")

print("Figure 3 生成完毕。")
