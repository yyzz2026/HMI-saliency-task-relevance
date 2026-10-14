import os
import csv
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# ========== 路径设置 ==========
base_dir = r"C:\Users\sjzhy\Desktop\HMI_study"
images_dir = os.path.join(base_dir, "images")
annotations_file = os.path.join(base_dir, "annotations.csv")
scenarios_file = os.path.join(base_dir, "scenarios.csv")

# 三个模型的显著性图文件夹及文件名后缀
model_dirs = {
    "SR": os.path.join(base_dir, "output"),
    "Itti": os.path.join(base_dir, "output_Itti"),
    "GBVS": os.path.join(base_dir, "output_GBVS")
}
model_suffix = {
    "SR": "_saliency.jpg",
    "Itti": "_saliency_itti.jpg",
    "GBVS": "_saliency_gbvs.jpg"
}
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

# 构建标注字典
ann_dict = {}
for ann in annotations:
    key = (ann["文件名"], ann["区域名称"])
    ann_dict[key] = {
        "x1": int(float(ann["x1"])),
        "y1": int(float(ann["y1"])),
        "x2": int(float(ann["x2"])),
        "y2": int(float(ann["y2"]))
    }

# 存储结果
results = []  # 每行: 场景编号, 图片, 目标区域, 模型, AUC, NSS, CC, 命中

for sc in scenarios:
    scene_id = sc["场景编号"]
    filename = sc["图片文件名"]
    target_region = sc["目标区域名称"]
    
    key = (filename, target_region)
    if key not in ann_dict:
        print(f"警告：未找到标注 {key}，跳过场景 {scene_id}")
        continue
    ann = ann_dict[key]
    x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
    
    # 读取原始图像获取尺寸
    img_path = os.path.join(images_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取原图：{filename}，跳过")
        continue
    h, w = img.shape[:2]
    
    # 生成任务掩膜（目标区域为1）
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    mask[y1:y2+1, x1:x2+1] = 1
    mask_flat = mask.flatten()
    mask_sum = mask.sum()
    if mask_sum == 0:
        print(f"掩膜为空：{filename}，跳过")
        continue
    
    for model_name, folder in model_dirs.items():
        base_name = os.path.splitext(filename)[0]
        sal_path = os.path.join(folder, base_name + model_suffix[model_name])
        if not os.path.exists(sal_path):
            print(f"缺失显著性图：{sal_path}，跳过")
            continue
        
        # 读取显著性图并调整尺寸
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            print(f"无法读取显著性图：{sal_path}，跳过")
            continue
        if sal.shape != (h, w):
            sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        
        sal_float = sal.astype(np.float32)
        sal_flat = sal_float.flatten()
        
        # AUC
        try:
            auc = roc_auc_score(mask_flat, sal_flat)
        except ValueError:
            auc = 0.5
        
        # NSS: 归一化显著性图，计算目标区域均值
        sal_mean = sal_float.mean()
        sal_std = sal_float.std()
        if sal_std > 0:
            sal_norm = (sal_float - sal_mean) / sal_std
        else:
            sal_norm = sal_float - sal_mean
        nss = sal_norm[mask == 1].mean()
        
        # CC
        if sal_flat.std() > 0 and mask_flat.std() > 0:
            cc, _ = pearsonr(sal_flat, mask_flat)
        else:
            cc = 0.0
        
        # 命中：目标区域均值 > 全图均值 + 1 SD
        threshold = sal_mean + sal_std
        target_mean = sal_float[mask == 1].mean()
        hit = 1 if target_mean > threshold else 0
        
        results.append({
            "场景编号": scene_id,
            "图片": filename,
            "目标区域": target_region,
            "模型": model_name,
            "AUC": round(auc, 4),
            "NSS": round(nss, 4),
            "CC": round(cc, 4),
            "命中": hit
        })

# 输出详细结果到 CSV
output_csv = os.path.join(base_dir, "model_metrics_detailed.csv")
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["场景编号","图片","目标区域","模型","AUC","NSS","CC","命中"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# 汇总统计
print("\n" + "="*60)
print("多模型评估汇总")
print("="*60)

for model in model_dirs.keys():
    model_rows = [r for r in results if r["模型"] == model]
    if not model_rows:
        continue
    auc_mean = np.mean([r["AUC"] for r in model_rows])
    nss_mean = np.mean([r["NSS"] for r in model_rows])
    cc_mean = np.mean([r["CC"] for r in model_rows])
    hit_rate = np.mean([r["命中"] for r in model_rows]) * 100
    print(f"{model:5s} | AUC: {auc_mean:.4f} | NSS: {nss_mean:.4f} | CC: {cc_mean:.4f} | 命中率: {hit_rate:.1f}%")

print("\n详细结果已保存到: " + output_csv)
