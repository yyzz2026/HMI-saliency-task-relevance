import os
import csv
import numpy as np
import cv2
# ========== 路径设置 ==========
base = r"C:\Users\sjzhy\Desktop\HMI_study"
scenarios_file = os.path.join(base, "scenarios.csv")
annotations_file = os.path.join(base, "annotations.csv")
# 三个模型的显著性图文件夹及文件名后缀
model_dirs = {
    "SR": (os.path.join(base, "output"), "_saliency.jpg"),
    "Itti": (os.path.join(base, "output_Itti"), "_saliency_itti.jpg"),
    "GBVS": (os.path.join(base, "output_GBVS"), "_saliency_gbvs.jpg")
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
# 构建标注字典：key = (文件名, 区域名称) -> (x1, y1, x2, y2)
ann_dict = {}
for a in annotations:
    key = (a["文件名"], a["区域名称"])
    ann_dict[key] = (
        int(float(a["x1"])), int(float(a["y1"])),
        int(float(a["x2"])), int(float(a["y2"]))
    )
# 要计算的 K 值（百分比）
K_list = [5, 10, 15]
# 存储结果：model -> K -> list of coverage values per scenario
results = {model: {K: [] for K in K_list} for model in model_dirs}
print("开始计算 Top-K 覆盖率...\n")
for sc in scenarios:
    scene_id = sc["场景编号"]
    filename = sc["图片文件名"]
    target_name = sc["目标区域名称"]
    key = (filename, target_name)
    if key not in ann_dict:
        print(f"警告：未找到标注 {key}，跳过场景 {scene_id}")
        continue
    x1, y1, x2, y2 = ann_dict[key]
    # 读取原图获取尺寸（用任一模型的显著性图也可以，但为了统一，用原图）
    # 这里直接读取 SR 显著性图来确定尺寸，但需注意如果尺寸不一致需要调整
    # 更稳妥：从 images 文件夹读原图，但显著性图通常已与原图同尺寸
    # 我们直接从第一个模型的显著性图读取
    base_name = os.path.splitext(filename)[0]
    first_model = list(model_dirs.keys())[0]
    first_dir, first_suffix = model_dirs[first_model]
    first_path = os.path.join(first_dir, base_name + first_suffix)
    if not os.path.exists(first_path):
        print(f"警告：{first_path} 不存在，跳过场景 {scene_id}")
        continue
    sample_sal = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if sample_sal is None:
        print(f"警告：无法读取 {first_path}，跳过场景 {scene_id}")
        continue
    h, w = sample_sal.shape
    # 生成目标区域掩膜
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    mask[y1:y2+1, x1:x2+1] = 1
    target_pixels = mask.sum()
    if target_pixels == 0:
        print(f"警告：{filename} 目标区域为空，跳过")
        continue
    # 对每个模型计算 Top-K 覆盖
    for model_name, (folder, suffix) in model_dirs.items():
        sal_path = os.path.join(folder, base_name + suffix)
        if not os.path.exists(sal_path):
            print(f"警告：{sal_path} 不存在，跳过 {model_name} - {scene_id}")
            continue
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            print(f"警告：无法读取 {sal_path}，跳过")
            continue
        # 如果尺寸不一致，调整到与掩膜相同尺寸
        if sal.shape != (h, w):
            sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        # 将显著性图展平并排序
        sal_flat = sal.flatten()
        # 获取排序后的索引（从大到小）
        sorted_indices = np.argsort(sal_flat)[::-1]
        total_pixels = len(sal_flat)
        for K in K_list:
            top_k_count = int(total_pixels * K / 100.0)
            if top_k_count == 0:
                top_k_count = 1
            top_k_indices = sorted_indices[:top_k_count]
            # 创建 Top-K 掩膜
            top_k_mask = np.zeros(total_pixels, dtype=np.uint8)
            top_k_mask[top_k_indices] = 1
            top_k_mask = top_k_mask.reshape((h, w))
            # 计算覆盖率：目标区域中被 Top-K 覆盖的像素比例
            covered = np.logical_and(mask, top_k_mask).sum()
            coverage = covered / target_pixels
            results[model_name][K].append(coverage)
# 输出结果
print("\n" + "="*60)
print("Top-K 覆盖率（目标区域中被 Top-K 像素覆盖的平均比例）")
print("="*60)
print(f"{'Model':<8} {'K=5%':<12} {'K=10%':<12} {'K=15%':<12}")
print("-"*60)
for model_name in model_dirs.keys():
    row = f"{model_name:<8}"
    for K in K_list:
        if results[model_name][K]:
            mean_cov = np.mean(results[model_name][K]) * 100
            row += f" {mean_cov:>6.2f}%     "
        else:
            row += f" {'N/A':>7}     "
    print(row)
# 保存详细结果到 CSV
output_csv = os.path.join(base, "topk_coverage_results.csv")
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["模型", "K值", "平均覆盖率(%)", "标准差(%)"])
    for model_name in model_dirs.keys():
        for K in K_list:
            vals = results[model_name][K]
            if vals:
                mean = np.mean(vals) * 100
                std = np.std(vals, ddof=1) * 100
                writer.writerow([model_name, f"{K}%", f"{mean:.2f}", f"{std:.2f}"])
            else:
                writer.writerow([model_name, f"{K}%", "N/A", "N/A"])
print(f"\n详细结果已保存到：{output_csv}")
