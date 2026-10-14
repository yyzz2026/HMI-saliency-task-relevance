import os
import cv2
import numpy as np
import pysaliency

# ========== 路径设置 ==========
input_folder = r"C:\Users\sjzhy\Desktop\HMI_study\images"
output_folder = r"C:\Users\sjzhy\Desktop\HMI_study\output_GBVS"
model_location = r"C:\Users\sjzhy\Desktop\HMI_study\models"
# ==============================

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(model_location):
    os.makedirs(model_location)

# 初始化 GBVS 模型
print("正在初始化 GBVS 模型，首次运行可能需要下载模型文件...")
try:
    gbvs = pysaliency.GBVSIttiKoch(location=model_location)
    print("GBVS 模型初始化成功。")
except Exception as e:
    print(f"GBVS 模型初始化失败：{e}")
    print("请检查网络连接或手动下载模型文件。")
    exit(1)

# 获取所有图片
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

print(f"共找到 {len(image_files)} 张图片，开始批量处理 GBVS...\n")

success_count = 0
fail_list = []

for i, filename in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{i}/{len(image_files)}] 读取失败，跳过：{filename}")
        fail_list.append(filename)
        continue

    # 转为 RGB 并归一化到 0-1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    try:
        # 计算 GBVS 显著性图
        saliency_map = gbvs.saliency_map(img_rgb)
    except Exception as e:
        print(f"[{i}/{len(image_files)}] GBVS 计算失败：{filename}，错误：{e}")
        fail_list.append(filename)
        continue

    # 归一化到 0-255
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
    saliency_map = (saliency_map * 255).astype(np.uint8)

    base_name = os.path.splitext(filename)[0]
    output_filename = base_name + "_saliency_gbvs.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, saliency_map)

    success_count += 1
    print(f"[{i}/{len(image_files)}] 完成：{filename} -> {output_filename}")

print("\n" + "="*50)
print(f"处理结束：成功 {success_count} 张，失败 {len(fail_list)} 张")
if fail_list:
    print("失败的图片：", fail_list)
print("="*50)
