# 批量显著性检测脚本
# 功能：遍历images文件夹中的所有图片，逐一运行谱残差法显著性检测，保存结果到output文件夹

import cv2
import os

# ========== 设置：修改这里的路径 ==========
# 输入图片文件夹的完整路径（你的images文件夹）
input_folder = r"C:\Users\sjzhy\Desktop\HMI_study\images"

# 输出文件夹的完整路径（你的output文件夹）
output_folder = r"C:\Users\sjzhy\Desktop\HMI_study\output"
# ========================================

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有图片文件的文件名
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

print(f"共找到 {len(image_files)} 张图片，开始批量处理...\n")

# 创建显著性检测器（只需创建一次，可重复使用）
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

success_count = 0
fail_list = []

for i, filename in enumerate(image_files, start=1):
    # 拼接完整路径
    img_path = os.path.join(input_folder, filename)
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{i}/{len(image_files)}] 读取失败，跳过：{filename}")
        fail_list.append(filename)
        continue
    
    # 运行显著性检测
    success, saliency_map = saliency.computeSaliency(img)
    if not success:
        print(f"[{i}/{len(image_files)}] 显著性检测失败，跳过：{filename}")
        fail_list.append(filename)
        continue
    
    # 保存输出图
    # 去掉原扩展名，统一加 _saliency.jpg
    base_name = os.path.splitext(filename)[0]
    output_filename = base_name + "_saliency.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, saliency_map * 255)  # 数值从0-1转换到0-255
    
    success_count += 1
    print(f"[{i}/{len(image_files)}] 完成：{filename} -> {output_filename}")

print("\n" + "="*50)
print(f"处理结束：成功 {success_count} 张，失败 {len(fail_list)} 张")
if fail_list:
    print("失败的图片：", fail_list)
print("="*50)
