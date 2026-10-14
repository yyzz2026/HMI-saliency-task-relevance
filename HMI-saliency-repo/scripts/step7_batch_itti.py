import cv2
import os

# ========== 路径设置 ==========
input_folder = r"C:\Users\sjzhy\Desktop\HMI_study\images"
output_folder = r"C:\Users\sjzhy\Desktop\HMI_study\output_Itti"
# ==================================

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

print(f"共找到 {len(image_files)} 张图片，开始批量处理 Itti 显著性检测...\n")

# 创建 Itti 显著性检测器
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

success_count = 0
fail_list = []

for i, filename in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{i}/{len(image_files)}] 读取失败，跳过：{filename}")
        fail_list.append(filename)
        continue

    success, saliency_map = saliency.computeSaliency(img)
    if not success:
        print(f"[{i}/{len(image_files)}] 显著性检测失败，跳过：{filename}")
        fail_list.append(filename)
        continue

    base_name = os.path.splitext(filename)[0]
    output_filename = base_name + "_saliency_itti.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, saliency_map * 255)

    success_count += 1
    print(f"[{i}/{len(image_files)}] 完成：{filename} -> {output_filename}")

print("\n" + "="*50)
print(f"处理结束：成功 {success_count} 张，失败 {len(fail_list)} 张")
if fail_list:
    print("失败的图片：", fail_list)
print("="*50)
