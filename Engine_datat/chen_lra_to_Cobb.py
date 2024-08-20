import json  
import os  
import cv2  
import cobb_evaluate
import numpy as np
import pre_proc
# 读取JSON文件  
image_dir = 'spinal-AI2024/spinal-AI2024-subset5'
output_file = 'Cobb_spinal-AI2024-test_15_01_lra_result.txt'
with open('spinal_AI2024_test_15_01.json', 'r', encoding='utf-8') as f:  
    data = json.load(f)  
with open('spinal-AI2024/spinal-AI2024-subset5.json' , 'r') as f:  
    original_annotation = json.load(f)  
# 创建一个字典来按image_id分组polys的特定坐标  
grouped_polys = {}  
  
# 遍历数据  
for item in data:  
    image_id = item['image_id']  
    polys = item['polys']  
      
    # 如果image_id不在分组字典中，则初始化一个空列表  
    if image_id not in grouped_polys:  
        grouped_polys[image_id] = []  
      
    # 提取polys中的第1组、第7组、第8组和最后一组坐标  
    selected_coords = [  
        polys[0],  # 第1组坐标  
        polys[6] if len(polys) > 6 else None,  # 第7组坐标（确保索引不越界）  
        polys[7] if len(polys) > 7 else None,  # 第8组坐标  
        polys[-1]  # 最后一组坐标  
    ]  
      
    # 过滤掉None值（如果polys长度不足）  
    selected_coords = [coord for coord in selected_coords if coord is not None]  
      
    # 将选取的坐标添加到当前image_id的列表中  
    grouped_polys[image_id].append(selected_coords)  
  
# 打印分组后的数据（可选）  
# for image_id, coords_list in grouped_polys.items():  
#     print(image_id,coords_list)
#formatted_coords = []  
# 创建一个新的字典来存储每个image_id对应的坐标  
formatted_coords = {}  
  
# 遍历grouped_polys中的每个image_id和对应的coords_list  
for image_id, coords_list in grouped_polys.items():  
    # 初始化当前image_id的坐标列表  
    formatted_coords[image_id] = []  
      
    # 遍历coords_list中的每个多边形坐标  
    for poly in coords_list:  
        # 将多边形的每个点添加到当前image_id的坐标列表中  
        formatted_coords[image_id].extend(poly)  
  
# formatted_coords现在是一个字典，其中键是image_id，值是包含所有坐标点的列表（N*2格式）  
# 打印以验证结果（可选）  
for image_id, coords in formatted_coords.items():  
    
    #file_name = f"{image_id+701:06d}.png"  
    file_name = original_annotation['images'][image_id]['file_name']  
    # print(file_name)
    # haha
    file_path = os.path.join(image_dir, file_name)  
    image = cv2.imread(file_path) 
   
    
    coords = np.array(coords)
    coords=pre_proc.rearrange_pts(coords)
    print(coords)
    haha
    result,type=cobb_evaluate.cobb_angle_calc(coords,image)
    line = f"{file_name},{result[1]},{result[3]},{result[5]}\n"  
    
    
   
  
    # 打开文件（如果文件很大，考虑使用'a'模式在文件末尾追加）  
    with open(output_file, 'a', encoding='utf-8') as f:  
        f.write(line)  
    #print()