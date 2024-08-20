import json  
import os  
import cv2  
import cobb_evaluate
import numpy as np
import pre_proc
import random
import string
from pre_proc import rearrange_pts
import math
from pycocotools.coco import COCO
# 读取JSON文件  
def euclidean_distance(x1, y1, x2, y2):  
    """  
    计算并返回两点之间的欧氏距离。  
      
    参数:  
    x1, y1 -- 第一个点的坐标  
    x2, y2 -- 第二个点的坐标  
      
    返回:  
    distance -- 两点之间的欧氏距离  
    """  
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  
    return distance  

def generate_unique_random_string():
    # 创建一个集合来存储已经生成的字符串
    #generated_strings = set()
    
    while True:
        # 生成一个随机长度（1-10）
        length = random.randint(1, 20)
        
        # 生成一个由随机字母和数字组成的字符串
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        
        # 如果生成的字符串不在集合中，则添加到集合并返回
        # if random_string not in generated_strings:
        #     generated_strings.add(random_string)
        return random_string
        
def is_point_in_rectangle(x, y, w, h, x1, y1):
    # 计算矩形框的右下角坐标
    right = x + w
    bottom = y + h

    # 判断点是否在矩形框内
    if x1 >= x and x1 <= right and y1 >= y and y1 <= bottom:
        return True
    else:
        print("错误坐标：",x1,y1)
        return False  

image_dir = 'spinal-AI2024-advanced/spinal-AI2024-advance-rest'
output_file = 'spinal-AI2024-advanced/spinal_AI2024_advance_subset6_stage0_best.json'#保存标签路径
output_file_txt= 'spinal-AI2024-advanced/spinal_AI2024_advance_subset6_stage0_discard.txt'#异常图片路径
with open('spinal_AI2024_advance_subset6_stage0.json', 'r', encoding='utf-8') as f:  #lra结果
    data = json.load(f)  
with open('spinal-AI2024-advanced/spinal-AI2024-rest.json' , 'r') as f:  #空壳标签
    original_annotation = json.load(f)  
add=20000
# 创建一个字典来按image_id分组polys的特定坐标  
grouped_polys = {}  
coco_dict = {
    'images': [],
    'annotations': [],
    'categories': []
}  
coco_dict['categories'].append({
    'supercategory': 'text',
    'id': 1,
    'name': 'text'
})

# 遍历数据  
for item in data: 
    # if item['category_id'] ==0:
    #     continue
    image_id = item['image_id']  
    polys = item['polys']  
   
    
    # print(score)
    # haha
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
    # print(image_id,coords_list)
    # haha  
    # 遍历coords_list中的每个多边形坐标  
    for poly in coords_list:  
        # 将多边形的每个点添加到当前image_id的坐标列表中  
        formatted_coords[image_id].extend(poly)  
  
# formatted_coords现在是一个字典，其中键是image_id，值是包含所有坐标点的列表（N*2格式）  
# 打印以验证结果（可选）  

for image_id, coords in formatted_coords.items():  
    # print(image_id, coords)
    # haha
    #file_name = f"{image_id+701:06d}.png"  
    file_name = original_annotation['images'][image_id]['file_name']  
    height=original_annotation['images'][image_id]['height'] 
    width=original_annotation['images'][image_id]['width'] 
    #print(file_name,height,width)
    
    #print(file_name)
    
    # print(file_name)
    # haha
    #print(len(coords))
    real_landmarks=coords
    #print(real_landmarks)
    #print(len(real_landmarks))
    landmarks_array = np.array(real_landmarks)  
    real_landmarks=rearrange_pts(landmarks_array)
    #print(real_landmarks)
    real_landmarks = real_landmarks.tolist()  
    
    
    coco_dict['images'].append({
        'id': len(coco_dict['images'])+1+add,  # 使用文件名作为图片ID
        'file_name': file_name,
        'height': height,
        'width': width,      
    })
    chenchenhaha=1
    current_average_distance= []
    before_center_x=-1
    before_center_y=-1
    pre_samples_length=0
    for group in range(0, len(real_landmarks),4):
        #print(file_name,group)
        
            
        chenchenhaha=1
        # 提取当前组的点坐标
        points = real_landmarks[group:group+4]
        
        polygon = [points[0], points[1], points[2], points[3]]
        polygon=sum(polygon,[])
        polygon = [round(x) for x in polygon]
        # 计算边界框
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        # 计算边界框的左上角和右下角坐标
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        # 计算边界框的宽度和高度
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2  
        center_y = y_min + height / 2 
        index_center=int(group/4)
        #print(index_center)
        
        if index_center>1:
            
            current_distance=euclidean_distance(center_x,center_y,before_center_x,before_center_y)
            #print(current_average_distance)
            
            
            before_center_x=center_x
            before_center_y=center_y
            if 3*sum(current_average_distance)/len(current_average_distance)<current_distance:    
                #print(current_distance)
                line = f"{image_id+1+add},{index_center}\n"  
                with open(output_file_txt, 'a', encoding='utf-8') as f:  
                    f.write(line)  
                continue
            current_average_distance.append(current_distance)
            
        elif index_center == 1 :
            current_distance=euclidean_distance(center_x,center_y,before_center_x,before_center_y)
            current_average_distance.append(current_distance)
            before_center_x = center_x
            before_center_y = center_y
            
        elif index_center == 0 :
            before_center_x = center_x
            before_center_y = center_y
            
        #print(int(x_min),int( y_min),round(width+1),round( height+1))
        #print(polygon)
        if round(width * height,2)<200:
            continue
        for j in range(4):  
            x1=polygon[j*2]
            y1=polygon[j*2+1]
            if is_point_in_rectangle(int(x_min), int( y_min), round( width+1),round( height+1), x1, y1):
                #print("点在矩形框内")
                haha=[]
            else: 
                chenchenhaha=0
                break

        if chenchenhaha==0:        
            continue
        
        # 将边界框信息添加到annotations字段中
        coco_dict['annotations'].append({
            'id': len(coco_dict['annotations'])+1,  # 使用唯一的标注ID
            'image_id': image_id+1+add,  # 图片ID
            'category_id': 1,  # 假设所有对象都属于第一个类别
            'bbox': [int(x_min), int( y_min), round( width+1),round( height+1)],  # 边界框信息
            'segmentation': [polygon],  # 分割信息留空
            'area': round(width * height,2),  # 计算面积
            'iscrowd': 0 , # 不是群体对象
            #'transcription': str(i)+"-"+str(result_type)+"-"+str(len(coco_dict['annotations'])+1),
            'transcription':generate_unique_random_string(),
            'score': 1.0  # 假设所有标注的置信度为1.0,
        })
        pre_samples_length=pre_samples_length+1
    if pre_samples_length<=10:
        line = f"{image_id+1+add},{-1}\n"  
        with open(output_file_txt, 'a', encoding='utf-8') as f:  
            f.write(line)  
    
coco = COCO()
coco.dataset=coco_dict
coco.createIndex()
with open(output_file, 'w') as json_file:
    json.dump(coco.dataset, json_file, indent=4)  # 使用缩进使JSON文件更易读

print('COCO format annotation file has been saved.')
# file_path = os.path.join(image_dir, file_name)  
# image = cv2.imread(file_path) 
   
    
# coords = np.array(coords)
# coords=pre_proc.rearrange_pts(coords)
# result,type=cobb_evaluate.cobb_angle_calc(coords,image)
# line = f"{file_name},{result[1]},{result[3]},{result[5]}\n"  
    
    
   
  
#     # 打开文件（如果文件很大，考虑使用'a'模式在文件末尾追加）  
# with open(output_file, 'a', encoding='utf-8') as f:  
#     f.write(line)  
#     #print()