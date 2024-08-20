import json  
from PIL import Image, ImageDraw, ImageFont  
import os  

# 定义一组固定且好看的颜色  
colors = [  
    (255, 99, 71),  # 亮红色  
    (33, 150, 243), # 亮蓝色  
    (255, 206, 86), # 亮黄色  
    (76, 175, 80),  # 亮绿色  
    (255, 127, 0),  # 橙色  
    (153, 102, 255),# 亮紫色  
    (238, 130, 238),# 淡紫色  
    (128, 0, 128),  # 深紫色  
    (0, 128, 0),    # 绿色  
    (0, 0, 128),    # 海军蓝  
    (255, 0, 0),    # 红色  
    (0, 255, 0),    # 鲜绿色  
    (0, 0, 255),    # 蓝色  
    (255, 255, 0),  # 黄色  
    (128, 128, 0),  # 橄榄绿  
    (192, 192, 192),# 银白色  
    (128, 128, 128),# 灰色  
    (255, 165, 0),  # 金色  
    (255, 255, 224),# 浅黄色  
    (0, 255, 255)   # 青色  
]  
# 设定文件路径  
json_path = 'my_text_results_AASEC2019.json'  #lra格式
image_folder = 'AASEC2019/test'  #原图文件夹
#image_folder="/mnt/data/experiments/LRANet/images_show/chenchen_spinal_AI2024_subset6_stage1_best"
output_folder = 'AASEC2019/0813'  #结果保存文件夹

  
# 确保输出文件夹存在  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  
  
# 读取JSON文件  
with open(json_path, 'r') as f:  
    annotations = json.load(f)  

with open('AASEC2019/AASEC2019_test17_fake.json' , 'r') as f:  #空壳标注
    original_annotation = json.load(f)  

# 用于收集相同image_id的标注  
annotations_by_image_id = {}  
# 遍历标注，按image_id分组  
for annotation in annotations:  
    image_id = annotation['image_id']  
    if image_id not in annotations_by_image_id:  
        annotations_by_image_id[image_id] = []  
    annotations_by_image_id[image_id].append(annotation)  
for image_id, image_annotations in annotations_by_image_id.items():  
    
    
# 遍历标注  
# for annotation in annotations:  
#     image_id = annotation['image_id']  
    #print(image_id,original_annotation['images'][image_id]['file_name'])
    
    # 构建输出文件名
    # 构建图片文件名（这里假设image_id与文件名是对应的）  
    #image_filename = f'0{image_id + 701:05d}.png'  # 转换image_id到对应的文件名格式，并添加前导零 
    image_filename = original_annotation['images'][image_id]['file_name']  # 转换image_id到对应的文件名格式，并添加前导零
    image_path = os.path.join(image_folder, image_filename)  
    #print(image_path)
    #haha
    # 检查图片文件是否存在  
    if os.path.exists(image_path):  
        # 打开图片  
        image = Image.open(image_path).convert('RGB')  
        draw = ImageDraw.Draw(image)  
  
        # 遍历标注中的多边形 
        chenchen=0
        for annotation in image_annotations:
            if annotation['category_id']==0:
                continue
            
            points=[] 
            for segmentation in annotation['polys']:  
            # 绘制多边形（假设segmentation是一个点的列表，每个点由x,y坐标组成）  
            #print(segmentation)
                segmentation[0]=round(segmentation[0])
                segmentation[1]=round(segmentation[1])
            #print(segmentation)
                segmentation=tuple(segmentation)
            #points = [tuple(map(int, point)) for point in segmentation]  
            #print(segmentation)
                points.append(segmentation)
        #print(points)
            
            # new_points = [points[0], points[6], points[7], points[-1]]  
            # for x,y in new_points:
            #     # print(x,y)
            #     # haha
            #     #x=x-3
            #     draw.ellipse([x-2, y-2, x+2, y+2], fill=(0,0,0), width=2)
            
            #draw.polygon(new_points, outline='red', width=2)  
            #draw.polygon(points, outline='red', width=2)  
            #draw.polygon(points, outline=colors[chenchen], width=2)  
            #chenchen=(chenchen+1)%20
            for point in points:  
                x, y = point  
                #draw.point((x, y), fill=(255),size=5)
                draw.rectangle([x-6, y-6, x+6, y+6], fill="red")  
    
        
        # 保存图片  
        output_filename = os.path.join(output_folder, image_filename)  
        image.save(output_filename)  
    else:  
        print(f"Image {image_filename} not found for image_id {image_id}")  
  
print("All images with annotations have been processed and saved.")