import json  
import os  
from PIL import Image, ImageDraw  
  
# 路径设置  
coco_json_path = 'spinal-AI2024/spinal_AI2024_test.json'  
image_dir = 'spinal-AI2024/spinal-AI2024-subset5'  
output_dir = 'spinal-AI2024-show/spinal_AI2024_test'  
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  
  
# 读取COCO标注文件  
with open(coco_json_path, 'r') as f:  
    coco_data = json.load(f)  
  
# 遍历每张图片和对应的标注  
for image in coco_data['images']:  
    image_id = image['id']  
    file_name = image['file_name']  
    image_path = os.path.join(image_dir, file_name)  
  
    # 加载图片  
    img = Image.open(image_path)  
    draw = ImageDraw.Draw(img)  
  
    # 遍历与当前图片相关的标注  
    for annotation in coco_data['annotations']:  
        if annotation['image_id'] == image_id:  
            # 假设segmentation字段是一个包含多边形顶点的列表  
            segmentation = annotation['segmentation'][0]  
            # segmentation可能是一个RLE（Run-Length Encoding）格式的列表，这里我们假设它是直接的多边形坐标列表  
            if len(segmentation) % 2 == 0:  # 确保坐标数量是偶数  
                # print(segmentation)
                # haha
                # points = [(x, y) for x, y in zip(segmentation[::2], segmentation[1::2])]  
                # # 绘制多边形  
                # print(points)
                # haha
                for x,y in zip(segmentation[::2], segmentation[1::2]):
                   
                    x=x-3
                    #draw.polygon(points, outline='red', fill=None, width=2)  
                
                    draw.ellipse([x-2, y-2, x+2, y+2], fill=(255,0,102), width=4)
                    #draw.ellipse([x-2, y-2, x+2, y+2], fill=(51,204,51), width=4)
                    
                
  
    # 保存图片  
    output_path = os.path.join(output_dir, file_name)  
    img.save(output_path)  
  
print("标注绘制完成，图片已保存到:", output_dir)