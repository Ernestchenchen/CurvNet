import json  
  
# 定义JSON文件路径  
file_path = 'spinal-AI2024/spinal_AI2024_all_stage6_best.json'  
  
# 读取并解析JSON文件  
with open(file_path, 'r') as file:  
    data = json.load(file)  
  
# 检查数据中是否存在images键  
if 'images' in data:  
    images = data['images']  
      
    # 初始化变量以存储统计结果  
    num_images = len(images)  
    max_width = 0  
    min_width = float('inf')  
    max_height = 0  
    min_height = float('inf')  
      
    # 遍历images数组，更新统计值  
    for image in images:  
        width = image['width']  
        height = image['height']  
          
        # 更新图片数量的统计  
        num_images = num_images  
          
        # 更新宽度和高度的最大值和最小值  
        if width > max_width:  
            max_width = width  
        if width < min_width:  
            min_width = width  
        if height > max_height:  
            max_height = height  
        if height < min_height:  
            min_height = height  
              
    # 输出统计结果  
    print(f"一共有 {num_images} 张图片。")  
    print(f"图片宽度的最大值是 {max_width}，最小值是 {min_width}。")  
    print(f"图片高度的最大值是 {max_height}，最小值是 {min_height}。")  
else:  
    print(f"Error: JSON文件中缺少'images'键。")