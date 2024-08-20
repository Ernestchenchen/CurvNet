import os
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
exclude=[
    '000020.png',
    '000021.png',
    '000026.png',
    '000073.png',
    '000131.png',
    '000176.png',
    '000186.png',
    '000223.png',
    '000224.png',
    '000314.png',
    '000338.png',
    '000355.png',
    '000374.png',
    
    '000438.png',
    '000444.png',
    '000493.png',
    '000713.png',
    '000803.png',
    '000236.png',
    '000346.png',
    '000241.png',
    '000214.png',
    '000357.png',
    '000495.png',
    '000460.png',
    '000841.png',
    
    '000706.png',
    '000882.png',
    '000774.png',
    
    
]  
exclude2=[
    
    '000443.png',
    '000681.png',
    '000514.png',
    '000541.png',
    '000737.png',
    '000438.png',
    '000803.png',
    '000214.png',
]

def calculate_ssim(image1, image2):
    resized_image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0])) 
    # 将图像转换为灰度图
    gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    diff = np.abs(gray1.astype("float") - gray2.astype("float"))  
      
    # 计算平均差异  
    mean_diff = np.mean(diff)  
      
    # 由于我们想要得到的是雷同程度，因此可以用1减去平均差异（或进行其他适当的归一化）  
    # 这里简单地使用1减去平均差异（注意：这里假设像素值范围在0到255之间）  
    # 得到的值越接近1，表示两张图像越相似  
    similarity = 1 - (mean_diff / 255.0) 
    
    # 计算SSIM
    
    
    (a, ssim_result) = ssim(gray1, gray2, full=True)
    return a,similarity



def find_most_similar_images(source_dir, target_dirs):
    match_file = open(os.path.join(source_dir, 'spinal-AI2024-subset0-match-sim.txt'), 'a')
    match_file2 = open(os.path.join(source_dir, 'spinal-AI2024-subset0-match-sim2.txt'), 'a')
    # 遍历源目录下的所有图片文件
     
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith('.jpg'):
            print(filename)
            top_20_similarities = [] 
            top_5_similarities=[]
            source_image_path = os.path.join(source_dir, filename)
            source_image = cv2.imread(source_image_path)
            # 遍历目标目录下的所有图片文件
            for target_dir in target_dirs:
                for target_filename in sorted(os.listdir(target_dir)):
                    if target_filename.endswith('.png'):
                        
                        target_image_path = os.path.join(target_dir, target_filename)
                        target_image= cv2.imread(target_image_path)
                        similarity,similarity2=calculate_ssim(source_image, target_image)
                        # 如果相似度大于当前最大相似度，则更新最大相似度和最相似图片的文件名
                        if len(top_20_similarities) < 10 and target_filename not in exclude:  
                            top_20_similarities.append([target_filename, similarity])  
                            top_20_similarities.sort(key=lambda x: x[1], reverse=True)  
                        elif target_filename not in exclude:  
                            # 如果新计算的相似度大于列表中最小的一个，则替换之  
                            if similarity > top_20_similarities[-1][1]:  
                                top_20_similarities.pop()  # 移除最小相似度的元素  
                                top_20_similarities.append([target_filename, similarity])  
                                top_20_similarities.sort(key=lambda x: x[1], reverse=True) 
                        if len(top_5_similarities) < 10 and target_filename not in exclude2:  
                            top_5_similarities.append([target_filename, similarity2])  
                            top_5_similarities.sort(key=lambda x: x[1], reverse=True)  
                        elif target_filename not in exclude2:   
                            # 如果新计算的相似度大于列表中最小的一个，则替换之  
                            if similarity2 > top_5_similarities[-1][1]:  
                                top_5_similarities.pop()  # 移除最小相似度的元素  
                                top_5_similarities.append([target_filename, similarity2])  
                                top_5_similarities.sort(key=lambda x: x[1], reverse=True) 
            # 将结果写入文件
            match_file.write(f"{filename} ")
            for index, (target_filename, similarity) in enumerate(top_20_similarities, start=1):  
                match_file.write(f"{target_filename} {similarity} ")
            match_file.write(f"\n")
            match_file2.write(f"{filename} ")
            for index, (target_filename, similarity) in enumerate(top_5_similarities, start=1):  
                match_file2.write(f"{target_filename} {similarity} ")
            match_file2.write(f"\n")

            
    match_file.close()
# 指定源目录和目标目录
source_dir = 'spinal-AI2024/spinal-AI2024-all'
target_dirs = ['spinal2023/test', 'spinal2023/train']
# 执行任务
find_most_similar_images(source_dir, target_dirs)