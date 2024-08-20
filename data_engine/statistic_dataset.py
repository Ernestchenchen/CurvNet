# 导入必要的库  
from collections import defaultdict  
  
# 初始化一个字典来存储每个区间的文件名和最大值  
intervals = defaultdict(list)  
globalmax=-1
globalmin=100000000  
# discard_file = 'spinal-AI2024/spinal_AI2024_all_subset0_discard-final.txt'  
# # 加载要剔除的图片ID列表  
# discard_ids = set()  
# with open(discard_file, 'r') as f:  
#     for line in f:  
#         # 假设每行仅包含一个ID，并转换为整数  
#         #print(line.strip())
#         if line.strip()=='':
#             continue
#         discard_ids.add(int(line.strip())) 
save45=[]
#output_file_txt='Cobb_spinal-AI2024-advanced-45'
# 读取文件并处理  
with open('Cobb_spinal-AI2024-advanced_gt.txt', 'r') as file:  
    lines = file.readlines()  
    file_count = 0  # 用于计数文件名  
   
    for line in lines:  
        parts = line.strip().split(',')  # 按逗号分割每行  
        if len(parts) == 4:  # 确保每行都有四个部分  
            file_name = parts[0]  
#             prefix = file_name.split('.')[0]  # 假设文件名中只有一个'.'  
  
# # 将前缀（字符串）转换为整型  
#             id_int = int(prefix)  
#             if id_int in discard_ids:
#                 continue
            
            numbers = [float(num) for num in parts[1:]]  # 将数字部分转换为浮点数  
            max_num = max(numbers)  # 找到最大值  
            file_count += 1  # 文件名计数  
            if max_num > globalmax:
                globalmax=max_num
            if max_num < globalmin:
                globalmin=max_num
            # 根据最大值所在区间，将文件名和最大值添加到相应的列表中  
            if 0 <= max_num < 10:  
                intervals['0-10'].append((file_name, max_num))  
            elif 10 <= max_num < 30:  
                intervals['11-30'].append((file_name, max_num))  
            elif 30 <= max_num < 45:  
                intervals['31-45'].append((file_name, max_num))  
            else:  
                intervals['45+'].append((file_name, max_num))  
                # line = f"{file_name},{max_num}\n"  
                # with open(output_file_txt, 'a', encoding='utf-8') as f:  
                #     f.write(line)  
                    
  
# 输出结果  



print(f"Total file names: {file_count}")  
  
for interval, data in intervals.items():  
    if data:  
        print(f"{interval} interval: {len(data)} files")  
        avg_max = sum(max_num for _, max_num in data) / len(data)  
        print(f"Average of max numbers: {avg_max:.2f}")  
    else:  
        print(f"{interval} interval: 0 files")
print("最大角度：",globalmax)
print("最小角度：",globalmin)