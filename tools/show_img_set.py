import re

# 假设img_metas.txt文件的每一行都是一个字典的字符串表示，且"ori_filename"的值为图片文件名
# 例如: '{"ori_filename": "000001.png", "other_key": "value"}'

def extract_numbers_from_file(file_path):
    numbers = set()  # 使用集合来自动去重
    with open(file_path, 'r') as file:
        for line in file:
            # 假设字典的键值对用双引号包围，且ori_filename的值为"数字.png"
            #print(line)
            match = re.search(r"'ori_filename':\s*'(\d{6})\.png'", line)
            if match:
                print(match.group(1))
                
                numbers.add(int(match.group(1)))  # 将找到的数字添加到集合中
    
    # 将集合转换为列表并按数字大小排序
    sorted_numbers = sorted(list(numbers))
    return sorted_numbers

# 获取img_metas.txt文件的路径
file_path = 'img_metas.txt'

# 调用函数并打印结果
numbers = extract_numbers_from_file(file_path)
print(numbers)