import numpy as np  
import math
def SMAPE(y_true, y_pred):  
    out_abs = np.abs(y_true - y_pred)  
    out_add = y_true + y_pred  
    # 避免除以零  
    out_add[out_add == 0] = 1e-9  
    term1 = np.sum(out_abs)  
    term2 = np.sum(out_add)  
    SMAPE_value = (term1 / term2) * 100  
    return SMAPE_value  

def chebyshev_distance(point1, point2):  
    """  
    计算两个点（由列表表示，每个列表包含三个数字）之间的Chebyshev距离  
  
    参数:  
    point1 (list of int or float): 第一个点，包含三个数字  
    point2 (list of int or float): 第二个点，包含三个数字  
  
    返回:  
    float: 两个点之间的Chebyshev距离  
    """  
    # 确保两个点都有三个维度  
    if len(point1) != 3 or len(point2) != 3:  
        raise ValueError("Each point must have exactly three dimensions")  
  
    # 计算Chebyshev距离  
    return max(abs(p1 - p2) for p1, p2 in zip(point1, point2))  

def euclidean_distance(point1, point2):  
    """  
    计算两个点（由列表表示，每个列表包含三个数字）之间的欧氏距离  
  
    参数:  
    point1 (list of int or float): 第一个点，包含三个数字  
    point2 (list of int or float): 第二个点，包含三个数字  
  
    返回:  
    float: 两个点之间的欧氏距离  
    """  
    # 确保两个点都有三个维度  
    if len(point1) != 3 or len(point2) != 3:  
        raise ValueError("Each point must have exactly three dimensions")  
  
    # 计算欧氏距离  
    distance = math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))  
    return distance  
  
def calculate_smape(file1, file2):  
    with open(file1, 'r') as f1, open(file2, 'r') as f2:  
        lines1 = f1.readlines()  
        lines2 = f2.readlines()  
  
    # # 假设两个文件具有相同数量的行  
    # if len(lines1) != len(lines2):  
    #     raise ValueError("Files do not have the same number of lines.")  
  
    smape_values_per_column = [[] for _ in range(3)]  # 用于存储每个列的SMAPE值  
    manhattan_distance_values_per_column = [[] for _ in range(3)]
    euclidean_distance_values_per_column = []
    chebyshev_distance_values_per_column = []
    for line1, line2 in zip(lines1, lines2):  
        parts1 = line1.strip().split(',')  
        parts2 = line2.strip().split(',')  
        if len(parts1) < 4 or len(parts2) < 4:  
            continue  # 跳过行数不足的行  
        
        values1 = [float(v) for v in parts1[1:]]  
        values2 = [float(v) for v in parts2[1:]]  
        #print(values1)
        
        #print(values2)
        euclidean_distance_values_per_column.append(euclidean_distance(values1, values2))
        chebyshev_distance_values_per_column.append(chebyshev_distance(values1, values2))
        # 计算每对数值的SMAPE值  
        for i in range(3):  
            smape_value = SMAPE(np.array([values1[i]]), np.array([values2[i]]))  
            smape_values_per_column[i].append(smape_value)  
            manhattan_distance_values_per_column[i].append(abs(np.array([values1[i]])-np.array([values2[i]])))
  
    # 计算每个列的平均SMAPE值  
    avg_smape_per_column = [np.mean(values) for values in smape_values_per_column]  
    
    manhattan_distance_values_per_column=[np.mean(values) for values in manhattan_distance_values_per_column]
    
    chebyshev_distance_values_per_column=[np.mean(chebyshev_distance_values_per_column)]
    cd=chebyshev_distance_values_per_column[0]
    euclidean_distance_values_per_column=[np.mean(euclidean_distance_values_per_column) ]
    ed=euclidean_distance_values_per_column[0]
    
    md=sum(manhattan_distance_values_per_column)
    
    # 计算每列的最大值的SMAPE值
    # 额外计算每行的最大值的SMAPE值（如果需要的话）  
    max_smape_values = []  
    for values1, values2 in zip(lines1, lines2):  
        parts1 = values1.strip().split(',')  
        parts2 = values2.strip().split(',')  
        if len(parts1) < 4 or len(parts2) < 4:  
            continue  
  
        values1 = [float(v) for v in parts1[1:]]  
        values2 = [float(v) for v in parts2[1:]]  
  
        max_values1 = np.max(values1)  
        max_values2 = np.max(values2)  
        max_smape_values.append(SMAPE(np.array([max_values1]), np.array([max_values2])))  
  
    avg_max_smape = np.mean(max_smape_values) if max_smape_values else None  
  
    return avg_smape_per_column, avg_max_smape,md,ed,cd  
  
# 使用示例  
file1 = 'Cobb_spinal2023_GT_new.txt'  
file2 = 'Cobb_spinal2023_GT.txt'  
# file1="a.txt"
# file2="b.txt"
avg_smape_per_column, avg_max_smape ,md,ed ,cd = calculate_smape(file1, file2)  
  
print("Average SMAPE per column:")  
for i, avg_smape in enumerate(avg_smape_per_column, start=1):  
    print(f"Column {i}: {avg_smape:.2f}%")  
  
print(f"Average SMAPE for maximum values: {avg_max_smape:.2f}%")

print(f"ED: {ed:.2f}°")
print(f"MD: {md:.2f}°")
print(f"CD: {cd:.2f}°")
