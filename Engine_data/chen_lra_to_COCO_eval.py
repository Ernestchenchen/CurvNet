import json  
  
# 假设这是从文件中读取的结果列表的加载函数  
def load_results(file_path):  
    with open(file_path, 'r') as f:  
        results = json.load(f)  
    return results  
  
# 将多边形转换为边界框的函数  
def poly_to_bbox(poly):  
    x_coords = [p[0] for p in poly]  
    y_coords = [p[1] for p in poly]  
    xmin = min(x_coords)  
    ymin = min(y_coords)  
    xmax = max(x_coords)  
    ymax = max(y_coords)  
    return [xmin, ymin, xmax - xmin, ymax - ymin]  # [x, y, width, height]  
def poly_to_bbox_specific(poly):  
    # 检查poly的长度是否足够  
    if len(poly) < 8:  # 因为需要第七组和第八组点，所以最小长度为8  
        raise ValueError("poly must have at least 8 points")  
      
    # 提取特定点的坐标  
    points = [poly[0], poly[6], poly[7], poly[-1]]  # 第一组、第七组、第八组和最后一组  
      
    # 分离x和y坐标  
    x_coords = [p[0] for p in points]  
    y_coords = [p[1] for p in points]  
      
    # 计算边界框的xmin, ymin, xmax, ymax  
    xmin = min(x_coords)  
    ymin = min(y_coords)  
    xmax = max(x_coords)  
    ymax = max(y_coords)  
      
    # 计算宽度和高度  
    width = xmax - xmin if xmax > xmin else 0  
    height = ymax - ymin if ymax > ymin else 0  
      
    # 返回边界框  
    return [int(xmin), int(ymin), round(width+1), round(height+1)]  # [x, y, width, height]    
# 主函数  
def convert_results_to_bbox(input_file, output_file):  
    # 加载结果  
    results = load_results(input_file)  
  
    # 转换结果  
    converted_results = []  
    for result in results:  

        poly = result['polys']  
        bbox = poly_to_bbox_specific(poly)  
       
        x_min, y_min, width, height = bbox  
  
 
        if width * height  <200:
            continue
        converted_result = {  
            "image_id": result["image_id"]+1+add,  
            "category_id": result["category_id"],  
            "bbox": bbox,  
            "score": result["score"]  
        }  
        converted_results.append(converted_result)  
  
    # 保存转换后的结果  
    with open(output_file, 'w') as f:  
        json.dump(converted_results, f, indent=4)  
  
# 调用主函数
add=16000
input_file = 'spinal_AI2024_test_3_03.json'  #需要评估的lra结果
output_file = 'spinal-AI2024-eval/spinal_AI2024_test_3_03_results.json'  #生成评估文件
convert_results_to_bbox(input_file, output_file)  
  
print("边界框评估结果已保存为", output_file)