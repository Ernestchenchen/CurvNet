import json  
  
def filter_coco_results(results_file, discard_file, output_file):  
    # 加载检测结果文件  
    with open(results_file, 'r') as f:  
        results_data = json.load(f)  
  
    # 加载要剔除的图片ID列表  
    discard_ids = set()  
    with open(discard_file, 'r') as f:  
        for line in f:  
            discard_ids.add(int(line.strip()))  
  
    # 过滤掉包含在discard_ids中的图片ID对应的检测结果  
    filtered_results = [result for result in results_data if result['image_id'] not in discard_ids]  
  
    # 将过滤后的结果写入新文件  
    with open(output_file, 'w') as f:  
        json.dump(filtered_results, f, indent=4)  
  
# 调用函数  
results_file = 'spinal-AI2024-eval/spinal_AI2024_subset0_stage5_results.json'  
discard_file = 'spinal-AI2024/spinal_AI2024_all_stage5_discard.txt'  
output_file = 'spinal-AI2024-eval/spinal_AI2024_subset0_stage5_self_selection_results.json'  
  
filter_coco_results(results_file, discard_file, output_file)