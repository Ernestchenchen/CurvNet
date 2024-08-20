import json  
  
# 加载COCO标注文件  
with open('spinal-AI2024-advanced/spinal_AI2024_advance_all_stage0_best_exist.json', 'r') as f:  
    coco_data = json.load(f)  
  
# 对图像进行排序（虽然这个步骤对于重新编号标注的id不是必需的，但如果您需要保持图像顺序，则可以保留）  
coco_data['images'].sort(key=lambda x: x['id'])  
  
# 对标注进行排序，先按图像ID排序，再按标注ID排序  
coco_data['annotations'].sort(key=lambda x: (x['image_id'], x['id']))  
  
# 重新为标注分配连续的id，从1开始  
new_annotation_ids = iter(range(1, len(coco_data['annotations']) + 1))  
for annotation in coco_data['annotations']:  
    annotation['id'] = next(new_annotation_ids)  
  
# 保存排序并重新编号后的COCO标注文件  
with open('spinal-AI2024-advanced/spinal_AI2024_advance_all_stage0_best_exist_reorder.json', 'w') as f:  
    json.dump(coco_data, f, indent=4)  
  
print("COCO标注文件已按ID排序并重新编号后保存为：spinal-AI2024/spinal_AI2024_subset1_5_stage6_best_exist_sorted_and_renumbered.json")