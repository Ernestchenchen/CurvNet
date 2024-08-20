import json  
  
def filter_coco_annotations(coco_file, discard_file, output_file):  
    # 加载COCO标注文件  
    with open(coco_file, 'r') as f:  
        coco_data = json.load(f)  
  
    # 加载要剔除的图片ID列表  
    discard_ids = set()  
    with open(discard_file, 'r') as f:  
        for line in f:  
            # 假设每行仅包含一个ID，并转换为整数  
            discard_ids.add(int(line.strip()))  
  
    # 过滤掉包含在discard_ids中的图片及其标注  
    filtered_images = [img for img in coco_data['images'] if img['id'] not in discard_ids]  
    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] not in discard_ids]  
  
    # 更新COCO数据中的图片和标注  
    coco_data['images'] = filtered_images  
    coco_data['annotations'] = filtered_annotations  
  
    # 更新categories（如果需要根据图片过滤结果更新的话，但通常不需要）  
    # 这里假设categories不需要更新  
  
    # 将过滤后的COCO数据写入新文件  
    with open(output_file, 'w') as f:  
        json.dump(coco_data, f, indent=4)  
  
# 调用函数  
coco_file = 'spinal-AI2024/spinal_AI2024_all_stage6_best.json'  
discard_file = 'spinal-AI2024/spinal_AI2024_all_stage5_discard.txt'  
output_file = 'spinal-AI2024/spinal_AI2024_all_stage6_self5_selection.json'  
  
filter_coco_annotations(coco_file, discard_file, output_file)