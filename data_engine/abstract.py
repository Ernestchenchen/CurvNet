import json  
  
# 加载原始的COCO标注文件  
with open('spinal-AI2024-advanced/spinal_AI2024_advance_subset6_stage0_best.json', 'r') as f:  
    coco_data = json.load(f)  
  
# 加载需要保留的图片ID列表  
with open('spinal-AI2024-advanced/selected_images6.txt', 'r') as f:  
    list_images=f.read().splitlines()
    exist_ids = set([int(list_image.split('.')[0]) for list_image in list_images]) 
    # for i in list_image:
    #     # print(i)
    #     # haha
    #     i=int(i.split(".")[0])
    # print(list_image)
    # haha
    # #exist_ids = set(map(int, f.read().splitlines().spilt(.)[0]))  

# print(exist_ids)
# haha
# 初始化新的COCO数据结构  
new_coco_data = {  
    'images': [],  
    'annotations': [],  
    'categories': coco_data.get('categories', []),  # 假设类别不需要修改  
    'info': coco_data.get('info', {}),  # 假设其他信息也不需要修改  
    'licenses': coco_data.get('licenses', []),  # 假设许可证信息也不需要修改  
}  
  
# 遍历原始COCO数据中的images和annotations，根据exist_ids筛选  
image_id_to_info = {image['id']: image for image in coco_data['images']}  
annotations_by_image_id = {}  
for annotation in coco_data['annotations']:  
    annotations_by_image_id.setdefault(annotation['image_id'], []).append(annotation)  
  
for image_id in exist_ids:  
    if image_id in image_id_to_info:  
        # 添加图片信息  
        new_coco_data['images'].append(image_id_to_info[image_id])  
        # 添加对应的标注信息  
        if image_id in annotations_by_image_id:  
            new_coco_data['annotations'].extend(annotations_by_image_id[image_id])  
  
# 写入新的COCO标注文件  
with open('spinal-AI2024-advanced/spinal_AI2024_advance_subset6_stage0_best_exist.json', 'w') as f:  
    json.dump(new_coco_data, f, indent=4)  
  
print("新的COCO标注文件已生成!")