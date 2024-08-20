import os  
import json  
import matplotlib.pyplot as plt  
  
# 定义保存图像的路径  
#work_dir = 'chenchen_spinal_AASEC2019_newpca'  
work_dir = 'spinal_AI2024_train_03_on'
image_save_path = os.path.join('work_dirs', work_dir,'loss_final')

  
# 初始化存储loss的字典  
losses = {  
    'loss_ce_sparse': [],  
    'loss_point_sparse': [],  
    'loss_ce_dense': [],  
    'loss_point_dense': [],  
    'loss': []  
}  
  
# 遍历指定目录下的所有文件  
for root, dirs, files in os.walk(os.path.join('work_dirs', work_dir)):  
    for file in files:  
        if file.endswith('.log.json'):  
            file_path = os.path.join(root, file)  
            with open(file_path, 'r') as f:  
                # 跳过第一行（通常可能是文件头或元数据）  
                next(f)  
                for line in f:  
                    data = json.loads(line)  
                    epoch = data['epoch']  
                    for loss_key, loss_list in losses.items():  
                        if loss_key in data:  
                            loss_list.append((epoch, data[loss_key]))  
  
# 将每个loss的列表转换成字典，其中epoch作为键，loss作为值  
for loss_key, loss_list in losses.items():  
    losses[loss_key] = {epoch: loss for epoch, loss in loss_list}  
  
# 提取所有的epoch（假设它们是连续的）  
epochs = sorted(set(epoch for loss_dict in losses.values() for epoch in loss_dict.keys()))  
  
# 绘制折线图  
plt.figure(figsize=(12, 6))  
for loss_key, loss_dict in losses.items():  
    plt.plot(epochs, [loss_dict[epoch] if epoch in loss_dict else None for epoch in epochs], label=loss_key, marker='o')  
  
plt.title('Final Losses over Epochs')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.grid(True)  
plt.legend()  
plt.tight_layout()  
  
# 保存图像  
plt.savefig(image_save_path)  
