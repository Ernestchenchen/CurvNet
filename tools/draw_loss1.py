import os  
import json  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 文件夹路径  
#work_dir = 'chenchen_spinal_AASEC2019_newpca'  
work_dir = 'spinal_AI2024_train_03_on'
image_save_path = os.path.join('work_dirs', work_dir,'loss_average')
folder_path = os.path.join('work_dirs', work_dir)
# loss键的列表  
loss_keys = ['loss_ce_sparse', 'loss_point_sparse', 'loss_ce_dense', 'loss_point_dense', 'loss']  
# 初始化字典来存储每个epoch的loss列表  
epoch_losses = {k: [] for k in loss_keys}  
# 用来存储所有的epoch值  
all_epochs = set()  
  
# 遍历文件夹下的文件  
for filename in os.listdir(folder_path):  
    if filename.endswith('.log.json'):  
        file_path = os.path.join(folder_path, filename)  
          
        # 读取文件内容  
        with open(file_path, 'r') as f:  
            # 跳过第一行（通常是文件头）  
            next(f)  
              
            # 读取并解析JSON数据  
            for line in f:  
                
                data = json.loads(line)  
                epoch = data['epoch']  
                all_epochs.add(epoch)  # 收集所有的epoch值  
                  
                # 更新epoch_losses字典中的loss值  
                for key in loss_keys:  
                    if key in data:  
                        epoch_losses[key].append((epoch, data[key]))  
  
# 转换epoch_losses字典为更容易处理的格式  
# 我们将创建一个新的字典，其中键是epoch，值是包含每个loss键的损失的列表  
epoch_data = {epoch: {k: [] for k in loss_keys} for epoch in all_epochs}  
for key, values in epoch_losses.items():  
    for epoch, value in values:  
        epoch_data[epoch][key].append(value)  
  
# 计算每个epoch的loss平均值  
for epoch, loss_dict in epoch_data.items():  
    for key in loss_keys:  
        if loss_dict[key]:  # 如果列表非空  
            epoch_data[epoch][key] = np.mean(loss_dict[key])  
  
# 绘制折线图  
plt.figure(figsize=(12, 6))  
epochs = sorted(epoch_data.keys())  # 对epoch进行排序以确保绘图顺序  
for i, key in enumerate(loss_keys):  
    plt.plot(epochs, [epoch_data[epoch][key] for epoch in epochs], label=key, marker='o')  
  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.title('Average Loss over Epochs')  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
  
# 保存图像  
plt.savefig(image_save_path)  
plt.show()