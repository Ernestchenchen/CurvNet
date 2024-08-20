import os  
import json  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 设置工作目录  
work_dir = 'work_dirs/chenchen_spinal_AASEC2019_groundtruth1'  
  
# 用来存储每个epoch的iter和对应的loss  
losses_per_iter = {}  
  
# 遍历目录下的所有文件  
for filename in os.listdir(work_dir):  
    if filename.endswith('.log.json'):  
        # 读取文件  
        with open(os.path.join(work_dir, filename), 'r') as f:  
            # 跳过第一行（通常是文件头）  
            next(f)  
            epoch = None  
            for line in f:  
                data = json.loads(line)  
                if 'epoch' in data:  
                    epoch = data['epoch']  
                if epoch not in losses_per_iter:  
                    losses_per_iter[epoch] = {  
                        'loss_ce_sparse': [],  
                        'loss_point_sparse': [],  
                        'loss_ce_dense': [],  
                        'loss_point_dense': [],  
                        'loss': []  
                    }  
                  
                # 将loss添加到对应epoch和iter的列表中  
                for loss_key in ['loss_ce_sparse', 'loss_point_sparse', 'loss_ce_dense', 'loss_point_dense', 'loss']:  
                    if 'iter' in data:  # 假设数据中包含了'iter'字段  
                        iter_num = data['iter']  
                        losses_per_iter[epoch][loss_key].append((iter_num, data[loss_key]))  
  
# 用户输入想要查看的epoch范围，例如 '1-3'  
user_input = input("请输入想要查看的epoch范围（例如 '1-3'）：")  
try:  
    start_epoch, end_epoch = map(int, user_input.split('-'))  
except ValueError:  
    print("无效的输入，请输入形如 '1-3' 的epoch范围。")  
    exit()  
  
# 确保输入范围在有效数据内  
valid_epochs = sorted(losses_per_iter.keys())  
if start_epoch < min(valid_epochs) or end_epoch > max(valid_epochs) or start_epoch > end_epoch:  
    print("无效的epoch范围。")  
    exit()  
  
# 绘制折线图  
plt.figure(figsize=(12, 6))  
  
# 定义要绘制的loss的键  
loss_keys = ['loss_ce_sparse', 'loss_point_sparse', 'loss_ce_dense', 'loss_point_dense', 'loss']  
colors = ['r', 'g', 'b', 'y', 'm']  # 为每种loss分配不同的颜色  
  
# 对每个epoch绘制loss  
for epoch in range(start_epoch, end_epoch + 1):  
    if epoch in losses_per_iter:  
        for i, loss_key in enumerate(loss_keys):  
            # 提取当前epoch的所有iter和loss  
            iters = [iter_num for iter_num, _ in losses_per_iter[epoch][loss_key]]  
            losses = [loss for _, loss in losses_per_iter[epoch][loss_key]]  
              
            # 绘制当前epoch的loss曲线  
            plt.plot(iters, losses, label=f'Epoch {epoch} {loss_key}', color=colors[i])  
  
# 设置图例和标题  
plt.legend()  
plt.title('Losses over Iterations for Selected Epochs')  
plt.xlabel('Iteration')  
plt.ylabel('Loss')  
  
# 显示图表  
plt.grid(True)  # 可选：添加网格线  
#plt.show()  
  
# 如果你想要保存图表为文件，可以使用下面的代码  
plt.savefig('epoch_loss.png')