import numpy as np  
from sklearn.metrics import jaccard_score  # 可以用来计算IOU，虽然不是直接计算但可用于相似目的
import json
from collections import OrderedDict


def vertices_to_bbox(vertices):  
    """  
    将四个顶点的坐标转换成矩形框的x1, y1, x2, y2格式。  
      
    参数:  
    - vertices: 一个包含8个数字的列表或NumPy数组，格式为[x1, y1, x2, y2, x3, y3, x4, y4]。  
      
    返回:  
    - bbox: 一个包含4个元素的列表或NumPy数组，格式为[x1, y1, x2, y2]。  
    """  
    # 确保vertices是一个NumPy数组以便进行向量化操作  
    vertices = np.array(vertices).reshape(-1, 2)  
      
    # 计算x和y的最小值和最大值  
    x1, y1 = vertices.min(axis=0)  
    x2, y2 = vertices.max(axis=0)  
      
    # 返回矩形框的坐标  
    return [x1, y1, x2, y2]  

def iou_score(box1, box2):  
    """计算两个矩形框的IOU。  
    box1, box2: (x1, y1, x2, y2) 格式  
    """  
    inter_rect = [  
        max(box1[0], box2[0]),  
        max(box1[1], box2[1]),  
        min(box1[2], box2[2]),  
        min(box1[3], box2[3])  
    ]  
    inter_area = max(0, inter_rect[2] - inter_rect[0]) * max(0, inter_rect[3] - inter_rect[1])  
  
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  
  
    union_area = box1_area + box2_area - inter_area  
    iou = inter_area / union_area  
    return iou  
  
def regression_loss(pred, target):  
    """计算回归loss，这里使用L1 loss。"""  
    return np.mean(np.abs(pred - target))  
  
def classification_loss(pred, target):  
    """计算分类loss，这里使用交叉熵的简化形式（假设二分类）。"""  
    return -np.mean(target * np.log(pred + 1e-9) + (1 - target) * np.log(1 - pred + 1e-9))

def combined_loss(pred_classes, pred_boxes, target_classes, target_boxes, weights=[1.0, 1.0, 1.0]):  
    """  
    计算组合loss。  
      
    pred_classes: 预测的分类结果，形状 (N,)  
    pred_boxes: 预测的矩形框，形状 (N, M, 4)，其中M是预测框的数量，可能每个样本不同  
    target_classes: 真实的分类结果，形状 (N,)  
    target_boxes: 真实的矩形框，形状 (N, K, 4)，其中K是真实框的数量，每个样本可能不同  
    weights: 权重列表，分别对应分类loss、平均回归loss和平均IOU loss  
    """  
    # 分类loss  
    class_loss = classification_loss(pred_classes, target_classes)  
      
    # 初始化回归和IOU loss列表  
    reg_losses = []  
    iou_losses = []  
    #print(pred_boxes)
    # print(pred_boxes[20438])
    # print(pred_boxes[20439])
    # haha
    # 遍历每个样本  
    for i in range(len(target_classes)):  
        if target_classes[i] == 1 and pred_classes[i]==1:  # 只计算正样本的回归和IOU loss  
            num_pred_boxes = pred_boxes[i].shape[0]  
            num_target_boxes = target_boxes[i].shape[0]  
            reg_loss = regression_loss(num_pred_boxes, num_target_boxes)   
            # 计算每个预测框与所有真实框之间的最小回归和IOU loss  
            
            min_iou_loss = np.inf  
            if num_pred_boxes <= 0 :
                print(i,target_classes[i],pred_classes[i])
            for j in range(num_pred_boxes):  
                
                min_iou_loss_j = np.inf  
                for k in range(num_target_boxes):  
                    # 计算回归loss  
                   
                    # 计算IOU loss (1 - iou)  
                    iou = iou_score(pred_boxes[i][j], target_boxes[i][k]) 
                    
                    iou_loss_jk = 1 - iou  
                      
                    # 更新最小loss  
                    
                    if iou_loss_jk < min_iou_loss_j:  
                        min_iou_loss_j = iou_loss_jk  
                  
                # 对每个预测框取最小loss  
               
                min_iou_loss = min(min_iou_loss, min_iou_loss_j)  
                #print(min_iou_loss)
                
            # 累加最小loss  
            reg_losses.append(reg_loss)  
            #print(min_iou_loss)
            iou_losses.append(min_iou_loss)  
            #print(reg_loss,min_iou_loss)
      
    # 如果存在正样本，则计算平均回归和IOU loss；否则设为0  
    if reg_losses:  
        reg_loss = np.mean(reg_losses)  
    else:  
        reg_loss = 0  
    if iou_losses:  
        iou_loss = np.mean(iou_losses)  
    else:  
        iou_loss = 0  
          
    total_loss = weights[0] * class_loss + weights[1] * reg_loss + weights[2] * iou_loss  
    print(class_loss,reg_loss,iou_loss)    
    #print(total_loss)
    return total_loss  # 修改为返回loss，以便在其他地方使用  


      
      
  
# 文件路径  
file_path = 'spinal-AI2024/spinal_AI2024_all_stage5_discard.txt'  
target_file_path = 'spinal-AI2024/spinal_AI2024_all_stage6_discard.txt'
coco_file_path = 'spinal-AI2024/spinal_AI2024_all_stage5_best.json'  
target_coco_file_path = 'spinal-AI2024/spinal_AI2024_all_stage6_best.json'   
# 创建一个长度为20440的数组，并初始化为1  
pred_classes = np.ones(20440, dtype=int)  
# 读取文件并处理  
with open(file_path, 'r') as file:  
    # 遍历文件的每一行  
    for line in file:  
        # 去除行尾的换行符，并将字符串转换为整数  
        num = int(line.strip())  
          
        # 检查数字是否在有效范围内，并更新数组中对应位置的值  
        if 0 <= num < 20440:  
            pred_classes[num-1] = 0  
  

# 创建一个长度为20440的数组，并初始化为1  
target_classes = np.ones(20440, dtype=int)  
# 读取文件并处理  
with open(target_file_path, 'r') as file:  
    # 遍历文件的每一行  
    for line in file:  
        # 去除行尾的换行符，并将字符串转换为整数  
        num = int(line.strip())  
        # 检查数字是否在有效范围内，并更新数组中对应位置的值  
        if 0 <= num < 20440:  
            target_classes[num-1] = 0  
  

# 读取JSON文件  
with open(coco_file_path, 'r') as f:  
    coco_data = json.load(f)  

image_boxes = {} 
  
# 遍历annotations
for image_id in range(1,20441):  
    image_boxes[image_id] = []  
for annotation in coco_data['annotations']: 
    if annotation['category_id']==0:
        continue
   
    image_id = annotation['image_id']  
    segmentation = annotation['segmentation']  
    bbox=vertices_to_bbox(segmentation[0])
    if isinstance(segmentation, list) and len(segmentation) == 1 and len( bbox) == 4:  
        x1, y1, x2, y2 =  bbox
        # 将矩形框添加到对应图片的列表中（如果还没有，则先初始化列表）  
        image_boxes[image_id].append([x1, y1, x2, y2])  
   
boxes_list = [np.array(boxes, dtype=int) if boxes else np.array([], dtype=int) for boxes in image_boxes.values()]
pred_boxes=np.array(boxes_list,dtype=object)

  
# 读取JSON文件  
with open(target_coco_file_path, 'r') as f:  
    target_coco_data = json.load(f)  
  
# 初始化一个字典来存储每张图片的矩形框  
target_image_boxes = {}
for image_id in range(1,20441):  
    target_image_boxes[image_id] = [] 
# 遍历annotations  
for annotation in target_coco_data['annotations']:  
    if annotation['category_id']==0:
        continue
    image_id = annotation['image_id'] 
    
    segmentation = annotation['segmentation']  
    #print(segmentation)
    bbox=vertices_to_bbox(segmentation[0])
    if isinstance(segmentation, list) and len(segmentation) == 1 and len( bbox) == 4:  
        x1, y1, x2, y2 =  bbox
        # 将矩形框添加到对应图片的列表中（如果还没有，则先初始化列表）  
        target_image_boxes[image_id].append([x1, y1, x2, y2])  
  
# 创建一个列表，其中每个元素都是对应图片的矩形框列表  
target_boxes_list = [np.array(boxes, dtype=int) if boxes else np.array([], dtype=int) for boxes in target_image_boxes.values()]
  
target_boxes=np.array(target_boxes_list,dtype=object)

# 权重列表（分类loss, 回归loss, IOU loss）  
weights = [1.0, 1.0, 0.5]  # 假设我们给IOU loss较低的权重  
  
total_loss = combined_loss(pred_classes, pred_boxes, target_classes, target_boxes, weights)  
  
print("Total Loss:", total_loss)