from pycocotools.coco import COCO  
from pycocotools.cocoeval import COCOeval  
import json  
  
# 加载ground truth标注  
cocoGt = COCO('spinal-AI2024/spinal_AI2024_all_stage6_selection.json')  # 假设使用COCO val2017的标注  
  
# 加载用户提交的预测结果（需要是JSON格式，且格式与COCO的标注文件兼容）  
cocoDt = cocoGt.loadRes('spinal-AI2024-eval/spinal_AI2024_subset0_stage2_selection_results.json')  # 假设预测结果保存在results.json中  
  
# 初始化COCOeval对象，指定评估的任务类型（如'bbox'表示边界框检测）  
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  
  
# 执行评估  
cocoEval.evaluate()  
cocoEval.accumulate()  
cocoEval.summarize()