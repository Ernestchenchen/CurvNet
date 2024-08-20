import torch
from torch import nn
from mmocr.models.textdet.detectors import FCENet
from mmdet.models.builder import build_head
from mmdet.models.builder import DETECTORS
import math
import numpy as np
@DETECTORS.register_module()
class LRANet(FCENet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None,):
        super(LRANet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, show_score, init_cfg)
    def set_epoch(self,epoch):
        self.bbox_head.loss_module.epoch=epoch

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # print(img_metas)
        
        # print("这是img：",img)
        x = self.extract_feat(img)
        
        #print("这是骨干网络的特征：",x[0][0][0][0][0],len(x),x[0].size(),x)
#         if math.isnan(x[0][0][0][0][0]):
#             print("这是img：",img)
#             print(img_metas)
#             with open('img_metas.txt', 'a', encoding='utf-8') as f:
#                 f.write(str(img_metas)+'\n')  # '\n' 是换行符，确保每一项都写在新的一行
#             losses = {
#     'loss_ce_sparse': torch.randn(1, 1, requires_grad=True),
#     'loss_point_sparse': torch.randn(1, 1, requires_grad=True),
#     'loss_ce_dense': torch.randn(1, 1, requires_grad=True),
#     'loss_point_dense': torch.randn(1, 1, requires_grad=True)
# }
#             return losses
        preds = self.bbox_head(x)
        
        #print(preds)
       
        losses = self.bbox_head.loss(preds,**kwargs)
        #print(type(losses),losses)
        #haha
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        #print("这是图片：",img.size())#1, 3, 1824, 800
        x = self.extract_feat(img)#琛 3张特征图
        #print("这是骨干网络的特征：",x[0].size())#torch.Size([1, 256, 228, 100]) ([1, 256, 114, 50]) ([1, 256, 57, 25])
        
        outs = self.bbox_head(x)#琛 共12张特征图 3*4
        # print(len(outs)) #3,4,
        # print(len(outs[0]))
        #print(outs[0][0].size())
        #torch.Size([1, 1, 228, 100]) torch.Size([1, 16, 228, 100]) torch.Size([1, 1, 228, 100])torch.Size([1, 16, 228, 100])
        #torch.Size([1, 1, 114, 50])……
        #torch.Size([1, 1, 57, 25])
        #print(max(img.cpu().numpy().reshape(-1)),min(img.cpu().numpy().reshape(-1)))
        #print(max(x[0].cpu().numpy().reshape(-1)),min(x[0].cpu().numpy().reshape(-1)))
        #haha
        #print(max(outs[0][0].cpu().numpy().reshape(-1)),min(outs[0][0].cpu().numpy().reshape(-1)))
        #print(max(outs[0][1].cpu().numpy().reshape(-1)),min(outs[0][1].cpu().numpy().reshape(-1)))
    #     # 将NumPy数组写入文件  
    #     with open('feature_not_normal.txt', 'w') as f:  
    #     # 写入第一个张量的数据  
    #         #np.savetxt(f, outs[0][0].cpu().numpy().reshape(-1), fmt='%.4f')  # 假设你想保留4位小数  
    # #         f.write('\n')  # 添加一个换行符以便区分两个张量的数据  
    # # # 写入第二个张量的数据  
    #         np.savetxt(f, x[0].cpu().numpy().reshape(-1), fmt='%.4f')  # 同样保留4位小数  
        #print(img_metas)
       
        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs

        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            
            boundaries = [
                self.bbox_head.get_boundary(outs, img_metas, True)
            ]
        
        return boundaries
    
