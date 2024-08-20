import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmdet.core import multi_apply
from mmdet.models.builder import LOSSES
from fvcore.nn import sigmoid_focal_loss_jit
from mmocr.utils.misc import get_world_size, is_dist_avail_and_initialized
from scipy.optimize import linear_sum_assignment

INF = 1000000000

class SetCriterion(nn.Module):
    """ This class computes the loss for OneNet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        src_logits = src_logits.permute(0, 2, 1)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_inst
        losses = {'loss_ce_sparse': class_loss}

        return losses

    def loss_points(self, outputs, targets, indices, num_inst):

        idx = self._get_src_permutation_idx(indices)

        src_points = outputs['pred_points']
        bs, k, hw = src_points.shape
        src_points = src_points.permute(0, 2, 1).reshape(bs, hw, k)
        src_points = src_points[idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_points = F.smooth_l1_loss(src_points, target_points, reduction='none')
        losses['loss_point_sparse'] = loss_points.mean(-1).sum() / num_inst

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_inst = sum(len(t["labels"]) for t in targets)
        num_inst = torch.as_tensor([num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst))

        return losses


class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):

        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, k, hw = outputs["pred_logits"].shape
        # We flatten to compute the cost matrices in a batch
        batch_out_prob = outputs["pred_logits"].permute(0, 2, 1).sigmoid()  # [batch_size, num_queries, num_classes]
        batch_out_po =  outputs["pred_points"].permute(0, 2, 1) 
        indices = []

        tr_train_masks = outputs['tr_train_masks']

        for i in range(bs):

            tgt_ids = targets[i]["labels"]
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]), torch.as_tensor([])))
                continue

            tgt_bbox = targets[i]["points"]
            img_size = targets[i]["image_size"]
            out_prob = batch_out_prob[i]
            out_po = batch_out_po[i] / img_size
            tgt_po = tgt_bbox.clone() / img_size
            cost_po = torch.cdist(out_po, tgt_po, p=1)

            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            tr_train_masks_i = tr_train_masks[i]
            pos_idx = torch.where(tr_train_masks_i <= 0)[0]
            C = self.cost_class * cost_class +  self.cost_point * cost_po
            C[pos_idx] = INF
            src_ind, tgt_ind = linear_sum_assignment(C.cpu())
            indices.append((src_ind, tgt_ind))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@LOSSES.register_module()
class LRALoss(nn.Module):

    def __init__(self, num_coefficients, path_lra, ohem_ratio=3.,
                 with_weight=True,with_area_weight=True, steps = [8,16,32]
                 ):
        super().__init__()

        self.eps = 1e-6
        self.ohem_ratio = ohem_ratio
        self.with_center_weight = with_weight
        self.with_area_weight = with_area_weight
        self.steps = steps
        self.num_coefficients = num_coefficients
        U_t = np.load(path_lra)['components_c']
        U_t = torch.from_numpy(U_t)
        self.U_t = U_t

        losses = ["labels","points"]
        #weight_dict = {"loss_ce_sparse": 2, 'loss_point_sparse': 0.1, "loss_ce_dense": 1, 'loss_point_dense': 4, }
        weight_dict = {"loss_ce_sparse": 1, 'loss_point_sparse': 0.1, "loss_ce_dense": 1, 'loss_point_dense': 0.1, }#0
        #weight_dict = {"loss_ce_sparse": 1, 'loss_point_sparse': 0.0, "loss_ce_dense": 1, 'loss_point_dense': 0.0, }
        #weight_dict = {"loss_ce_sparse": 0.0, 'loss_point_sparse': 0.2, "loss_ce_dense": 0.0, 'loss_point_dense': 0.2, } 3
        #weight_dict = {"loss_ce_sparse": 1.0, 'loss_point_sparse': 0.1, "loss_ce_dense": 0.0, 'loss_point_dense': 0.0, } #4
        #weight_dict = {"loss_ce_sparse": 0.0, 'loss_point_sparse': 0.0, "loss_ce_dense": 1.0, 'loss_point_dense': 0.1, }#5
        #weight_dict = {"loss_ce_sparse": 1.0, 'loss_point_sparse': 0.1, "loss_ce_dense": 1, 'loss_point_dense': 0.2, }#6
        #weight_dict = {"loss_ce_sparse": 1.0, 'loss_point_sparse': 0.3, "loss_ce_dense": 1, 'loss_point_dense': 0.1, }#7
        #weight_dict = {"loss_ce_sparse": 1, 'loss_point_sparse': 0.1, "loss_ce_dense": 1, 'loss_point_dense': 0.05, }#8AASEC2019_truth
        #weight_dict = {"loss_ce_sparse": 4.0, 'loss_point_sparse': 0.2, "loss_ce_dense": 2.0, 'loss_point_dense': 0.1, }#9groudtruth_weight
        matcher = MinCostMatcher(cost_class=0.5, cost_point=1)
        self.criterion = SetCriterion(cfg=1,
                                      num_classes=1,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      losses=losses)

        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)


    def forward(self, preds, _, p3_maps, p4_maps, p5_maps,polygons_area=None, lra_polys = None,**kwargs):

        assert isinstance(preds, list)

        clsass_sparse = []
        reg_points_sparse = []
        device = preds[0][0].device

        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):#转浮点
            gts[idx] = maps.float()
        
        if self.with_area_weight:#琛结论：生成一个1*15的张量
            assert polygons_area is not None
            max_num_polygon = max([len(p) for p in polygons_area])
            #print(len(polygons_area),polygons_area, max_num_polygon)      
            pad_polygon_areas = torch.zeros(len(polygons_area), max_num_polygon, device=device)
            for bi, po in enumerate(polygons_area):
                #print(bi,po)           
                if len(po) == 0:
                    continue
                pad_polygon_areas[bi, :len(po)] = torch.from_numpy(polygons_area[bi]).to(device)
                #print(pad_polygon_areas.size(),pad_polygon_areas)                
        else: 
            pad_polygon_areas = None
        gt_polygons_areas = [pad_polygon_areas] * 3
        down_sample_rates = self.steps

        
        #print(len(preds),len(gts),len(down_sample_rates),len(gt_polygons_areas))
        #print(len(preds[1]),preds[1][0].size(),preds[1][1].size(),preds[1][2].size(),preds[1][3].size())
        #print("——————————————————————————————配套分割线————————————————————————————————")
        losses = multi_apply(self.forward_single, preds, gts, down_sample_rates, gt_polygons_areas)#群体遍历（共三套）
        #print(len(losses),losses)
        #print("————————————————————————————————————————————————————————————————————————")
        loss_ce_dense = torch.tensor(0., device=device,requires_grad=True).float()
        loss_point_dense = torch.tensor(0.0, device=device,requires_grad=True).float()

        tr_train_masks = []

        for idx, data in enumerate(losses):
            if idx == 0:#琛 其实是三个数字表示loss
                loss_ce_dense = loss_ce_dense + sum(data)
            elif idx == 1:#琛 其实是三个数字表示loss
                loss_point_dense = loss_point_dense + sum(data)    
            elif idx == 2:
                #print(data[0].size(),data[1].size(),data[2].size())#琛 14400 3600 900个点，依次对应120、60、30三个特征图像素点
                tr_train_masks.append(data)
       
        #print(len(tr_train_masks))
        tr_train_masks = torch.cat(tr_train_masks[0], dim=1)#琛 相融合后18900个像素点
        #print(tr_train_masks.size())
        
        #print(len(preds),len(preds[0]),preds[0])
        #print(len(preds[2][3]),preds[2][3].size(),preds[2][3])
        
        for i in range(len(preds)):#琛 总共3*4张特征图 第一维遍历P3P4P5
            
            
            cls_sparse = preds[i][2][:,0,:,:][:, None]
            bs, _, h, w = cls_sparse.shape#琛 仍然是1*1*120*120 当i=0
            cls_sparse = cls_sparse.permute(0, 2, 3, 1).contiguous()#琛 交换通道 1*120*120*1 当i=0
            #print(cls_sparse.size())
            cls_sparse = cls_sparse.view(bs, h*w, -1)#琛 改变形状 1*14400*1 当i=0
            #print(cls_sparse.size())
            clsass_sparse.append(cls_sparse)
            
            reg_lra_sparse = preds[i][3].permute(0, 2, 3, 1).contiguous()#琛 交换通道 1*120*120*16 当i=0
            #print(reg_lra_sparse.size())
            reg_lra_sparse = reg_lra_sparse[:, :, :, :].view(-1, self.num_coefficients)#琛 改变形状 14400*16 其中coefficient=16 当i=0
            #print(reg_lra_sparse.size())  
            p_pre = torch.matmul(reg_lra_sparse,self.U_t.to(device))#琛 矩阵乘法 U_t形状16*28 结果14400*28 当i=0
            locations = self.generate_locations(bs, w, h, device)#琛 生成全场位置编号【0,0】【0,1】...【1,0】【1,1】...
            #print(p_pre.size(),p_pre)
            p_pre[:,0::2] += locations[:,0].unsqueeze(1)#对p_pre的偶数索引位置（即x坐标）进行更新，加上locations中的x坐标
            p_pre[:,1::2] += locations[:,1].unsqueeze(1)#对p_pre的奇数索引位置（即y坐标）进行更新，加上locations中的y坐标。
            #print(p_pre.size(),p_pre)
            p_pre = p_pre * self.steps[i]#琛 乘缩放因子8、16、32 当i=0,1,2
            #print(p_pre.size(),p_pre)
            reg_points_sparse.append(p_pre.view(bs, h*w, -1))

        reg_points_sparse = torch.cat(reg_points_sparse, dim=1).permute(0, 2, 1)#P3P4P5融合操作-》1*1*18900
        clsass_sparse = torch.cat(clsass_sparse, dim=1).permute(0, 2, 1)#1*28*18900
        #print(reg_points_sparse.size(),clsass_sparse.size())
        
        new_targets = []
        cnt = 0
        # print(len(lra_polys[0][40]),lra_polys[0][40])
        # print(len(lra_polys),len(lra_polys[0]),len(lra_polys[0][0]))
        
        for i in range(len(lra_polys)):#1*45*28 其中28为14个点坐标
            cnt += lra_polys[i].shape[0]#cnt+=45 当i=0
            if lra_polys[i].shape[0] == 0:
                clsass_sparse = torch.cat((clsass_sparse[:i, :, :], clsass_sparse[i+1:, : ,:]), dim=0)
                reg_points_sparse = torch.cat((reg_points_sparse[:i, :, :], reg_points_sparse[i+1:, : ,:]), dim=0)
                tr_train_masks = torch.cat((tr_train_masks[:i, :], tr_train_masks[i+1:, :]), dim=0)
                continue

            target = {}
            image_size = p3_maps.size(-1) * self.steps[0]#960
            gt_classes = torch.zeros(lra_polys[i].shape[0]).type(torch.int64).to(device)#45个0
            
            target["labels"] = gt_classes.to(device)
            target["image_size"] = image_size
            target['points'] =torch.from_numpy(lra_polys[i]).to(device)
            
            new_targets.append(target)

        output_sparse = {'pred_logits': clsass_sparse, 'pred_points': reg_points_sparse, 'tr_train_masks': tr_train_masks}
  
        if cnt > 0:
            loss_dict = self.criterion(output_sparse, new_targets)
        else:
            loss_dict = {'loss_ce_sparse': reg_points_sparse.sum()*0}
            loss_dict['loss_point_sparse'] = reg_points_sparse.sum()*0

        loss_dict['loss_ce_dense'] = loss_ce_dense 
        loss_dict['loss_point_dense'] = loss_point_dense   
        
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]#{'loss_ce_sparse': 2, 'loss_point_sparse': 0.1, 'loss_ce_dense': 1, 'loss_point_dense': 4}
        return loss_dict
    
    def forward_single(self, pred, gt, downsample_rate=None,areas=None):
        #pred是什么成分
        #print(len(gt),gt[0].size())#1 torch.Size([19, 120, 120])
        
        #print(len(pred),pred[0].size(),pred[1].size(),pred[2].size(),pred[3].size(),'pred_shape')
        #4 torch.Size([1, 1, 120, 120]) torch.Size([1, 16, 120, 120]) torch.Size([1, 1, 120, 120]) torch.Size([1, 16, 120, 120])
        cls_dense = pred[0].permute(0, 2, 3, 1).contiguous()
        
        reg_dense = pred[1].permute(0, 2, 3, 1).contiguous()
        
        gt = gt.permute(0, 2, 3, 1).contiguous()
        tr_pred = cls_dense[:, :, :, :1].view(-1).sigmoid()#[14400] 降成一维
        lra_pred = reg_dense[:, :, :, :].view(-1, self.num_coefficients)#14400*16 
        #print(lra_pred.size(),'lra_pred.size()')
        #print(tr_pred.size(),'tr_pred.size()')
       
        device = lra_pred.device 

        if self.with_area_weight:
            tr_mask_idx = gt[:, :, :, :1].long()
            tr_mask = (tr_mask_idx != 0).view(-1)
            batch_size, H, W, _ = tr_mask_idx.shape
            batch_idx = torch.arange(0, batch_size)[:, None, None].repeat(1, H, W).to(device)#这行代码什么成分
            batch_idx = batch_idx.view(-1)
            tr_mask_idx = tr_mask_idx.view(-1) - 1
        else:
            tr_mask = gt[:, :, :, :1].view(-1)

        tcl_mask = gt[:, :, :, 1:2].view(-1)#TCL（True Confidence Level）
        train_mask = gt[:, :, :, 2:3].view(-1)
        tps_map = gt[:, :, :, 3:].view(-1, self.num_coefficients)#TPS（Thin-Plate Spline）

        tr_train_mask = ((train_mask * tr_mask) > 0).float()
        # print(train_mask.size(),'train_maskkkk')
        # print(tr_mask.size(),'tr_maskkkkkkkk')
        # print(tr_train_mask.size(),'tr_maskkkkkkkk')

        bs,h,w,_ = gt.shape

        locations = self.generate_locations(bs, w, h, device)
        
        #计算分类损失
        loss_tr = self.ohem(tr_pred, tr_mask.float(), train_mask.long()) #text-region mask and effective mask
       
        pos_idx = torch.where(tr_train_mask > 0)[0]
        #print(downsample_rate, pos_idx.size())

        # regression loss

        loss_point = torch.tensor(0.,device=device,requires_grad=True).float()
        #print(self.with_area_weight,self.with_center_weight)都是TRUE
        
        num_pos = tr_train_mask.sum().item()
        if num_pos> 0:
            if self.with_area_weight:
                batch_idx = batch_idx[pos_idx]
                tr_mask_idx = tr_mask_idx[pos_idx]
            if self.with_center_weight:
                weight = (tr_mask[pos_idx].float() + tcl_mask[pos_idx].float()) / 2#均值
                weight = weight.contiguous()
            else:
                weight = torch.ones(num_pos,dtype=torch.float32, device=tps_map.device)#否则纯1

            if self.with_area_weight:
                pos_area = areas[batch_idx, tr_mask_idx] / downsample_rate**2
                num_instance = torch.sum(areas > 0)
                if num_instance == 0:#琛 隔4个epoch报错潜藏 无实例问题
                    #return loss_tr,  loss_point     
                    with open('chenchen.txt', 'a', encoding='utf-8') as file:  
   
                        file.write('123\n')
                    return loss_tr,  loss_point,  tr_train_mask.reshape(bs,-1)
                if torch.any(pos_area<=1):
                    pos_area[pos_area <=1] = INF
                area_weight = 1.0/pos_area
                weight = weight * area_weight * (1.0/num_instance)
            else:
                weight = weight * 1.0/pos_idx.shape[0]
            # print(num_pos)
            # print(weight.shape,weight)
            
            p_pre = torch.matmul(lra_pred,self.U_t.to(device))
            p_gt = torch.matmul(tps_map,self.U_t.to(device))
            p_gt[:,0::2] -= locations[:,0].unsqueeze(1)
            p_gt[:,1::2] -= locations[:,1].unsqueeze(1)#从p_gt中的坐标中减去相应的偏移量

            loss_point =  torch.sum(weight*F.smooth_l1_loss(
                p_gt[pos_idx],
                p_pre[pos_idx],
                reduction='none').mean(dim=-1))#平滑L1损失
        
            #loss_point = torch.sum(weight*(loss_point))
        # print(loss_point)
        # haha
        #print(downsample_rate, "tr_train_mask:",tr_train_mask.size(),tr_train_mask)
        return loss_tr, loss_point, tr_train_mask.reshape(bs,-1)


    def generate_locations(self, bs, w, h, device):#生成位置编号
        shifts_x = torch.arange(
            0, w, step=1,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h, step=1,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.repeat(bs,1) 
        return locations
    
    def ohem(self, predict, target, train_mask):
        #print(train_mask)纯1 
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()
        n_pos = pos.float().sum()#正样本数量

        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])#print
            n_neg = min(int(neg.float().sum().item()), int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

