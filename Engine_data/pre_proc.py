import cv2
import torch
# from draw_gaussian import *
# import transform
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
def processing_test(image, input_h, input_w):
    image = cv2.resize(image, (input_w, input_h))
    # 将图像缩放到输入图像的大小
    out_image = image.astype(np.float32) / 255.
    # 将图像转换为float32类型
    out_image = out_image - 0.5
    # 将图像减去0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    # 将图像转换为tensor
    out_image = torch.from_numpy(out_image)
    return out_image


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    # 在图像上绘制出4个点
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        # 在图像上添加文字，文字内容为i+1
        cv2.putText(out_image, '{}'.format(i+1), (int(pts[i, 0]), int(pts[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0),1,1)
    # 在图像上绘制出4个线段
    for i,j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(out_image,
                 (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])),
                 color=colors[i], thickness=1, lineType=1)
    return out_image


def rearrange_pts(pts):
    # rearrange left right sequence
    boxes = []
    centers = []
    # 初始化pts数组，每一个元素为一个点的坐标
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        # 对pts_4进行排序，按照x坐标升序排序
        x_inds = np.argsort(pts_4[:, 0])
        # 将pts_4中的x_inds[:2]和x_inds[2:]按照y坐标升序排序
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        # 将pt_l和pt_r中的y_inds_l[0]和y_inds_r[0]按照x坐标升序排序
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        # 将pt_l和pt_r中的第一个元素放入tl，第二个元素放入tr，第三个元素放入bl，第四个元素放入br中
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
        # centers.append(np.mean(pts_4, axis=0))
        centers.append(np.mean(pts_4, axis=0))
    # 将boxes和centers数组按照y坐标升序排序
    bboxes = np.asarray(boxes, np.float32)
    # rearrange top to bottom sequence
    # 将top to bottom sequence
    centers = np.asarray(centers, np.float32)
    # 对centers数组进行排序，按照y坐标升序排序
    sort_tb = np.argsort(centers[:,1])
    # 初始化新的bboxes数组
    new_bboxes = []
    # 遍历sort_tb中的每一个元素
    for sort_i in sort_tb:
        # 将bboxes中的4*sort_i，4*sort_i+1，4*sort_i+2，4*sort_i+3中的元素放入新的bboxes中
        new_bboxes.append(bboxes[4*sort_i, :])
        new_bboxes.append(bboxes[4*sort_i+1, :])
        new_bboxes.append(bboxes[4*sort_i+2, :])
        new_bboxes.append(bboxes[4*sort_i+3, :])
    # 将新的bboxes数组返回
    new_bboxes = np.asarray(new_bboxes, np.float32)
    # 返回新的bboxes
    return new_bboxes


def generate_ground_truth(image, pts_2, image_h, image_w, img_id):
    #print(image.shape)
    #print(image_w,image_h)
    #print(pts_2[point,1]), int(pts_2[point,0])
    #print(pts_2)
    hm = np.zeros((68, image_h, image_w), dtype=np.float32) 
    # wh = np.zeros((17, 2*4), dtype=np.float32) 
    # reg = np.zeros((17, 2), dtype=np.float32)  
    # ind = np.zeros((17), dtype=np.int64)
    # reg_mask = np.zeros((17), dtype=np.uint8)
    #heatmap=[]
    for point in range(68):
        
        #print(int(pts_2[point,0]), int(pts_2[point,1]))
        hm [point, int(pts_2[point,1])-1, int(pts_2[point,0])-1]=1
        
    
        
    #print(hm.shape)
    # if pts_2.shape[0]!=68:
    #     print('ATTENTION!! image {} pts does not equal to 68!!! '.format(img_id))
    
    # for k in range(17):
    #     pts = pts_2[4*k:4*k+4,:] 
    #     # bbox_h = np.mean([np.sqrt(np.sum((pts[0,:]-pts[2,:])**2)),
    #     #                 np.sqrt(np.sum((pts[1,:]-pts[3,:])**2))])
    #     # bbox_w = np.mean([np.sqrt(np.sum((pts[0,:]-pts[1,:])**2)),
    #     #                 np.sqrt(np.sum((pts[2,:]-pts[3,:])**2))])
    #     # cen_x, cen_y = np.mean(pts, axis=0)
    #     # ct = np.asarray([cen_x, cen_y], dtype=np.float32)
    #     # ct_int = ct.astype(np.int32)
    #     # radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
    #     # radius = max(0, int(radius))
    #     for i in range (4):
    #         #print(pts[i,0], pts[i,1])
    #         one_point = np.asarray([pts[i,0], pts[i,1]], dtype=np.float32)
    #         one_point_int = one_point.astype(np.int32)
    #         heatmap=draw_umich_gaussian(hm[4*k+i,:,:], one_point_int, radius=1)
            #heatmap=draw_umich_gaussian(hm[0,:,:], one_point_int, radius=1)
        #print("hm:",points_num(hm))
        #print("heatmap",points_num(heatmap))
        #cv2.imwrite("heatmap1.jpg",heatmap)
    #     ind[k] = ct_int[1] * image_w + ct_int[0] 
    #     reg[k] = ct - ct_int
    #     reg_mask[k] = 1
    #     for i in range(4):
    #         wh[k,2*i:2*i+2] = ct-pts[i,:]
    
    # plt.imshow(heatmap, cmap='hot')
    # plt.colorbar()
    # plt.savefig("heatmap4.jpg")
    
    # ret = {'input': image,
    #         'hm': hm,
    #         'ind': ind,
    #         'reg': reg,
    #         'wh': wh,
    #         'reg_mask': reg_mask,
    #         }
    #print("image_shape",image.shape)
    #print("hm_shape",hm.shape)
    ones_count = np.sum(hm == 1).item()
    if ones_count!=68:  
        print("热图有问题-hm_shape",hm.shape,hm)
    
    ground_truth = {'input': image,
            'heatmap': hm,
            }
    
    return ground_truth
def points_num(arr):
       
    return np.count_nonzero(arr)
        
# def filter_pts(pts, w, h):
#     pts_new = []
#     for pt in pts:
#         if any(pt) < 0 or pt[0] > w - 1 or pt[1] > h - 1:
#             continue
#         else:
#             pts_new.append(pt)
#     return np.asarray(pts_new, np.float32)


def processing_train(image, pts, image_h, image_w, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    #h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    #print("处理前的点：",pts,img_id)
    data_aug = {'train': transform.Compose([transform.Resize_image(h=image_h, w=image_w),
                                            transform.ConvertImgFloat(),
                                            transform.PhotometricDistort(),
                                            #transform.RandomMirror_w(),
                                            transform.Expand(max_scale=1.5, mean=(0, 0, 0)),                 
                                            transform.Resize_image_and_point(h=image_h//down_ratio, w=image_w//down_ratio)
                                            ]),
                'val': transform.Compose([transform.Resize_image(h=image_h, w=image_w),
                                          transform.ConvertImgFloat(),
                                          transform.Resize_image_and_point(h=image_h//down_ratio, w=image_w//down_ratio)
                                          ])}
    
    if aug_label:
        out_image, pts = data_aug['train'](image.copy(), pts)
    else:
        out_image, pts = data_aug['val'](image.copy(), pts)
    #print("处理后的点：",pts[15],img_id)
    #print("预处理前的点：",pts)
    #out_image=image
    # print(out_image.shape)
    # print("哈哈：",out_image[out_image<0])
    # out_image = np.clip(out_image, a_min=0., a_max=255.)
    # print('after',out_image)
    # print(
    #     'out_image:',out_image.shape,
    # )
    out_image = np.transpose(out_image / 255. - 0.5, (2,0,1))
    pts = rearrange_pts(pts)
   
    #pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)
    #print("处理后的点：",pts,img_id)
    #print(pts2)
    #@print(out_image.shape)
    return np.asarray(out_image, np.float32), pts

