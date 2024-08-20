###########################################################################################
## This code is transfered from matlab version of the MICCAI challenge
## Oct 1 2019
###########################################################################################
import numpy as np
import cv2


def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    #print("以下是判断脊椎性状区间：")
    ll = []
    num = mid_p_v.shape[0]
    #print(num)
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    #print(ll)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    #print(ll)
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    #print(ll_pair)
    # print(ll_pair.shape)
    # print("hahahahaah")
    if ll_pair.shape[0] == 0:
        a=0
        b=0
    else:
        a = sum(sum(ll_pair))
    
        b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True

def cobb_angle_calc(pts, image):
    pts = np.asarray(pts, np.float32)   # 68 x 2
    #print(pts)
    h,w,c = image.shape
    num_pts = pts.shape[0]   # number of points, 68
    vnum = num_pts//4-1

    #print(vnum) #16

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2
    #print("中点坐标：",mid_p)
    for pt in mid_p:
        cv2.circle(image,
                   (int(pt[0]), int(pt[1])),
                   4, (0,255,255), -1, 1)

    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(image,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,0,255),
                 thickness=2, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    #print("17个向量vec_m：",vec_m)
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    #print("每个向量各自的模mod_v",mod_v)
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    #print("每个向量与其他向量的点积mod_v：",mod_v)
    # cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    cosine_angles = np.zeros_like(dot_v)
    nonzero_mask = mod_v != 0.0
    cosine_angles[nonzero_mask] = np.clip(dot_v[nonzero_mask] / mod_v[nonzero_mask], a_min=0., a_max=1.)
    cosine_angles[np.logical_not(nonzero_mask)] = 0
    #print("每个向量与其他向量的余弦值cosine_angles：",cosine_angles)
    angles = np.arccos(cosine_angles)   # 17 x 17
    
    #print("每个向量与其他向量的角度angles：",angles)
    pos1 = np.argmax(angles, axis=1)
    # print("每个向量与其他向量最大角度的下标数组pos1:",pos1)
    maxt = np.amax(angles, axis=1)
    # print("每个向量与其他向量最大角度数组pos1:",maxt)
    pos2 = np.argmax(maxt)
    #print(maxt)
    # print("全局最大角度下标pos2:",pos2)
    cobb_angle1 = np.amax(maxt)
    #print("全局最大角度cobb_angle1:",cobb_angle1)
    cobb_angle1 = cobb_angle1/np.pi*180
    #print(cobb_angle1)
    # print("全局最大角度值",cobb_angle1)
    flag_s = is_S(mid_p_v)
    if not flag_s: # not S
        #print('Not S')
        cobb_angle2 = angles[0, pos2]/np.pi*180
        # print("角度值cobb_angle2:",cobb_angle2)
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        # print("角度值cobb_angle3:",cobb_angle3)
        cv2.line(image,
                 (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
                 (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=2, lineType=2)
        cv2.line(image,
                 (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                 (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=2, lineType=2)
        cobb_angle1=round(cobb_angle1,2)
        cobb_angle2=round(cobb_angle2,2)
        cobb_angle3=round(cobb_angle3,2)
        #print("琛琛提示：首节到第{}节角度为{}度；第{}节到第{}节角度为{}度；第{}节到尾节角度为{}度；仅有一个角度".format(pos2,cobb_angle2,pos2,pos1[pos2],cobb_angle1,pos1[pos2],cobb_angle3))
        ans=[0,cobb_angle2,int(pos2),cobb_angle1,int(pos1[pos2]),cobb_angle3,17]
        ans_type=flag_s
        #print(ans)
    else:
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            #print('Is S: condition1曲线型且最大角度在中部：')
            angle2 = angles[pos2,:(pos2+1)]
            # print("根据pos2取到angle2:",angle2)
            cobb_angle2 = np.max(angle2)
            # print("angle2中最大值cobb_angle2:",cobb_angle2)
            pos1_1 = np.argmax(angle2)
            # print("angle2中最大值下标pos1_1:",pos1_1)
            cobb_angle2 = cobb_angle2/np.pi*180
            # print("角度值cobb_angle2:",cobb_angle2)
            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            # print("根据pos1，pos2取到angle3:",angle3)
            cobb_angle3 = np.max(angle3)
            # print("angle3中最大值cobb_angle3:",cobb_angle3)
            pos1_2 = np.argmax(angle3)
            # print("angle3中最大值下标pos1_2:",pos1_2)
            cobb_angle3 = cobb_angle3/np.pi*180
            # print("角度值cobb_angle3:",cobb_angle3)
            pos1_2 = pos1_2 + pos1[pos2]-1
            # print("pos1_2更新:",pos1_2)

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos2 * 2, 0]), int(mid_p[pos2 * 2, 1])),
                     (int(mid_p[pos2 * 2+1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                     (int(mid_p[pos1[pos2] * 2+1, 0]), int(mid_p[pos1[pos2]* 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cobb_angle1=round(cobb_angle1,2)
            cobb_angle2=round(cobb_angle2,2)
            cobb_angle3=round(cobb_angle3,2)
            #print("琛琛提示：第{}节到第{}节角度为{}度；第{}节到第{}节角度为{}度；第{}节到第{}节角度为{}度；最大角度发生中间段".format(pos1_1,pos2,cobb_angle2,pos2,pos1[pos2],cobb_angle1,pos1[pos2],pos1_2,cobb_angle3))
            ans=[int(pos1_1),cobb_angle2,int(pos2),cobb_angle1, int(pos1[pos2]),cobb_angle3, int(pos1_2)]
            ans_type=str(flag_s)+"_mid"
            #print(ans)
        else:
            #print('Is S: condition2多段曲折：')
            angle2 = angles[pos2,:(pos2+1)]
            # print("根据pos2取到angle2:",angle2)
            cobb_angle2 = np.max(angle2)
            # print("angle2中最大值cobb_angle2:",cobb_angle2)
            pos1_1 = np.argmax(angle2)
            # print("angle2中最大值下标pos1_1:",pos1_1)
            cobb_angle2 = cobb_angle2/np.pi*180
            # print("角度值cobb_angle2:",cobb_angle2)
            angle3 = angles[pos1_1, :(pos1_1+1)]
            # print("根据pos1_1取到angle3:",angle3)
            cobb_angle3 = np.max(angle3)
            # print("angle3中最大值cobb_angle3:",cobb_angle3)
            pos1_2 = np.argmax(angle3)
            # print("angle3中最大值下标pos1_2:",pos1_2)
            cobb_angle3 = cobb_angle3/np.pi*180
            # print("角度值cobb_angle3:",cobb_angle3)

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos2 * 2, 0]), int(mid_p[pos2 * 2, 1])),
                     (int(mid_p[pos2 * 2+1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                     (int(mid_p[pos1[pos2] * 2+1, 0]), int(mid_p[pos1[pos2]* 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=2, lineType=2)
            cobb_angle1=round(cobb_angle1,2)
            cobb_angle2=round(cobb_angle2,2)
            cobb_angle3=round(cobb_angle3,2)
            #print("琛琛提示：第{}节到第{}节角度为{}度；第{}节到第{}节角度为{}度；第{}节到第{}节角度为{}度；最大角度发生下部".format(pos1_2,pos1_1,cobb_angle3,pos1_1,pos2,cobb_angle2,pos2,pos1[pos2],cobb_angle1))
            ans=[int(pos1_2),cobb_angle3,int(pos1_1),cobb_angle2,int(pos2),cobb_angle1,int(pos1[pos2])]
            #print(ans)
            ans_type=str(flag_s)+"_dou"
    #print(ans)
      
    return ans,ans_type