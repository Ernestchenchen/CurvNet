import os
import json
import numpy as np
from scipy.interpolate import splprep, splev
import torch
from scipy.sparse import csc_matrix
from scipy.special import comb as n_over_k
from numpy.linalg import norm
from sklearn.decomposition import IncrementalPCA


def resample_line2(line, n):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """

    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0
    #print(line)
    length_list = [
        norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
    ]
    #print(length_list)
    total_length = sum(length_list)
    length_cumsum = np.cumsum([0.0] + length_list)
    delta_length = total_length / (float(n) + 1e-8)

    current_edge_ind = 0
    resampled_line = [line[0]]

    for i in range(1, n):
        current_line_len = i * delta_length
        
        #print(len(length_cumsum),length_cumsum,current_edge_ind + 1,current_line_len)
        
        while current_line_len >= length_cumsum[current_edge_ind + 1]:
            current_edge_ind += 1
        current_edge_end_shift = current_line_len - length_cumsum[
            current_edge_ind]
        end_shift_ratio = current_edge_end_shift / length_list[
            current_edge_ind]
        current_point = line[current_edge_ind] + (
            line[current_edge_ind + 1] -
            line[current_edge_ind]) * end_shift_ratio
        resampled_line.append(current_point)

    resampled_line.append(line[-1])
    resampled_line = np.array(resampled_line)

    return resampled_line


def resample_polygon(top_line,bot_line, n=None):

    resample_line = []
    for polygon in [top_line, bot_line]:
        if polygon.shape[0] >= 3:
            x,y = polygon[:,0], polygon[:,1]
            tck, u = splprep([x, y], k=3 if polygon.shape[0] >=5 else 2, s=0)
            u = np.linspace(0, 1, num=n, endpoint=True)
            out = splev(u, tck)
            new_polygon = np.stack(out, axis=1).astype('float32')
        else:
            #print(polygon,n)
            #以下是袁怡琛为了避免同一点差值失败自制的特值判别模块
            if polygon[0][0]-polygon[1][0] < 0.00000001 and polygon[0][1]-polygon[1][1]<0.00000001:
                polygon[1][0]=polygon[0][0]+1
                polygon[1][1]=polygon[0][1]+1
            #琛
            new_polygon = resample_line2(polygon, n-1)

        resample_line.append(np.array(new_polygon))

    resampled_polygon = np.concatenate([resample_line[0], resample_line[1]]).flatten()
    resampled_polygon = np.expand_dims(resampled_polygon,axis=0)
    return resampled_polygon




def reorder_poly_edge2(points):
    """Get the respective points composing head edge, tail edge, top
    sideline and bottom sideline.

    Args:
        points (ndarray): The points composing a text polygon.

    Returns:
        head_edge (ndarray): The two points composing the head edge of text
            polygon.
        tail_edge (ndarray): The two points composing the tail edge of text
            polygon.
        top_sideline (ndarray): The points composing top curved sideline of
            text polygon.
        bot_sideline (ndarray): The points composing bottom curved sideline
            of text polygon.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    orientation_thr=2.0
    head_inds, tail_inds = find_head_tail(points,
                                                orientation_thr)
    head_edge, tail_edge = points[head_inds], points[tail_inds]

    pad_points = np.vstack([points, points])
    if tail_inds[1] < 1:
        tail_inds[1] = len(points)
    sideline1 = pad_points[head_inds[1]:tail_inds[1]]
    sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
    sideline_mean_shift = np.mean(
        sideline1, axis=0) - np.mean(
            sideline2, axis=0)

    if sideline_mean_shift[1] > 0:
        top_sideline, bot_sideline = sideline2, sideline1
    else:
        top_sideline, bot_sideline = sideline1, sideline2

    return head_edge, tail_edge, top_sideline, bot_sideline
    

def vector_angle(vec1, vec2):
    if vec1.ndim > 1:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
    if vec2.ndim > 1:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
    return np.arccos(
        np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))

def vector_sin(vec):
    assert len(vec) == 2
    return vec[1] / (norm(vec) + 1e-8)

def vector_cos(vec):
    assert len(vec) == 2
    return vec[0] / (norm(vec) + 1e-8)

def find_head_tail(points, orientation_thr):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    assert isinstance(orientation_thr, float)

    if len(points) > 4:
        pad_points = np.vstack([points, points[0]])
        edge_vec = pad_points[1:] - pad_points[:-1]

        theta_sum = []
        adjacent_vec_theta = []
        for i, edge_vec1 in enumerate(edge_vec):
            adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
            adjacent_edge_vec = edge_vec[adjacent_ind]
            temp_theta_sum = np.sum(
                vector_angle(edge_vec1, adjacent_edge_vec))
            temp_adjacent_theta = vector_angle(
                adjacent_edge_vec[0], adjacent_edge_vec[1])
            theta_sum.append(temp_theta_sum)
            adjacent_vec_theta.append(temp_adjacent_theta)
        theta_sum_score = np.array(theta_sum) / np.pi
        adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
        poly_center = np.mean(points, axis=0)
        edge_dist = np.maximum(
            norm(pad_points[1:] - poly_center, axis=-1),
            norm(pad_points[:-1] - poly_center, axis=-1))
        dist_score = edge_dist / np.max(edge_dist)
        position_score = np.zeros(len(edge_vec))
        score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
        score += 0.35 * dist_score
        if len(points) % 2 == 0:
            position_score[(len(score) // 2 - 1)] += 1
            position_score[-1] += 1
        score += 0.1 * position_score
        pad_score = np.concatenate([score, score])
        score_matrix = np.zeros((len(score), len(score) - 3))
        x = np.arange(len(score) - 3) / float(len(score) - 4)
        gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
            (x - 0.5) / 0.5, 2.) / 2)
        gaussian = gaussian / np.max(gaussian)
        for i in range(len(score)):
            score_matrix[i, :] = score[i] + pad_score[
                (i + 2):(i + len(score) - 1)] * gaussian * 0.3

        head_start, tail_increment = np.unravel_index(
            score_matrix.argmax(), score_matrix.shape)
        tail_start = (head_start + tail_increment + 2) % len(points)
        head_end = (head_start + 1) % len(points)
        tail_end = (tail_start + 1) % len(points)

        if head_end > tail_end:
            head_start, tail_start = tail_start, head_start
            head_end, tail_end = tail_end, head_end
        head_inds = [head_start, head_end]
        tail_inds = [tail_start, tail_end]
    else:
        if vector_slope(points[1] - points[0]) + vector_slope(
                points[3] - points[2]) < vector_slope(
                    points[2] - points[1]) + vector_slope(points[0] -
                                                                points[3]):
            horizontal_edge_inds = [[0, 1], [2, 3]]
            vertical_edge_inds = [[3, 0], [1, 2]]
        else:
            horizontal_edge_inds = [[3, 0], [1, 2]]
            vertical_edge_inds = [[0, 1], [2, 3]]

        vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                points[vertical_edge_inds[0][1]]) + norm(
                                    points[vertical_edge_inds[1][0]] -
                                    points[vertical_edge_inds[1][1]])
        horizontal_len_sum = norm(
            points[horizontal_edge_inds[0][0]] -
            points[horizontal_edge_inds[0][1]]) + norm(
                points[horizontal_edge_inds[1][0]] -
                points[horizontal_edge_inds[1][1]])

        if vertical_len_sum > horizontal_len_sum * orientation_thr:
            head_inds = horizontal_edge_inds[0]
            tail_inds = horizontal_edge_inds[1]
        else:
            head_inds = vertical_edge_inds[0]
            tail_inds = vertical_edge_inds[1]

    return head_inds, tail_inds



def clockwise(head_edge, tail_edge, top_sideline, bot_sideline):
    hc = head_edge.mean(axis=0)
    tc = tail_edge.mean(axis=0)
    d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
    dx = np.abs(hc[0] - tc[0])
    if not dx / d <= 1:
        print(dx / d)
    angle = np.arccos(dx / d)
    PI = 3.1415926
    direction = 0 if angle <= PI / 4 else 1  # 0 horizontal, 1 vertical
    if top_sideline[0, direction] > top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
    else:
        top_indx = np.arange(0, top_sideline.shape[0])
    top_sideline = top_sideline[top_indx]
    if direction == 1 and top_sideline[0, direction] < top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
        top_sideline = top_sideline[top_indx]

    if bot_sideline[0, direction] > bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
    else:
        bot_indx = np.arange(0, bot_sideline.shape[0])
    bot_sideline = bot_sideline[bot_indx]
    if direction == 1 and bot_sideline[0, direction] < bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
        bot_sideline = bot_sideline[bot_indx]
    if top_sideline[:, 1 - direction].mean() > bot_sideline[:, 1 - direction].mean():
        top_sideline, bot_sideline = bot_sideline, top_sideline

    return top_sideline, bot_sideline, direction
    

def reorder_poly_edge(points):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2

    head_edge, tail_edge, top_sideline, bot_sideline = reorder_poly_edge2(points)
    top_sideline, bot_sideline,_ = clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
   
    return top_sideline, bot_sideline[::-1]



anno = json.load(open("/mnt/data/experiments/datasets/spinal-AI2024/spinal_AI2024_train.json"))['annotations']
resample_lines = []

# 定义文件名和路径  

# chen_filename = '/mnt/data/experiments/datasets/spinal-AI2024/spinal_AI2024_all_stage6_discard.txt'  

  
# # 打开文件并读取所有行  
# with open(chen_filename, 'r') as file:  
#     # 使用列表推导式读取每一行并转换为整数，然后存储到列表中  
#     numbers = [int(line.strip()) for line in file]  
  
# 输出列表以验证  


for i in range(len(anno)):
    
    # if anno[i]['image_id'] in numbers:
    #     continue #琛琛设计的筛选机制
    # if anno[i]['category_id']== 0:
    #     continue
   
    
    anno_i = anno[i]['segmentation'][0]
    polygon = np.array(anno_i).reshape(-1,2).astype(np.float32)

    if polygon.shape[0] % 2 !=0:
        continue
    #print(polygon)
       
    top_sideline, bot_sideline = reorder_poly_edge(polygon)
    #print(top_sideline.shape)
    #print(bot_sideline.shape)
    resample_line = resample_polygon(top_sideline, bot_sideline, 7)
    resample_lines.append(resample_line)


resample_lines = np.concatenate(resample_lines,axis=0)

n_components = 16
pca = IncrementalPCA(n_components=n_components, copy=False)
pca.fit(resample_lines)
components_c = pca.components_.astype(np.float32)
output_path = os.path.join('./eigenanchors/pca_spinal_AI2024_train_' + str(n_components) 
                            + '.npz')
print("Save the pca matrix: " + output_path)
np.savez(output_path,
            components_c=components_c,)
