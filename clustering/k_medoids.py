import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import data.dataprocess as dp
from base.loss_transfer import TransferLoss

def k_medoids_clustering(area_list, cluster_center, loss_type='mmd'):
    cluster_list = []
    criterion_transder = TransferLoss(loss_type=loss_type, input_dim=area_list[0].shape[-1])
    for i in range(len(cluster_center)):
        cluster_list.append([])

    for i in range(len(area_list)):
        distance = []
        for ii in range(len(cluster_center)):
            distance.append(criterion_transder.compute(torch.tensor(area_list[i], dtype=torch.float32), torch.tensor(area_list[cluster_center[ii]], dtype=torch.float32)).item())
        index = distance.index(min(distance))
        cluster_list[index].append(i)
    for cluster in cluster_list:
        pass
    return cluster_list

def update_clustering_center(cluster_list, area_list, loss_type='mmd'):
    criterion_transder = TransferLoss(loss_type=loss_type, input_dim=area_list[0].shape[-1])

    index_center = -1
    cluster_center = []
    for i in cluster_list:
        min_distance = np.inf
        for ii in i:
            distance = criterion_transder.compute(area_list[ii].float(), torch.cat([area_list[iii] for iii in i]).float())
            if distance <= min_distance:
                index_center = ii
                min_distance = distance
        cluster_center.append(index_center)
    return cluster_center

def area_clustering_sequence(feat_list, loss_type='mmd', K=2):
    cluster_center = [30, 49, 3, 11]#3, 11
    index_list = []
    containers_list = []
    # for i in range(K):
    #     cluster_center.append(i)

    for feat_i in feat_list:
        feat_i = feat_i.reshape(feat_i.shape[0], -1)
        for ii in range(feat_i.shape[0]):
            if feat_i[ii].sum() > 0:
                index_list.append(ii)
        containers_list.append(torch.tensor(feat_i)[index_list])
        index_list.clear()

    for i in range(100):
        cluster_list = k_medoids_clustering(containers_list, cluster_center, loss_type)
        update_center = update_clustering_center(cluster_list, containers_list, loss_type='mmd')
        if update_center == cluster_center:
            break
        cluster_center = copy.deepcopy(update_center)

    return cluster_list

def area_clustering(feat_list, loss_type='mmd', K=2):
    cluster_center = []
    index_list = []
    containers_list = []
    for i in range(K):
        cluster_center.append(i)

    for feat_i in feat_list:
        feat_i = feat_i.reshape(-1, feat_i.shape[-1])
        for ii in range(feat_i.shape[0]):
            if feat_i[ii].sum() > 0:
                index_list.append(ii)
        containers_list.append(torch.tensor(feat_i)[index_list])
        index_list.clear()

    for i in range(100):
        cluster_list = k_medoids_clustering(containers_list, cluster_center, loss_type)
        update_center = update_clustering_center(cluster_list, containers_list, loss_type='mmd')
        if update_center == cluster_center:
            break
        cluster_center = copy.deepcopy(update_center)

    return cluster_list
