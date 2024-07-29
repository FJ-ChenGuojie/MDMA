import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
#import data.dataprocess as dp
from data import dataprocess

class area_cluster:
    def __init__(self, data=None, area=None):
        if data is not None:
            self.area = [area]
            self.data = [data]
            self.distance = []
            self.time_index = []

        else:
            self.area = []
            self.data = []
            self.distance = []
            self.time_index = []

    def combine(self, cluster_2, distance, time_index):
        for index in range(len(cluster_2.area)):
            self.area.append(cluster_2.area[index])
            self.data.append(cluster_2.data[index])
        self.distance.append(distance)
        self.time_index.append(time_index)

def cluser_area(area_list, loss_type='mmd'):
    clusters_list = []
    clusters = []
    area_new_list = []

    ########init_clusters##########
    for i, data in enumerate(area_list):
        cluster = area_cluster(area=i, data=data)
        clusters.append(cluster)
    ###########################

    ############cluster_copy##############
    clusters_mid = []
    for index, cluster in enumerate(clusters):
        cluster_mid = area_cluster()
        cluster_mid.area = cluster.area.copy()
        for data in cluster.data:
            cluster_mid.data.append(data.clone())
        cluster_mid.distance = cluster.distance.copy()
        cluster_mid.time_index = cluster.time_index.copy()
        clusters_mid.append(cluster_mid)
    clusters_list.append(clusters_mid.copy())
    ###################################

    for i in range(len(area_list)-2):
        area_new_list.clear()
        for cluster in clusters:
            area_new_list.append(torch.cat(cluster.data))
        heat_matrix = dataprocess.area_dis(area_new_list, loss_type=loss_type)
        min_distance = np.inf
        i_index = 0
        j_index = 0
        for heat_i in range(0, heat_matrix.shape[0]-1):
            for heat_j in range(heat_i+1, heat_matrix.shape[0]):
                if heat_matrix[heat_i, heat_j] < min_distance:
                    min_distance = heat_matrix[heat_i, heat_j]
                    i_index = heat_i
                    j_index = heat_j
        clusters[i_index].combine(clusters[j_index], min_distance, i)
        del clusters[j_index]

        ############cluster_copy##############
        clusters_mid = []
        for index, cluster in enumerate(clusters):
            cluster_mid = area_cluster()
            cluster_mid.area = cluster.area.copy()
            for data in cluster.data:
                cluster_mid.data.append(data.clone())
            cluster_mid.distance = cluster.distance.copy()
            cluster_mid.time_index = cluster.time_index.copy()
            clusters_mid.append(cluster_mid)
        clusters_list.append(clusters_mid.copy())
        ###################################
    return clusters_list

