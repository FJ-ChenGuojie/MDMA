import os
import struct

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import pandas as pd
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from base.loss_transfer import TransferLoss
from params import args
import clustering.hierarchical as hierarchical

IYC_CSZ_CSIZECD = {'20': 0, '40': 1, '45': 2}
IYC_CTYPECD = {'BU': 0, 'FR': 1, 'GP': 2, 'HC': 3, 'OT': 4, 'PF': 5, 'TK': 6, 'TU': 7, 'RF': 8}
IYC_CHEIGHTCD = {'HQ': 0, 'LQ': 1, 'MQ': 2, 'PQ': 3}
IYC_CWEIGHT = {'1': 0, '2': 1, '3': 2, '4': 3}
IYC_STS_CSTATUSCD = {'CF': 0, 'EE': 1, 'IE': 2, 'IF': 3, 'IZ': 4, 'NF': 5, 'OE': 6, 'OF': 7, 'OZ': 8, 'RE': 9, 'RF': 10, 'T': 11, 'TE': 12, 'ZE': 13}
vocab = [IYC_CSZ_CSIZECD, IYC_CTYPECD, IYC_CHEIGHTCD, IYC_CWEIGHT, IYC_STS_CSTATUSCD]


def bay_data(file_dir):
    file_list = os.listdir(file_dir)
    label_dir = file_dir + '/' + file_list[0]
    seq_dir = file_dir + '/' + file_list[2]
    label_dir_list_old = os.listdir(label_dir)
    seq_dir_list_old = os.listdir(seq_dir)
    label_list = []
    label_mask_list = []
    feat_list = []
    time_list = []
    location_list = []
    dict_label = {}

    seq_dir_list = []
    label_dir_list = []
    print(len(label_dir_list_old))
    for i in range(1, len(label_dir_list_old) + 1):
        for j in range(0, len(label_dir_list_old)):
            if int(label_dir_list_old[j][:-4][10:]) == i:
                label_dir_list.append(label_dir_list_old[j])
                # print(label_dir_list_old[j])

    print(len(seq_dir_list_old))
    for i in range(1, len(seq_dir_list_old) + 1):
        for j in range(0, len(seq_dir_list_old)):
            if int(seq_dir_list_old[j][:-4][10:]) == i:
                seq_dir_list.append(seq_dir_list_old[j])
                # print(seq_dir_list_old[j])

    for index, label_file in enumerate(label_dir_list):
        label, label_mask = labelprocess(label_dir + '/' + label_file)
        label_list.append(label)
        label_mask_list.append(label_mask)
        dict_label[int(label_file[:-4][10:])] = index

    for index, seq_file in enumerate(seq_dir_list):
        print(seq_file)
        feat, time, location = seq_convert(seq_dir + '/' + seq_file, vocab)
        feat_list.append(feat)
        time_list.append(time)
        location_list.append(location)
        dict_label[int(seq_file[:-4][10:])] = index


    return feat_list, time_list, location_list, label_list, label_mask_list

def bay_data_2(file_dir):
    seq_dir = file_dir + '/' + '3Area_48_zdxqdatatest.csv'
    label_dir = file_dir + '/' + '1Area_48_zdxqdatatest.csv'
    label, label_mask = labelprocess(label_dir)
    feat_list = []
    time_list = []
    location_list = []
    label_list = []
    label_mask_list = []

    feat, time, location = seq_convert(seq_dir, vocab)
    for i in range(1, 62):
        area_list = []
        for ii in range(len(feat)):
            if location[ii, -1, 0] == i:
                area_list.append(ii)
        random.seed(1)
        random.shuffle(area_list)
        feat_list.append(feat[area_list])
        time_list.append(time[area_list])
        location_list.append(location[area_list])
        label_list.append(label[area_list])
        label_mask_list.append(label_mask[area_list])

    ############reserve squence#############
    for i in range(len(feat_list)):
        for ii in range(len(feat_list[i])):
            mid_feature = np.zeros((feat_list[i].shape[1], feat_list[i].shape[2]))
            mid_time = np.zeros((time_list[i].shape[1], time_list[i].shape[2]))
            mid_location = np.zeros((location_list[i].shape[1], location_list[i].shape[2]))
            for iii in range(5):
                mid_feature[5 - iii - 1] = feat_list[i][ii, iii]
                mid_time[5 - iii - 1] = time_list[i][ii, iii]
                mid_location[5 - iii - 1] = location_list[i][ii, iii]
            feat_list[i][ii] = mid_feature
            time_list[i][ii] = mid_time
            location_list[i][ii] = mid_location
    #####################################

    return feat_list[1:], time_list[1:], location_list[1:], label_list[1:], label_mask_list[1:]

def data_covert(feat_list, time_list, location_list, label_list, label_mask_list):#en:[00012] de:[3] en:[00001] de:[2]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[20, 16, 12, 8, 4]].sum())
            en_feat = np.zeros((5, 34))
            en_time = np.zeros((5, 4))
            en_location = np.zeros((5, 4))
            for index in range(1, len_seq):
                en_feat[0:index] = np.ones((index, 34))
                en_time[0:index] = np.ones((index, 4))
                en_location[0:index] = np.ones((index, 4))
                list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
                list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
                list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
                list_feat_de.append((feat_list[i][ii][index].reshape(1, 1, -1)))
                list_time_de.append((time_list[i][ii][index].reshape(1, 1, -1)))
                list_location_de.append((location_list[i][ii][index].reshape(1, 1, -1)))
                list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
                list_mask.append(index)
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_1(feat_list, time_list, location_list, label_list, label_mask_list):#en:[00012] de:[00300] en:[00001] de:[00020]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[20, 16, 12, 8, 4]].sum())
            en_feat = np.zeros((5, 34))
            en_time = np.zeros((5, 4))
            en_location = np.zeros((5, 4))
            #de_feat = np.zeros((5, 34))
            #de_time = np.zeros((5, 4))
            #de_location = np.zeros((5, 4))
            for index in range(1, len_seq):
                en_feat[0:index + 1] = np.ones((index + 1, 34))
                en_time[0:index + 1] = np.ones((index + 1, 4))
                en_location[0:index+1] = np.ones((index+1, 4))

                feat = en_feat*feat_list[i][ii]
                ts = en_time*time_list[i][ii]
                location = en_location*location_list[i][ii]
                list_label.append(1 - label_list[i][ii][[20, 16, 12, 8, 4]][index])
                list_mask.append(index)

                en_feat_mid = np.zeros((5, 34))
                en_time_mid = np.zeros((5, 4))
                en_location_mid = np.zeros((5, 4))
                de_feat_mid = np.zeros((5, 34))
                de_time_mid = np.zeros((5, 4))
                de_location_mid = np.zeros((5, 4))
                for iii in range(0, index):
                    en_feat_mid[iii+5-index] = feat[iii]
                    en_time_mid[iii+5-index] = ts[iii]
                    en_location_mid[iii+5-index] = location[iii]
                list_feat_en.append(en_feat_mid.reshape(1, 5, 34))
                list_time_en.append(en_time_mid.reshape(1, 5, 4))
                list_location_en.append(en_location_mid.reshape(1, 5, 4))

                for iii in range(0, index+1):
                    de_feat_mid[iii + 4 - index] = feat[iii]
                    de_time_mid[iii + 4 - index] = ts[iii]
                    de_location_mid[iii + 4 - index] = location[iii]

                list_feat_de.append(de_feat_mid.reshape(1, 5, 34))
                list_time_de.append(de_time_mid.reshape(1, 5, 4))
                list_location_de.append(de_location_mid.reshape(1, 5, 4))
                pass
            # for index in range(1, len_seq):
            #     en_feat[0:index+1] = np.ones((index+1, 34))
            #     en_time[0:index+1] = np.ones((index+1, 4))
            #     en_location[0:index+1] = np.ones((index+1, 4))
            #
            #     list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
            #     de_feat[index] = np.ones((1, 34))
            #     de_time[index] = np.ones((1, 4))
            #     de_location[index] = np.ones((1, 4))
            #     list_feat_de.append((de_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_de.append((de_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_de.append((de_location*location_list[i][ii]).reshape(1, 5, -1))
            #     list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
            #     list_mask.append(index)
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_2(feat_list, time_list, location_list, label_list, label_mask_list):#en:[00012] de:[00123]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[20, 16, 12, 8, 4]].sum())
            feat = feat_list[i][ii]
            ts = time_list[i][ii]
            location = location_list[i][ii]
            en_feat_mid = np.zeros((5, 34))
            en_time_mid = np.zeros((5, 4))
            en_location_mid = np.zeros((5, 4))
            de_feat_mid = np.zeros((5, 34))
            de_time_mid = np.zeros((5, 4))
            de_location_mid = np.zeros((5, 4))

            list_label.append(1 - label_list[i][ii][[20, 16, 12, 8, 4]][len_seq-1])
            list_mask.append(len_seq-1)
            for iii in range(0, len_seq-1):
                en_feat_mid[iii + 6-len_seq] = feat[iii]
                en_time_mid[iii + 6-len_seq] = ts[iii]
                en_location_mid[iii + 6-len_seq] = location[iii]
            list_feat_en.append(en_feat_mid.reshape(1, 5, 34))
            list_time_en.append(en_time_mid.reshape(1, 5, 4))
            list_location_en.append(en_location_mid.reshape(1, 5, 4))

            for iii in range(0, len_seq):
                de_feat_mid[iii + 5-len_seq] = feat[iii]
                de_time_mid[iii + 5-len_seq] = ts[iii]
                de_location_mid[iii + 5-len_seq] = location[iii]

            list_feat_de.append(de_feat_mid.reshape(1, 5, 34))
            list_time_de.append(de_time_mid.reshape(1, 5, 4))
            list_location_de.append(de_location_mid.reshape(1, 5, 4))
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_3(feat_list, time_list, location_list, label_list, label_mask_list):#en:[000123] de:[00012] [000123]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[20, 16, 12, 8, 4]].sum())

            en_feat = np.zeros((5, 34))
            en_time = np.zeros((5, 4))
            en_location = np.zeros((5, 4))
            for index in range(1, len_seq):
                en_feat[0:index + 1] = np.ones((index + 1, 34))
                en_time[0:index + 1] = np.ones((index + 1, 4))
                en_location[0:index+1] = np.ones((index+1, 4))

                feat = en_feat*feat_list[i][ii]
                ts = en_time*time_list[i][ii]
                location = en_location*location_list[i][ii]
                list_label.append(1 - label_list[i][ii][[20, 16, 12, 8, 4]][index])
                list_mask.append(index)

                en_feat_mid = np.zeros((5, 34))
                en_time_mid = np.zeros((5, 4))
                en_location_mid = np.zeros((5, 4))
                de_feat_mid = np.zeros((5, 34))
                de_time_mid = np.zeros((5, 4))
                de_location_mid = np.zeros((5, 4))
                for iii in range(0, len_seq):
                    en_feat_mid[iii+5-len_seq] = feat_list[i][ii][iii]
                    en_time_mid[iii+5-len_seq] = time_list[i][ii][iii]
                    en_location_mid[iii+5-len_seq] = location_list[i][ii][iii]
                list_feat_en.append(en_feat_mid.reshape(1, 5, 34))
                list_time_en.append(en_time_mid.reshape(1, 5, 4))
                list_location_en.append(en_location_mid.reshape(1, 5, 4))

                for iii in range(0, index+1):
                    de_feat_mid[iii + 4 - index] = feat[iii]
                    de_time_mid[iii + 4 - index] = ts[iii]
                    de_location_mid[iii + 4 - index] = location[iii]

                list_feat_de.append(de_feat_mid.reshape(1, 5, 34))
                list_time_de.append(de_time_mid.reshape(1, 5, 4))
                list_location_de.append(de_location_mid.reshape(1, 5, 4))
                pass
            # for index in range(1, len_seq):
            #     en_feat[0:index+1] = np.ones((index+1, 34))
            #     en_time[0:index+1] = np.ones((index+1, 4))
            #     en_location[0:index+1] = np.ones((index+1, 4))
            #
            #     list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
            #     de_feat[index] = np.ones((1, 34))
            #     de_time[index] = np.ones((1, 4))
            #     de_location[index] = np.ones((1, 4))
            #     list_feat_de.append((de_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_de.append((de_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_de.append((de_location*location_list[i][ii]).reshape(1, 5, -1))
            #     list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
            #     list_mask.append(index)
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_4(feat_list, time_list, location_list, label_list, label_mask_list):#en:[000123] de:[00012] [000123]
    dim = feat_list[0].shape[-1]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[20, 16, 12, 8, 4]].sum())

            en_feat = np.zeros((5, dim))
            en_time = np.zeros((5, 4))
            en_location = np.zeros((5, 4))
            for index in range(0, len_seq):
                en_feat[0:index + 1] = np.ones((index + 1, dim))
                en_time[0:index + 1] = np.ones((index + 1, 4))
                en_location[0:index+1] = np.ones((index+1, 4))

                feat = en_feat*feat_list[i][ii]
                ts = en_time*time_list[i][ii]
                location = en_location*location_list[i][ii]
                list_label.append(1 - label_list[i][ii][[20, 16, 12, 8, 4]][index])
                list_mask.append(index)

                en_feat_mid = np.zeros((5, dim))
                en_time_mid = np.zeros((5, 4))
                en_location_mid = np.zeros((5, 4))
                de_feat_mid = np.zeros((5, dim))
                de_time_mid = np.zeros((5, 4))
                de_location_mid = np.zeros((5, 4))
                for iii in range(0, len_seq):
                    en_feat_mid[iii+5-len_seq] = feat_list[i][ii][iii]
                    en_time_mid[iii+5-len_seq] = time_list[i][ii][iii]
                    en_location_mid[iii+5-len_seq] = location_list[i][ii][iii]
                list_feat_en.append(en_feat_mid.reshape(1, 5, dim))
                list_time_en.append(en_time_mid.reshape(1, 5, 4))
                list_location_en.append(en_location_mid.reshape(1, 5, 4))

                for iii in range(0, index+1):
                    de_feat_mid[iii + 4 - index] = feat[iii]
                    de_time_mid[iii + 4 - index] = ts[iii]
                    de_location_mid[iii + 4 - index] = location[iii]

                list_feat_de.append(de_feat_mid.reshape(1, 5, dim))
                list_time_de.append(de_time_mid.reshape(1, 5, 4))
                list_location_de.append(de_location_mid.reshape(1, 5, 4))
                pass
            # for index in range(1, len_seq):
            #     en_feat[0:index+1] = np.ones((index+1, 34))
            #     en_time[0:index+1] = np.ones((index+1, 4))
            #     en_location[0:index+1] = np.ones((index+1, 4))
            #
            #     list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
            #     de_feat[index] = np.ones((1, 34))
            #     de_time[index] = np.ones((1, 4))
            #     de_location[index] = np.ones((1, 4))
            #     list_feat_de.append((de_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_de.append((de_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_de.append((de_location*location_list[i][ii]).reshape(1, 5, -1))
            #     list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
            #     list_mask.append(index)
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_5(feat_list, time_list, location_list, label_list, label_mask_list):#en:[000123] de:[000123]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    for i in range(len(feat_list)):
        list_feat_en = []
        list_time_en = []
        list_location_en = []
        list_feat_de = []
        list_time_de = []
        list_location_de = []
        list_label = []
        list_mask = []
        for ii in range(len(feat_list[i])):
            len_seq = int(label_mask_list[i][ii][[4, 8, 12, 16, 20]].sum())

            en_feat = np.zeros((5, 34))
            en_time = np.zeros((5, 4))
            en_location = np.zeros((5, 4))
            for index in range(0, len_seq):
                en_feat[0:index + 1] = np.ones((index + 1, 34))
                en_time[0:index + 1] = np.ones((index + 1, 4))
                en_location[0:index+1] = np.ones((index+1, 4))

                feat = en_feat*feat_list[i][ii]
                ts = en_time*time_list[i][ii]
                location = en_location*location_list[i][ii]
                list_label.append(1 - label_list[i][ii][[4, 8, 12, 16, 20]])
                list_mask.append(len_seq)

                en_feat_mid = np.zeros((5, 34))
                en_time_mid = np.zeros((5, 4))
                en_location_mid = np.zeros((5, 4))
                de_feat_mid = np.zeros((5, 34))
                de_time_mid = np.zeros((5, 4))
                de_location_mid = np.zeros((5, 4))
                for iii in range(0, len_seq):
                    en_feat_mid[iii+5-len_seq] = feat_list[i][ii][iii]
                    en_time_mid[iii+5-len_seq] = time_list[i][ii][iii]
                    en_location_mid[iii+5-len_seq] = location_list[i][ii][iii]
                list_feat_en.append(en_feat_mid.reshape(1, 5, 34))
                list_time_en.append(en_time_mid.reshape(1, 5, 4))
                list_location_en.append(en_location_mid.reshape(1, 5, 4))

                for iii in range(0, index+1):
                    de_feat_mid[iii + 4 - index] = feat[iii]
                    de_time_mid[iii + 4 - index] = ts[iii]
                    de_location_mid[iii + 4 - index] = location[iii]

                list_feat_de.append(de_feat_mid.reshape(1, 5, 34))
                list_time_de.append(de_time_mid.reshape(1, 5, 4))
                list_location_de.append(de_location_mid.reshape(1, 5, 4))
                pass
            # for index in range(1, len_seq):
            #     en_feat[0:index+1] = np.ones((index+1, 34))
            #     en_time[0:index+1] = np.ones((index+1, 4))
            #     en_location[0:index+1] = np.ones((index+1, 4))
            #
            #     list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
            #     de_feat[index] = np.ones((1, 34))
            #     de_time[index] = np.ones((1, 4))
            #     de_location[index] = np.ones((1, 4))
            #     list_feat_de.append((de_feat*feat_list[i][ii]).reshape(1, 5, -1))
            #     list_time_de.append((de_time*time_list[i][ii]).reshape(1, 5, -1))
            #     list_location_de.append((de_location*location_list[i][ii]).reshape(1, 5, -1))
            #     list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
            #     list_mask.append(index)
        en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
        en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
        en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
        de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
        de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
        de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
        label_convert_list.append(np.array(list_label))
        mask_convert_list.append(np.array(list_mask))

    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def data_covert_individual_4(feat_list, time_list, location_list, label_list, label_mask_list):#en:[000123] de:[00012] [000123]
    en_feat_list = []
    en_ts_list = []
    en_location_list = []
    de_feat_list = []
    de_ts_list = []
    de_location_list = []
    label_convert_list = []
    mask_convert_list = []
    list_feat_en = []
    list_time_en = []
    list_location_en = []
    list_feat_de = []
    list_time_de = []
    list_location_de = []
    list_label = []
    list_mask = []
    for ii in range(len(feat_list)):
        len_seq = int(label_mask_list[ii][[20, 16, 12, 8, 4]].sum())

        en_feat = np.zeros((5, 34))
        en_time = np.zeros((5, 4))
        en_location = np.zeros((5, 4))
        for index in range(0, len_seq):
            en_feat[0:index + 1] = np.ones((index + 1, 34))
            en_time[0:index + 1] = np.ones((index + 1, 4))
            en_location[0:index + 1] = np.ones((index + 1, 4))

            feat = en_feat * feat_list[ii]
            ts = en_time * time_list[ii]
            location = en_location * location_list[ii]
            list_label.append(1 - label_list[ii][[20, 16, 12, 8, 4]][index])
            list_mask.append(index)

            en_feat_mid = np.zeros((5, 34))
            en_time_mid = np.zeros((5, 4))
            en_location_mid = np.zeros((5, 4))
            de_feat_mid = np.zeros((5, 34))
            de_time_mid = np.zeros((5, 4))
            de_location_mid = np.zeros((5, 4))
            for iii in range(0, len_seq):
                en_feat_mid[iii + 5 - len_seq] = feat_list[ii][iii]
                en_time_mid[iii + 5 - len_seq] = time_list[ii][iii]
                en_location_mid[iii + 5 - len_seq] = location_list[ii][iii]
            list_feat_en.append(en_feat_mid.reshape(1, 5, 34))
            list_time_en.append(en_time_mid.reshape(1, 5, 4))
            list_location_en.append(en_location_mid.reshape(1, 5, 4))

            for iii in range(0, index + 1):
                de_feat_mid[iii + 4 - index] = feat[iii]
                de_time_mid[iii + 4 - index] = ts[iii]
                de_location_mid[iii + 4 - index] = location[iii]

            list_feat_de.append(de_feat_mid.reshape(1, 5, 34))
            list_time_de.append(de_time_mid.reshape(1, 5, 4))
            list_location_de.append(de_location_mid.reshape(1, 5, 4))
            pass
        # for index in range(1, len_seq):
        #     en_feat[0:index+1] = np.ones((index+1, 34))
        #     en_time[0:index+1] = np.ones((index+1, 4))
        #     en_location[0:index+1] = np.ones((index+1, 4))
        #
        #     list_feat_en.append((en_feat*feat_list[i][ii]).reshape(1, 5, -1))
        #     list_time_en.append((en_time*time_list[i][ii]).reshape(1, 5, -1))
        #     list_location_en.append((en_location*location_list[i][ii]).reshape(1, 5, -1))
        #     de_feat[index] = np.ones((1, 34))
        #     de_time[index] = np.ones((1, 4))
        #     de_location[index] = np.ones((1, 4))
        #     list_feat_de.append((de_feat*feat_list[i][ii]).reshape(1, 5, -1))
        #     list_time_de.append((de_time*time_list[i][ii]).reshape(1, 5, -1))
        #     list_location_de.append((de_location*location_list[i][ii]).reshape(1, 5, -1))
        #     list_label.append(1-label_list[i][ii][[20, 16, 12, 8, 4]][index])
        #     list_mask.append(index)
    en_feat_list.append(np.concatenate(tuple(list_feat_en), axis=0))
    en_ts_list.append(np.concatenate(tuple(list_time_en), axis=0))
    en_location_list.append(np.concatenate(tuple(list_location_en), axis=0))
    de_feat_list.append(np.concatenate(tuple(list_feat_de), axis=0))
    de_ts_list.append(np.concatenate(tuple(list_time_de), axis=0))
    de_location_list.append(np.concatenate(tuple(list_location_de), axis=0))
    label_convert_list.append(np.array(list_label))
    mask_convert_list.append(np.array(list_mask))


    return en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list

def bay_loader_old(en_in_dir, de_in_dir, en_timestamp_dir, de_timestamp_dir, label_dir, label_mask_dir, batch_size=[32, 32, 1]):
    df_en_in = pd.read_csv(en_in_dir, header=None, dtype=int)
    en_in = df_en_in.values.reshape(-1, 10, 5, df_en_in.values.shape[-1])
    df_de_in = pd.read_csv(de_in_dir, header=None, dtype=int)
    de_in = df_de_in.values.reshape(-1, 10, 5, df_de_in.values.shape[-1])
    df_en_ts = pd.read_csv(en_timestamp_dir, header=None, dtype=float)
    en_ts = df_en_ts.values.reshape(-1, 10, 5, df_en_ts.values.shape[-1])
    df_de_ts = pd.read_csv(de_timestamp_dir, header=None, dtype=float)
    de_ts = df_de_ts.values.reshape(-1, 10, 5, df_de_ts.values.shape[-1])
    df_label = pd.read_csv(label_dir, header=None, dtype=float)
    label = df_label.values.reshape(-1, df_label.values.shape[-1])
    df_label_mask = pd.read_csv(label_mask_dir, header=None, dtype=int)
    label_mask = df_label_mask.values.reshape(-1, df_label_mask.values.shape[-1])

    clip = int(label.shape[0] / 10)

    train_loader = data_loader(en_in[:8 * clip], de_in[:8 * clip], en_ts[:8 * clip], de_ts[:8 * clip], label[:8 * clip], label_mask[:8 * clip])
    valid_loader = data_loader(en_in[8 * clip:9 * clip], de_in[8 * clip:9 * clip], en_ts[8 * clip:9 * clip], de_ts[8 * clip:9 * clip], label[8 * clip:9 * clip], label_mask[8 * clip:9 * clip])
    test_loader = data_loader(en_in[9 * clip:], de_in[9 * clip:], en_ts[9 * clip:], de_ts[9 * clip:], label[9 * clip:], label_mask[9 * clip:])

    return DataLoader(train_loader, batch_size=batch_size[0], shuffle=True), \
           DataLoader(valid_loader, batch_size=batch_size[1], shuffle=False), \
           DataLoader(test_loader, batch_size=batch_size[2], shuffle=False)


def bay_loader(feat_list, time_list, location_list, label_list, label_mask_list, clusters_list, K=5, batch_size=[128, 128, 1]):
    en_in_list = []
    de_in_list = []
    en_timestamp_list = []
    de_timestamp_list = []
    en_de_location_list = []
    label = []
    label_mask = []

    stack = feat_list[0].shape[1]
    tier = feat_list[0].shape[2]
    dim_feat = feat_list[0].shape[3]
    dim_time = time_list[0].shape[3]
    for index in range(len(feat_list)):
        label_index_list = []
        en_in = []
        de_in = []
        en_timestamp = []
        de_timestamp = []
        en_de_location = []
        for i in range(label_list[index].shape[0]):
            de_in_mid = np.zeros((1, feat_list[index].shape[1], feat_list[index].shape[2], feat_list[index].shape[3]))
            de_timestamp_mid = np.zeros((1, time_list[index].shape[1], time_list[index].shape[2], time_list[index].shape[3]))
            if label_mask_list[index][i].sum() != 0:
                index0 = int(i / 10)
                index1 = i % 10
                label_index_list.append(i)
                en_in.append(feat_list[index][index0])
                en_timestamp.append(time_list[index][index0])
                en_de_location.append(location_list[index][index0])
                de_in_mid[0, index1] = feat_list[index][index0, index1]
                de_timestamp_mid[0, index1] = time_list[index][index0, index1]
                de_in.append(de_in_mid)
                de_timestamp.append(de_timestamp_mid)
        en_in_list.append(np.concatenate(en_in).reshape(-1, stack, tier, dim_feat))
        de_in_list.append(np.concatenate(de_in))
        en_timestamp_list.append(np.concatenate(en_timestamp).reshape(-1, stack, tier, dim_time))
        de_timestamp_list.append(np.concatenate(de_timestamp))
        en_de_location_list.append(np.concatenate(en_de_location).reshape(-1, stack, tier, dim_time))
        label.append(label_list[index][label_index_list])
        label_mask.append(label_mask_list[index][label_index_list])

        # shuffle_list = list(range(en_in_list[-1].shape[0]))
        # random.shuffle(shuffle_list)
        pass
    train_en_in_list = []
    train_de_in_list = []
    train_en_timestamp_list = []
    train_de_timestamp_list = []
    train_en_de_location_list = []
    train_label_list = []
    train_label_mask_list = []

    valid_en_in_list = []
    valid_de_in_list = []
    valid_en_timestamp_list = []
    valid_de_timestamp_list = []
    valid_en_de_location_list = []
    valid_label_list = []
    valid_label_mask_list = []

    test_en_in_list = []
    test_de_in_list = []
    test_en_timestamp_list = []
    test_de_timestamp_list = []
    test_en_de_location_list = []
    test_label_list = []
    test_label_mask_list = []
    train_list_loader = []
    for index in range(len(label)):
        clip = int(len(label[index]) / 10)
        train_en_in_list.append(en_in_list[index][:8 * clip])
        train_de_in_list.append(de_in_list[index][:8 * clip])
        train_en_timestamp_list.append(en_timestamp_list[index][:8 * clip])
        train_de_timestamp_list.append(de_timestamp_list[index][:8 * clip])
        train_en_de_location_list.append(en_de_location_list[index][:8 * clip])
        train_label_list.append(label[index][:8 * clip])
        train_label_mask_list.append(label_mask[index][:8 * clip])

        '''train_list_loader.append(DataLoader(data_loader(en_in_list[index][:8 * clip],
                               de_in_list[index][:8 * clip],
                               en_timestamp_list[index][:8 * clip],
                               de_timestamp_list[index][:8 * clip],
                               en_de_location_list[index][:8 * clip],
                               label[index][:8 * clip],
                               label_mask[index][:8 * clip]), batch_size = batch_size[0], shuffle=True))'''

        valid_en_in_list.append(en_in_list[index][8 * clip:9 * clip])
        valid_de_in_list.append(de_in_list[index][8 * clip:9 * clip])
        valid_en_timestamp_list.append(en_timestamp_list[index][8 * clip:9 * clip])
        valid_de_timestamp_list.append(de_timestamp_list[index][8 * clip:9 * clip])
        valid_en_de_location_list.append(en_de_location_list[index][8 * clip:9 * clip])
        valid_label_list.append(label[index][8 * clip:9 * clip])
        valid_label_mask_list.append(label_mask[index][8 * clip:9 * clip])

        test_en_in_list.append(en_in_list[index][9 * clip:])
        test_de_in_list.append(de_in_list[index][9 * clip:])
        test_en_timestamp_list.append(en_timestamp_list[index][9 * clip:])
        test_de_timestamp_list.append(de_timestamp_list[index][9 * clip:])
        test_en_de_location_list.append(en_de_location_list[index][9 * clip:])
        test_label_list.append(label[index][9 * clip:])
        test_label_mask_list.append(label_mask[index][9 * clip:])

    train_en_in_all = np.concatenate(train_en_in_list)
    train_de_in_all = np.concatenate(train_de_in_list)
    train_en_timestamp_all = np.concatenate(train_en_timestamp_list)
    train_de_timestamp_all = np.concatenate(train_de_timestamp_list)
    train_en_de_location_all = np.concatenate(train_en_de_location_list)
    train_label_all = np.concatenate(train_label_list)
    train_label_mask_all = np.concatenate(train_label_mask_list)

    train_all_loader = DataLoader(data_loader(train_en_in_all,
                                              train_de_in_all,
                                              train_en_timestamp_all,
                                              train_de_timestamp_all,
                                              train_en_de_location_all,
                                              train_label_all,
                                              train_label_mask_all), batch_size=batch_size[0], shuffle=True)

    valid_loader = DataLoader(data_loader(np.concatenate(valid_en_in_list),
                                          np.concatenate(valid_de_in_list),
                                          np.concatenate(valid_en_timestamp_list),
                                          np.concatenate(valid_de_timestamp_list),
                                          np.concatenate(valid_en_de_location_list),
                                          np.concatenate(valid_label_list),
                                          np.concatenate(valid_label_mask_list)), batch_size=batch_size[1], shuffle=False)

    test_loader = DataLoader(data_loader(np.concatenate(test_en_in_list),
                                         np.concatenate(test_de_in_list),
                                         np.concatenate(test_en_timestamp_list),
                                         np.concatenate(test_de_timestamp_list),
                                         np.concatenate(test_en_de_location_list),
                                         np.concatenate(test_label_list),
                                         np.concatenate(test_label_mask_list)), batch_size=batch_size[2], shuffle=False)

    for index, area_cluster in enumerate(clusters_list[1 - K]):
        list_en_in = []
        list_de_in = []
        list_en_ts = []
        list_de_ts = []
        list_locat = []
        list_label = []
        list_mask = []
        for i in area_cluster.area:
            list_en_in.append(train_en_in_list[i])
            list_de_in.append(train_de_in_list[i])
            list_en_ts.append(train_en_timestamp_list[i])
            list_de_ts.append(train_de_timestamp_list[i])
            list_locat.append(train_en_de_location_list[i])
            list_label.append(train_label_list[i])
            list_mask.append(train_label_mask_list[i])
        train_list_loader.append(DataLoader(data_loader(np.concatenate(list_en_in),
                                                        np.concatenate(list_de_in),
                                                        np.concatenate(list_en_ts),
                                                        np.concatenate(list_de_ts),
                                                        np.concatenate(list_locat),
                                                        np.concatenate(list_label),
                                                        np.concatenate(list_mask)), batch_size=batch_size[0], shuffle=True))

    return train_list_loader, train_all_loader, valid_loader, test_loader

def stack_loader(feat_list, time_list, location_list, label_list, label_mask_list, clusters_list, K=5, batch_size=[128, 128, 1]):
    en_in_list = []
    de_in_list = []
    en_timestamp_list = []
    de_timestamp_list = []
    en_de_location_list = []
    label = []
    label_mask = []
    np.random.seed(1)

    for index in range(len(feat_list)):
        index_list = []
        for i in range(label_list[index].shape[0]):
            if label_mask_list[index][i].sum() != 0:
                index_list.append(i)
        random.shuffle(index_list)
        en_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        de_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        en_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        de_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        en_de_location_list.append(location_list[index].reshape(-1, 5, 4)[index_list])
        label.append(label_list[index][index_list])
        label_mask.append(label_mask_list[index][index_list])

        # shuffle_list = list(range(en_in_list[-1].shape[0]))
        # random.shuffle(shuffle_list)
        pass
    train_en_in_list = []
    train_de_in_list = []
    train_en_timestamp_list = []
    train_de_timestamp_list = []
    train_en_de_location_list = []
    train_label_list = []
    train_label_mask_list = []

    valid_en_in_list = []
    valid_de_in_list = []
    valid_en_timestamp_list = []
    valid_de_timestamp_list = []
    valid_en_de_location_list = []
    valid_label_list = []
    valid_label_mask_list = []

    test_en_in_list = []
    test_de_in_list = []
    test_en_timestamp_list = []
    test_de_timestamp_list = []
    test_en_de_location_list = []
    test_label_list = []
    test_label_mask_list = []
    train_list_loader = []
    for index in range(len(label)):
        clip = int(len(label[index]) / 10)
        train_en_in_list.append(en_in_list[index][:8 * clip])
        train_de_in_list.append(de_in_list[index][:8 * clip])
        train_en_timestamp_list.append(en_timestamp_list[index][:8 * clip])
        train_de_timestamp_list.append(de_timestamp_list[index][:8 * clip])
        train_en_de_location_list.append(en_de_location_list[index][:8 * clip])
        train_label_list.append(label[index][:8 * clip])
        train_label_mask_list.append(label_mask[index][:8 * clip])

        '''train_list_loader.append(DataLoader(data_loader(en_in_list[index][:8 * clip],
                               de_in_list[index][:8 * clip],
                               en_timestamp_list[index][:8 * clip],
                               de_timestamp_list[index][:8 * clip],
                               en_de_location_list[index][:8 * clip],
                               label[index][:8 * clip],
                               label_mask[index][:8 * clip]), batch_size = batch_size[0], shuffle=True))'''

        valid_en_in_list.append(en_in_list[index][8 * clip:9 * clip])
        valid_de_in_list.append(de_in_list[index][8 * clip:9 * clip])
        valid_en_timestamp_list.append(en_timestamp_list[index][8 * clip:9 * clip])
        valid_de_timestamp_list.append(de_timestamp_list[index][8 * clip:9 * clip])
        valid_en_de_location_list.append(en_de_location_list[index][8 * clip:9 * clip])
        valid_label_list.append(label[index][8 * clip:9 * clip])
        valid_label_mask_list.append(label_mask[index][8 * clip:9 * clip])

        test_en_in_list.append(en_in_list[index][9 * clip:])
        test_de_in_list.append(de_in_list[index][9 * clip:])
        test_en_timestamp_list.append(en_timestamp_list[index][9 * clip:])
        test_de_timestamp_list.append(de_timestamp_list[index][9 * clip:])
        test_en_de_location_list.append(en_de_location_list[index][9 * clip:])
        test_label_list.append(label[index][9 * clip:])
        test_label_mask_list.append(label_mask[index][9 * clip:])

    train_en_in_all = np.concatenate(train_en_in_list)
    train_de_in_all = np.concatenate(train_de_in_list)
    train_en_timestamp_all = np.concatenate(train_en_timestamp_list)
    train_de_timestamp_all = np.concatenate(train_de_timestamp_list)
    train_en_de_location_all = np.concatenate(train_en_de_location_list)
    train_label_all = np.concatenate(train_label_list)
    train_label_mask_all = np.concatenate(train_label_mask_list)

    train_all_loader = DataLoader(data_loader(train_en_in_all,
                                              train_de_in_all,
                                              train_en_timestamp_all,
                                              train_de_timestamp_all,
                                              train_en_de_location_all,
                                              train_label_all,
                                              train_label_mask_all), batch_size=batch_size[0], shuffle=True)

    valid_loader = DataLoader(data_loader(np.concatenate(valid_en_in_list),
                                          np.concatenate(valid_de_in_list),
                                          np.concatenate(valid_en_timestamp_list),
                                          np.concatenate(valid_de_timestamp_list),
                                          np.concatenate(valid_en_de_location_list),
                                          np.concatenate(valid_label_list),
                                          np.concatenate(valid_label_mask_list)), batch_size=batch_size[1], shuffle=False)

    test_loader = DataLoader(data_loader(np.concatenate(test_en_in_list),
                                         np.concatenate(test_de_in_list),
                                         np.concatenate(test_en_timestamp_list),
                                         np.concatenate(test_de_timestamp_list),
                                         np.concatenate(test_en_de_location_list),
                                         np.concatenate(test_label_list),
                                         np.concatenate(test_label_mask_list)), batch_size=batch_size[2], shuffle=False)

    for index, area_cluster in enumerate(clusters_list[1 - K]):
        list_en_in = []
        list_de_in = []
        list_en_ts = []
        list_de_ts = []
        list_locat = []
        list_label = []
        list_mask = []
        for i in area_cluster.area:
            list_en_in.append(train_en_in_list[i])
            list_de_in.append(train_de_in_list[i])
            list_en_ts.append(train_en_timestamp_list[i])
            list_de_ts.append(train_de_timestamp_list[i])
            list_locat.append(train_en_de_location_list[i])
            list_label.append(train_label_list[i])
            list_mask.append(train_label_mask_list[i])
        train_list_loader.append(DataLoader(data_loader(np.concatenate(list_en_in),
                                                        np.concatenate(list_de_in),
                                                        np.concatenate(list_en_ts),
                                                        np.concatenate(list_de_ts),
                                                        np.concatenate(list_locat),
                                                        np.concatenate(list_label),
                                                        np.concatenate(list_mask)), batch_size=batch_size[0], shuffle=True))

    return train_list_loader, train_all_loader, valid_loader, test_loader

def stack_loader_k_medoid(feat_list, time_list, location_list, label_list, label_mask_list, clusters_list, batch_size=[128, 128, 1]):
    en_in_list = []
    de_in_list = []
    en_timestamp_list = []
    de_timestamp_list = []
    en_de_location_list = []
    label = []
    label_mask = []
    np.random.seed(1)

    for index in range(len(feat_list)):
        index_list = []
        for i in range(label_list[index].shape[0]):
            if label_mask_list[index][i].sum() != 0:
                index_list.append(i)
        random.shuffle(index_list)
        en_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        de_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        en_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        de_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        en_de_location_list.append(location_list[index].reshape(-1, 5, 4)[index_list])
        label.append(label_list[index][index_list])
        label_mask.append(label_mask_list[index][index_list])

        # shuffle_list = list(range(en_in_list[-1].shape[0]))
        # random.shuffle(shuffle_list)
        pass
    train_en_in_list = []
    train_de_in_list = []
    train_en_timestamp_list = []
    train_de_timestamp_list = []
    train_en_de_location_list = []
    train_label_list = []
    train_label_mask_list = []

    valid_en_in_list = []
    valid_de_in_list = []
    valid_en_timestamp_list = []
    valid_de_timestamp_list = []
    valid_en_de_location_list = []
    valid_label_list = []
    valid_label_mask_list = []

    test_en_in_list = []
    test_de_in_list = []
    test_en_timestamp_list = []
    test_de_timestamp_list = []
    test_en_de_location_list = []
    test_label_list = []
    test_label_mask_list = []
    train_list_loader = []
    for index in range(len(label)):
        clip = int(len(label[index]) / 100)
        train_en_in_list.append(en_in_list[index][:95 * clip])
        train_de_in_list.append(de_in_list[index][:95 * clip])
        train_en_timestamp_list.append(en_timestamp_list[index][:95 * clip])
        train_de_timestamp_list.append(de_timestamp_list[index][:95 * clip])
        train_en_de_location_list.append(en_de_location_list[index][:95 * clip])
        train_label_list.append(label[index][:95 * clip])
        train_label_mask_list.append(label_mask[index][:95 * clip])

        '''train_list_loader.append(DataLoader(data_loader(en_in_list[index][:8 * clip],
                               de_in_list[index][:8 * clip],
                               en_timestamp_list[index][:8 * clip],
                               de_timestamp_list[index][:8 * clip],
                               en_de_location_list[index][:8 * clip],
                               label[index][:8 * clip],
                               label_mask[index][:8 * clip]), batch_size = batch_size[0], shuffle=True))'''

        valid_en_in_list.append(en_in_list[index][95 * clip:98 * clip])
        valid_de_in_list.append(de_in_list[index][95 * clip:98 * clip])
        valid_en_timestamp_list.append(en_timestamp_list[index][95 * clip:98 * clip])
        valid_de_timestamp_list.append(de_timestamp_list[index][95 * clip:98 * clip])
        valid_en_de_location_list.append(en_de_location_list[index][95 * clip:98 * clip])
        valid_label_list.append(label[index][95 * clip:98 * clip])
        valid_label_mask_list.append(label_mask[index][95 * clip:98 * clip])

        test_en_in_list.append(en_in_list[index][98 * clip:])
        test_de_in_list.append(de_in_list[index][98 * clip:])
        test_en_timestamp_list.append(en_timestamp_list[index][98 * clip:])
        test_de_timestamp_list.append(de_timestamp_list[index][98 * clip:])
        test_en_de_location_list.append(en_de_location_list[index][98 * clip:])
        test_label_list.append(label[index][98 * clip:])
        test_label_mask_list.append(label_mask[index][98 * clip:])

    train_en_in_all = np.concatenate(train_en_in_list)
    train_de_in_all = np.concatenate(train_de_in_list)
    train_en_timestamp_all = np.concatenate(train_en_timestamp_list)
    train_de_timestamp_all = np.concatenate(train_de_timestamp_list)
    train_en_de_location_all = np.concatenate(train_en_de_location_list)
    train_label_all = np.concatenate(train_label_list)
    train_label_mask_all = np.concatenate(train_label_mask_list)

    train_all_loader = DataLoader(data_loader(train_en_in_all,
                                              train_de_in_all,
                                              train_en_timestamp_all,
                                              train_de_timestamp_all,
                                              train_en_de_location_all,
                                              train_label_all,
                                              train_label_mask_all), batch_size=batch_size[0], shuffle=True)

    valid_loader = DataLoader(data_loader(np.concatenate(valid_en_in_list),
                                          np.concatenate(valid_de_in_list),
                                          np.concatenate(valid_en_timestamp_list),
                                          np.concatenate(valid_de_timestamp_list),
                                          np.concatenate(valid_en_de_location_list),
                                          np.concatenate(valid_label_list),
                                          np.concatenate(valid_label_mask_list)), batch_size=batch_size[1], shuffle=False)

    test_loader = DataLoader(data_loader(np.concatenate(test_en_in_list),
                                         np.concatenate(test_de_in_list),
                                         np.concatenate(test_en_timestamp_list),
                                         np.concatenate(test_de_timestamp_list),
                                         np.concatenate(test_en_de_location_list),
                                         np.concatenate(test_label_list),
                                         np.concatenate(test_label_mask_list)), batch_size=batch_size[2], shuffle=False)

    for cluster_list_i in clusters_list:
        list_en_in = []
        list_de_in = []
        list_en_ts = []
        list_de_ts = []
        list_locat = []
        list_label = []
        list_mask = []
        for ii in cluster_list_i:
            list_en_in.append(train_en_in_list[ii])
            list_de_in.append(train_de_in_list[ii])
            list_en_ts.append(train_en_timestamp_list[ii])
            list_de_ts.append(train_de_timestamp_list[ii])
            list_locat.append(train_en_de_location_list[ii])
            list_label.append(train_label_list[ii])
            list_mask.append(train_label_mask_list[ii])
        train_list_loader.append(DataLoader(data_loader(np.concatenate(list_en_in),
                                                        np.concatenate(list_de_in),
                                                        np.concatenate(list_en_ts),
                                                        np.concatenate(list_de_ts),
                                                        np.concatenate(list_locat),
                                                        np.concatenate(list_label),
                                                        np.concatenate(list_mask)), batch_size=batch_size[0], shuffle=True, drop_last=True))

    return train_list_loader, train_all_loader, valid_loader, test_loader

def stack_loader_k_medoid_1(en_feat_list, en_ts_list, en_location_list,
                               de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list,
                               clusters_list, batch_size=[128, 128, 1]):
    train_en_feat_list = []
    train_en_ts_list = []
    train_en_location_list = []
    train_de_feat_list = []
    train_de_ts_list = []
    train_de_location_list = []
    train_label_convert_list = []
    train_mask_convert_list = []

    valid_en_feat_list = []
    valid_en_ts_list = []
    valid_en_location_list = []
    valid_de_feat_list = []
    valid_de_ts_list = []
    valid_de_location_list = []
    valid_label_convert_list = []
    valid_mask_convert_list = []

    test_en_feat_list = []
    test_en_ts_list = []
    test_en_location_list = []
    test_de_feat_list = []
    test_de_ts_list = []
    test_de_location_list = []
    test_label_convert_list = []
    test_mask_convert_list = []

    cluster_en_feat_list = []
    cluster_en_ts_list = []
    cluster_en_location_list = []
    cluster_de_feat_list = []
    cluster_de_ts_list = []
    cluster_de_location_list = []
    cluster_label_convert_list = []
    cluster_mask_convert_list = []

    for clusters in clusters_list:
        middle_en_feat_list = []
        middle_en_ts_list = []
        middle_en_location_list = []
        middle_de_feat_list = []
        middle_de_ts_list = []
        middle_de_location_list = []
        middle_label_convert_list = []
        middle_mask_convert_list = []
        for index in clusters:
            middle_en_feat_list.append(en_feat_list[index])
            middle_en_ts_list.append(en_ts_list[index])
            middle_en_location_list.append(en_location_list[index])
            middle_de_feat_list.append(de_feat_list[index])
            middle_de_ts_list.append(de_ts_list[index])
            middle_de_location_list.append(de_location_list[index])
            middle_label_convert_list.append(label_convert_list[index])
            middle_mask_convert_list.append(mask_convert_list[index])
        cluster_en_feat_list.append(np.concatenate(tuple(middle_en_feat_list), axis=0))
        cluster_en_ts_list.append(np.concatenate(tuple(middle_en_ts_list), axis=0))
        cluster_en_location_list.append(np.concatenate(tuple(middle_en_location_list), axis=0))
        cluster_de_feat_list.append(np.concatenate(tuple(middle_de_feat_list), axis=0))
        cluster_de_ts_list.append(np.concatenate(tuple(middle_de_ts_list), axis=0))
        cluster_de_location_list.append(np.concatenate(tuple(middle_de_location_list), axis=0))
        cluster_label_convert_list.append(np.concatenate(middle_label_convert_list))
        cluster_mask_convert_list.append(np.concatenate(middle_mask_convert_list))

    for index in range(len(clusters_list)):
        shuffle_list = np.random.permutation(np.arange(len(cluster_en_feat_list[index])))
        cluster_en_feat_list[index] = cluster_en_feat_list[index][shuffle_list]
        cluster_en_ts_list[index] = cluster_en_ts_list[index][shuffle_list]
        cluster_en_location_list[index] = cluster_en_location_list[index]
        cluster_de_feat_list[index] = cluster_de_feat_list[index]
        cluster_de_ts_list[index] = cluster_de_ts_list[index]
        cluster_de_location_list[index] = cluster_de_location_list[index]
        cluster_label_convert_list[index] = cluster_label_convert_list[index]
        cluster_mask_convert_list[index] = cluster_mask_convert_list[index]

        clip = len(cluster_en_feat_list[index]) / 100

        train_en_feat_list.append(cluster_en_feat_list[index][0:int(80 * clip)])
        train_en_ts_list.append(cluster_en_ts_list[index][0:int(80 * clip)])
        train_en_location_list.append(cluster_en_location_list[index][0:int(80 * clip)])
        train_de_feat_list.append(cluster_de_feat_list[index][0:int(80 * clip)])
        train_de_ts_list.append(cluster_de_ts_list[index][0:int(80 * clip)])
        train_de_location_list.append(cluster_de_location_list[index][0:int(80 * clip)])
        train_label_convert_list.append(cluster_label_convert_list[index][0:int(80 * clip)])
        train_mask_convert_list.append(cluster_mask_convert_list[index][0:int(80 * clip)])

        valid_en_feat_list.append(cluster_en_feat_list[index][int(0 * clip):int(80 * clip)])
        valid_en_ts_list.append(cluster_en_ts_list[index][int(0 * clip):int(80 * clip)])
        valid_en_location_list.append(cluster_en_location_list[index][int(0 * clip):int(80 * clip)])
        valid_de_feat_list.append(cluster_de_feat_list[index][int(0 * clip):int(80 * clip)])
        valid_de_ts_list.append(cluster_de_ts_list[index][int(0 * clip):int(80 * clip)])
        valid_de_location_list.append(cluster_de_location_list[index][int(0 * clip):int(80 * clip)])
        valid_label_convert_list.append(cluster_label_convert_list[index][int(0 * clip):int(80 * clip)])
        valid_mask_convert_list.append(cluster_mask_convert_list[index][int(0 * clip):int(80 * clip)])

        test_en_feat_list.append(cluster_en_feat_list[index][int(80 * clip):])
        test_en_ts_list.append(cluster_en_ts_list[index][int(80 * clip):])
        test_en_location_list.append(cluster_en_location_list[index][int(80 * clip):])
        test_de_feat_list.append(cluster_de_feat_list[index][int(80 * clip):])
        test_de_ts_list.append(cluster_de_ts_list[index][int(80 * clip):])
        test_de_location_list.append(cluster_de_location_list[index][int(80 * clip):])
        test_label_convert_list.append(cluster_label_convert_list[index][int(80 * clip):])
        test_mask_convert_list.append(cluster_mask_convert_list[index][int(80 * clip):])

    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []

    for index in range(len(clusters_list)):
        train_loader = DataLoader(data_loader_1(train_en_feat_list[index],
                                                train_en_ts_list[index],
                                                train_en_location_list[index],
                                                train_de_feat_list[index],
                                                train_de_ts_list[index],
                                                train_de_location_list[index],
                                                train_label_convert_list[index],
                                                train_mask_convert_list[index]), batch_size=batch_size[0], shuffle=True)

        valid_loader = DataLoader(data_loader_1(valid_en_feat_list[index],
                                                valid_en_ts_list[index],
                                                valid_en_location_list[index],
                                                valid_de_feat_list[index],
                                                valid_de_ts_list[index],
                                                valid_de_location_list[index],
                                                valid_label_convert_list[index],
                                                valid_mask_convert_list[index]), batch_size=batch_size[1], shuffle=True)

        test_loader = DataLoader(data_loader_1(test_en_feat_list[index],
                                               test_en_ts_list[index],
                                               test_en_location_list[index],
                                               test_de_feat_list[index],
                                               test_de_ts_list[index],
                                               test_de_location_list[index],
                                               test_label_convert_list[index],
                                               test_mask_convert_list[index]), batch_size=batch_size[2], shuffle=True)

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)

    train_loader_all = DataLoader(data_loader_1(np.concatenate(train_en_feat_list),
                                                np.concatenate(train_en_ts_list),
                                                np.concatenate(train_en_location_list),
                                                np.concatenate(train_de_feat_list),
                                                np.concatenate(train_de_ts_list),
                                                np.concatenate(train_de_location_list),
                                                np.concatenate(train_label_convert_list),
                                                np.concatenate(train_mask_convert_list)), batch_size=batch_size[0], shuffle=True)

    valid_loader_all = DataLoader(data_loader_1(np.concatenate(valid_en_feat_list),
                                                np.concatenate(valid_en_ts_list),
                                                np.concatenate(valid_en_location_list),
                                                np.concatenate(valid_de_feat_list),
                                                np.concatenate(valid_de_ts_list),
                                                np.concatenate(valid_de_location_list),
                                                np.concatenate(valid_label_convert_list),
                                                np.concatenate(valid_mask_convert_list)), batch_size=batch_size[1], shuffle=True)

    test_loader_all = DataLoader(data_loader_1(np.concatenate(test_en_feat_list),
                                               np.concatenate(test_en_ts_list),
                                               np.concatenate(test_en_location_list),
                                               np.concatenate(test_de_feat_list),
                                               np.concatenate(test_de_ts_list),
                                               np.concatenate(test_de_location_list),
                                               np.concatenate(test_label_convert_list),
                                               np.concatenate(test_mask_convert_list)), batch_size=batch_size[2], shuffle=True)

    return train_loader_list, valid_loader_list, test_loader_list, \
           train_loader_all, valid_loader_all, test_loader_all

def stack_loader_k_medoid_fold(en_feat_list, en_ts_list, en_location_list,
                            de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list,
                            clusters_list, batch_size=[128, 128, 1]):
    
    train_en_feat_list = [] 
    train_en_ts_list = []
    train_en_location_list = []
    train_de_feat_list = []
    train_de_ts_list = []
    train_de_location_list = []
    train_label_convert_list = []
    train_mask_convert_list = []

    valid_en_feat_list = []
    valid_en_ts_list = []
    valid_en_location_list = []
    valid_de_feat_list = []
    valid_de_ts_list = []
    valid_de_location_list = []
    valid_label_convert_list = []
    valid_mask_convert_list = []

    test_en_feat_list = []
    test_en_ts_list = []
    test_en_location_list = []
    test_de_feat_list = []
    test_de_ts_list = []
    test_de_location_list = []
    test_label_convert_list = []
    test_mask_convert_list = []

    cluster_en_feat_list = []
    cluster_en_ts_list = []
    cluster_en_location_list = []
    cluster_de_feat_list = []
    cluster_de_ts_list = []
    cluster_de_location_list = []
    cluster_label_convert_list = []
    cluster_mask_convert_list = []

    for clusters in clusters_list:
        middle_en_feat_list = []
        middle_en_ts_list = []
        middle_en_location_list = []
        middle_de_feat_list = []
        middle_de_ts_list = []
        middle_de_location_list = []
        middle_label_convert_list = []
        middle_mask_convert_list = []
        for index in clusters:
            middle_en_feat_list.append(en_feat_list[index])
            middle_en_ts_list.append(en_ts_list[index])
            middle_en_location_list.append(en_location_list[index])
            middle_de_feat_list.append(de_feat_list[index])
            middle_de_ts_list.append(de_ts_list[index])
            middle_de_location_list.append(de_location_list[index])
            middle_label_convert_list.append(label_convert_list[index])
            middle_mask_convert_list.append(mask_convert_list[index])
        cluster_en_feat_list.append(np.concatenate(tuple(middle_en_feat_list), axis=0))
        cluster_en_ts_list.append(np.concatenate(tuple(middle_en_ts_list), axis=0))
        cluster_en_location_list.append(np.concatenate(tuple(middle_en_location_list), axis=0))
        cluster_de_feat_list.append(np.concatenate(tuple(middle_de_feat_list), axis=0))
        cluster_de_ts_list.append(np.concatenate(tuple(middle_de_ts_list), axis=0))
        cluster_de_location_list.append(np.concatenate(tuple(middle_de_location_list), axis=0))
        cluster_label_convert_list.append(np.concatenate(middle_label_convert_list))
        cluster_mask_convert_list.append(np.concatenate(middle_mask_convert_list))

    for index in range(len(clusters_list)):
        shuffle_list = np.random.permutation(np.arange(len(cluster_en_feat_list[index])))
        cluster_en_feat_list[index] = cluster_en_feat_list[index][shuffle_list]
        cluster_en_ts_list[index] = cluster_en_ts_list[index][shuffle_list]
        cluster_en_location_list[index] = cluster_en_location_list[index]
        cluster_de_feat_list[index] = cluster_de_feat_list[index]
        cluster_de_ts_list[index] = cluster_de_ts_list[index]
        cluster_de_location_list[index] = cluster_de_location_list[index]
        cluster_label_convert_list[index] = cluster_label_convert_list[index]
        cluster_mask_convert_list[index] = cluster_mask_convert_list[index]

        clip = len(cluster_en_feat_list[index]) / 100
        
        en_feat_interval_1 = cluster_en_feat_list[index][0:int(60 * clip)]
        en_feat_interval_2 = cluster_en_feat_list[index][0:int(40 * clip)]
        en_feat_interval_3 = cluster_en_feat_list[index][int(60 * clip):int(80 * clip)]
        en_feat_interval_4 = cluster_en_feat_list[index][0:int(20 * clip)]
        en_feat_interval_5 = cluster_en_feat_list[index][int(40 * clip):int(80 * clip)]
        en_feat_interval_6 = cluster_en_feat_list[index][int(20 * clip):int(80 * clip)]
        train_en_feat_list.append(np.concatenate([en_feat_interval_1, en_feat_interval_2, en_feat_interval_3, en_feat_interval_4,\
                                                  en_feat_interval_5, en_feat_interval_6]))

        en_ts_interval_1 = cluster_en_ts_list[index][0:int(60 * clip)]
        en_ts_interval_2 = cluster_en_ts_list[index][0:int(40 * clip)]
        en_ts_interval_3 = cluster_en_ts_list[index][int(60 * clip):int(80 * clip)]
        en_ts_interval_4 = cluster_en_ts_list[index][0:int(20 * clip)]
        en_ts_interval_5 = cluster_en_ts_list[index][int(40 * clip):int(80 * clip)]
        en_ts_interval_6 = cluster_en_ts_list[index][int(20 * clip):int(80 * clip)]
        train_en_ts_list.append(np.concatenate([en_ts_interval_1, en_ts_interval_2, en_ts_interval_3, en_ts_interval_4,\
                                                en_ts_interval_5, en_ts_interval_6]))

        en_location_interval_1 = cluster_en_location_list[index][0:int(60 * clip)]
        en_location_interval_2 = cluster_en_location_list[index][0:int(40 * clip)]
        en_location_interval_3 = cluster_en_location_list[index][int(60 * clip):int(80 * clip)]
        en_location_interval_4 = cluster_en_location_list[index][0:int(20 * clip)]
        en_location_interval_5 = cluster_en_location_list[index][int(40 * clip):int(80 * clip)]
        en_location_interval_6 = cluster_en_location_list[index][int(20 * clip):int(80 * clip)]
        train_en_location_list.append(np.concatenate([en_location_interval_1, en_location_interval_2, en_location_interval_3, en_location_interval_4,\
                                                      en_location_interval_5, en_location_interval_6]))

        de_feat_interval_1 = cluster_de_feat_list[index][0:int(60 * clip)]
        de_feat_interval_2 = cluster_de_feat_list[index][0:int(40 * clip)]
        de_feat_interval_3 = cluster_de_feat_list[index][int(60 * clip):int(80 * clip)]
        de_feat_interval_4 = cluster_de_feat_list[index][0:int(20 * clip)]
        de_feat_interval_5 = cluster_de_feat_list[index][int(40 * clip):int(80 * clip)]
        de_feat_interval_6 = cluster_de_feat_list[index][int(20 * clip):int(80 * clip)]
        train_de_feat_list.append(np.concatenate([de_feat_interval_1, de_feat_interval_2, de_feat_interval_3, de_feat_interval_4,\
                                                  de_feat_interval_5, de_feat_interval_6]))

        de_ts_interval_1 = cluster_de_ts_list[index][0:int(60 * clip)]
        de_ts_interval_2 = cluster_de_ts_list[index][0:int(40 * clip)]
        de_ts_interval_3 = cluster_de_ts_list[index][int(60 * clip):int(80 * clip)]
        de_ts_interval_4 = cluster_de_ts_list[index][0:int(20 * clip)]
        de_ts_interval_5 = cluster_de_ts_list[index][int(40 * clip):int(80 * clip)]
        de_ts_interval_6 = cluster_de_ts_list[index][int(20 * clip):int(80 * clip)]
        train_de_ts_list.append(np.concatenate([de_ts_interval_1, de_ts_interval_2, de_ts_interval_3, de_ts_interval_4,\
                                                de_ts_interval_5, de_ts_interval_6]))

        de_location_interval_1 = cluster_de_location_list[index][0:int(60 * clip)]
        de_location_interval_2 = cluster_de_location_list[index][0:int(40 * clip)]
        de_location_interval_3 = cluster_de_location_list[index][int(60 * clip):int(80 * clip)]
        de_location_interval_4 = cluster_de_location_list[index][0:int(20 * clip)]
        de_location_interval_5 = cluster_de_location_list[index][int(40 * clip):int(80 * clip)]
        de_location_interval_6 = cluster_de_location_list[index][int(20 * clip):int(80 * clip)]
        train_de_location_list.append(np.concatenate([de_location_interval_1, de_location_interval_2, de_location_interval_3, de_location_interval_4,\
                                                      de_location_interval_5, de_location_interval_6]))

        label_convert_interval_1 = cluster_label_convert_list[index][0:int(60 * clip)]
        label_convert_interval_2 = cluster_label_convert_list[index][0:int(40 * clip)]
        label_convert_interval_3 = cluster_label_convert_list[index][int(60 * clip):int(80 * clip)]
        label_convert_interval_4 = cluster_label_convert_list[index][0:int(20 * clip)]
        label_convert_interval_5 = cluster_label_convert_list[index][int(40 * clip):int(80 * clip)]
        label_convert_interval_6 = cluster_label_convert_list[index][int(20 * clip):int(80 * clip)]
        train_label_convert_list.append(np.concatenate([label_convert_interval_1, label_convert_interval_2, label_convert_interval_3, label_convert_interval_4,\
                                                        label_convert_interval_5, label_convert_interval_6]))

        mask_convert_interval_1 = cluster_mask_convert_list[index][0:int(60 * clip)]
        mask_convert_interval_2 = cluster_mask_convert_list[index][0:int(40 * clip)]
        mask_convert_interval_3 = cluster_mask_convert_list[index][int(60 * clip):int(80 * clip)]
        mask_convert_interval_4 = cluster_mask_convert_list[index][0:int(20 * clip)]
        mask_convert_interval_5 = cluster_mask_convert_list[index][int(40 * clip):int(80 * clip)]
        mask_convert_interval_6 = cluster_mask_convert_list[index][int(20 * clip):int(80 * clip)]
        train_mask_convert_list.append(np.concatenate([mask_convert_interval_1, mask_convert_interval_2, mask_convert_interval_3, mask_convert_interval_4,\
                                                       mask_convert_interval_5, mask_convert_interval_6]))

        en_feat_interval_1 = cluster_en_feat_list[index][int(60 * clip):int(80 * clip)]
        en_feat_interval_2 = cluster_en_feat_list[index][int(40 * clip):int(60 * clip)]
        en_feat_interval_3 = cluster_en_feat_list[index][int(20 * clip):int(40 * clip)]
        en_feat_interval_4 = cluster_en_feat_list[index][0:int(20 * clip)]
        valid_en_feat_list.append(np.concatenate([en_feat_interval_1, en_feat_interval_2, en_feat_interval_3, en_feat_interval_4]))

        en_ts_interval_1 = cluster_en_ts_list[index][int(60 * clip):int(80 * clip)]
        en_ts_interval_2 = cluster_en_ts_list[index][int(40 * clip):int(60 * clip)]
        en_ts_interval_3 = cluster_en_ts_list[index][int(20 * clip):int(40 * clip)]
        en_ts_interval_4 = cluster_en_ts_list[index][0:int(20 * clip)]
        valid_en_ts_list.append(np.concatenate([en_ts_interval_1, en_ts_interval_2, en_ts_interval_3, en_ts_interval_4]))

        en_location_interval_1 = cluster_en_location_list[index][int(60 * clip):int(80 * clip)]
        en_location_interval_2 = cluster_en_location_list[index][int(40 * clip):int(60 * clip)]
        en_location_interval_3 = cluster_en_location_list[index][int(20 * clip):int(40 * clip)]
        en_location_interval_4 = cluster_en_location_list[index][0:int(20 * clip)]
        valid_en_location_list.append(np.concatenate([en_location_interval_1, en_location_interval_2, en_location_interval_3, en_location_interval_4]))

        de_feat_interval_1 = cluster_de_feat_list[index][int(60 * clip):int(80 * clip)]
        de_feat_interval_2 = cluster_de_feat_list[index][int(40 * clip):int(60 * clip)]
        de_feat_interval_3 = cluster_de_feat_list[index][int(20 * clip):int(40 * clip)]
        de_feat_interval_4 = cluster_de_feat_list[index][0:int(20 * clip)]
        valid_de_feat_list.append(np.concatenate([de_feat_interval_1, de_feat_interval_2, de_feat_interval_3, de_feat_interval_4]))

        de_ts_interval_1 = cluster_de_ts_list[index][int(60 * clip):int(80 * clip)]
        de_ts_interval_2 = cluster_de_ts_list[index][int(40 * clip):int(60 * clip)]
        de_ts_interval_3 = cluster_de_ts_list[index][int(20 * clip):int(40 * clip)]
        de_ts_interval_4 = cluster_de_ts_list[index][0:int(20 * clip)]
        valid_de_ts_list.append(np.concatenate([de_ts_interval_1, de_ts_interval_2, de_ts_interval_3, de_ts_interval_4]))

        de_location_interval_1 = cluster_de_location_list[index][int(60 * clip):int(80 * clip)]
        de_location_interval_2 = cluster_de_location_list[index][int(40 * clip):int(60 * clip)]
        de_location_interval_3 = cluster_de_location_list[index][int(20 * clip):int(40 * clip)]
        de_location_interval_4 = cluster_de_location_list[index][0:int(20 * clip)]
        valid_de_location_list.append(np.concatenate([de_location_interval_1, de_location_interval_2, de_location_interval_3, de_location_interval_4]))

        label_convert_interval_1 = cluster_label_convert_list[index][int(60 * clip):int(80 * clip)]
        label_convert_interval_2 = cluster_label_convert_list[index][int(40 * clip):int(60 * clip)]
        label_convert_interval_3 = cluster_label_convert_list[index][int(20 * clip):int(40 * clip)]
        label_convert_interval_4 = cluster_label_convert_list[index][0:int(20 * clip)]
        valid_label_convert_list.append(np.concatenate([label_convert_interval_1, label_convert_interval_2, label_convert_interval_3, label_convert_interval_4]))

        mask_convert_interval_1 = cluster_mask_convert_list[index][int(60 * clip):int(80 * clip)]
        mask_convert_interval_2 = cluster_mask_convert_list[index][int(40 * clip):int(60 * clip)]
        mask_convert_interval_3 = cluster_mask_convert_list[index][int(20 * clip):int(40 * clip)]
        mask_convert_interval_4 = cluster_mask_convert_list[index][0:int(20 * clip)]
        valid_mask_convert_list.append(np.concatenate([mask_convert_interval_1, mask_convert_interval_2, mask_convert_interval_3, mask_convert_interval_4]))

        test_en_feat_list.append(cluster_en_feat_list[index][int(args.proportion * clip):])
        test_en_ts_list.append(cluster_en_ts_list[index][int(args.proportion * clip):])
        test_en_location_list.append(cluster_en_location_list[index][int(args.proportion * clip):])
        test_de_feat_list.append(cluster_de_feat_list[index][int(args.proportion * clip):])
        test_de_ts_list.append(cluster_de_ts_list[index][int(args.proportion * clip):])
        test_de_location_list.append(cluster_de_location_list[index][int(args.proportion * clip):])
        test_label_convert_list.append(cluster_label_convert_list[index][int(args.proportion * clip):])
        test_mask_convert_list.append(cluster_mask_convert_list[index][int(args.proportion * clip):])
    
    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []

    for index in range(len(clusters_list)):
        train_loader = DataLoader(data_loader_1(train_en_feat_list[index],
                                              train_en_ts_list[index],
                                              train_en_location_list[index],
                                              train_de_feat_list[index],
                                              train_de_ts_list[index],
                                              train_de_location_list[index],
                                              train_label_convert_list[index],
                                              train_mask_convert_list[index]), batch_size=batch_size[0], shuffle=True, drop_last=True)

        valid_loader = DataLoader(data_loader_1(valid_en_feat_list[index],
                                                valid_en_ts_list[index],
                                                valid_en_location_list[index],
                                                valid_de_feat_list[index],
                                                valid_de_ts_list[index],
                                                valid_de_location_list[index],
                                                valid_label_convert_list[index],
                                                valid_mask_convert_list[index]), batch_size=batch_size[1], shuffle=True, drop_last=True)

        test_loader = DataLoader(data_loader_1(test_en_feat_list[index],
                                                test_en_ts_list[index],
                                                test_en_location_list[index],
                                                test_de_feat_list[index],
                                                test_de_ts_list[index],
                                                test_de_location_list[index],
                                                test_label_convert_list[index],
                                                test_mask_convert_list[index]), batch_size=batch_size[2], shuffle=True, drop_last=True)
        
        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)
    
    train_loader_all = DataLoader(data_loader_1(np.concatenate(train_en_feat_list),
                                                np.concatenate(train_en_ts_list),
                                                np.concatenate(train_en_location_list),
                                                np.concatenate(train_de_feat_list),
                                                np.concatenate(train_de_ts_list),
                                                np.concatenate(train_de_location_list),
                                                np.concatenate(train_label_convert_list),
                                                np.concatenate(train_mask_convert_list)), batch_size=batch_size[0], shuffle=True)

    valid_loader_all = DataLoader(data_loader_1(np.concatenate(valid_en_feat_list),
                                                np.concatenate(valid_en_ts_list),
                                                np.concatenate(valid_en_location_list),
                                                np.concatenate(valid_de_feat_list),
                                                np.concatenate(valid_de_ts_list),
                                                np.concatenate(valid_de_location_list),
                                                np.concatenate(valid_label_convert_list),
                                                np.concatenate(valid_mask_convert_list)), batch_size=batch_size[1], shuffle=True)

    test_loader_all = DataLoader(data_loader_1(np.concatenate(test_en_feat_list),
                                                np.concatenate(test_en_ts_list),
                                                np.concatenate(test_en_location_list),
                                                np.concatenate(test_de_feat_list),
                                                np.concatenate(test_de_ts_list),
                                                np.concatenate(test_de_location_list),
                                                np.concatenate(test_label_convert_list),
                                                np.concatenate(test_mask_convert_list)), batch_size=batch_size[2], shuffle=True)

    return train_loader_list, valid_loader_list, test_loader_list,\
           train_loader_all, valid_loader_all, test_loader_all


def stack_loader_k_medoid_fold_simulate(en_feat_list, en_ts_list, en_location_list,
                               de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list,
                               clusters_list, batch_size=[128, 128, 1]):
    train_en_feat_list = []
    train_en_ts_list = []
    train_en_location_list = []
    train_de_feat_list = []
    train_de_ts_list = []
    train_de_location_list = []
    train_label_convert_list = []
    train_mask_convert_list = []

    valid_en_feat_list = []
    valid_en_ts_list = []
    valid_en_location_list = []
    valid_de_feat_list = []
    valid_de_ts_list = []
    valid_de_location_list = []
    valid_label_convert_list = []
    valid_mask_convert_list = []

    test_en_feat_list = []
    test_en_ts_list = []
    test_en_location_list = []
    test_de_feat_list = []
    test_de_ts_list = []
    test_de_location_list = []
    test_label_convert_list = []
    test_mask_convert_list = []

    cluster_en_feat_list = []
    cluster_en_ts_list = []
    cluster_en_location_list = []
    cluster_de_feat_list = []
    cluster_de_ts_list = []
    cluster_de_location_list = []
    cluster_label_convert_list = []
    cluster_mask_convert_list = []

    for clusters in clusters_list:
        middle_en_feat_list = []
        middle_en_ts_list = []
        middle_en_location_list = []
        middle_de_feat_list = []
        middle_de_ts_list = []
        middle_de_location_list = []
        middle_label_convert_list = []
        middle_mask_convert_list = []
        for index in clusters:
            middle_en_feat_list.append(en_feat_list[index])
            middle_en_ts_list.append(en_ts_list[index])
            middle_en_location_list.append(en_location_list[index])
            middle_de_feat_list.append(de_feat_list[index])
            middle_de_ts_list.append(de_ts_list[index])
            middle_de_location_list.append(de_location_list[index])
            middle_label_convert_list.append(label_convert_list[index])
            middle_mask_convert_list.append(mask_convert_list[index])
        cluster_en_feat_list.append(np.concatenate(tuple(middle_en_feat_list), axis=0))
        cluster_en_ts_list.append(np.concatenate(tuple(middle_en_ts_list), axis=0))
        cluster_en_location_list.append(np.concatenate(tuple(middle_en_location_list), axis=0))
        cluster_de_feat_list.append(np.concatenate(tuple(middle_de_feat_list), axis=0))
        cluster_de_ts_list.append(np.concatenate(tuple(middle_de_ts_list), axis=0))
        cluster_de_location_list.append(np.concatenate(tuple(middle_de_location_list), axis=0))
        cluster_label_convert_list.append(np.concatenate(middle_label_convert_list))
        cluster_mask_convert_list.append(np.concatenate(middle_mask_convert_list))

    for index in range(len(clusters_list)):
        shuffle_list = np.random.permutation(np.arange(len(cluster_en_feat_list[index])))
        cluster_en_feat_list[index] = cluster_en_feat_list[index][shuffle_list]
        cluster_en_ts_list[index] = cluster_en_ts_list[index][shuffle_list]
        cluster_en_location_list[index] = cluster_en_location_list[index]
        cluster_de_feat_list[index] = cluster_de_feat_list[index]
        cluster_de_ts_list[index] = cluster_de_ts_list[index]
        cluster_de_location_list[index] = cluster_de_location_list[index]
        cluster_label_convert_list[index] = cluster_label_convert_list[index]
        cluster_mask_convert_list[index] = cluster_mask_convert_list[index]

        clip = len(cluster_en_feat_list[index]) / 100

        train_en_feat_list.append(cluster_en_feat_list[index][:int(args.proportion * clip)])
        train_en_ts_list.append(cluster_en_ts_list[index][:int(args.proportion * clip)])
        train_en_location_list.append(cluster_en_location_list[index][:int(args.proportion * clip)])
        train_de_feat_list.append(cluster_de_feat_list[index][:int(args.proportion * clip)])
        train_de_ts_list.append(cluster_de_ts_list[index][:int(args.proportion * clip)])
        train_de_location_list.append(cluster_de_location_list[index][:int(args.proportion * clip)])
        train_label_convert_list.append(cluster_label_convert_list[index][:int(args.proportion * clip)])
        train_mask_convert_list.append(cluster_mask_convert_list[index][:int(args.proportion * clip)])

        valid_en_feat_list.append(cluster_en_feat_list[index][:int(args.proportion * clip)])
        valid_en_ts_list.append(cluster_en_ts_list[index][:int(args.proportion * clip)])
        valid_en_location_list.append(cluster_en_location_list[index][:int(args.proportion * clip)])
        valid_de_feat_list.append(cluster_de_feat_list[index][:int(args.proportion * clip)])
        valid_de_ts_list.append(cluster_de_ts_list[index][:int(args.proportion * clip)])
        valid_de_location_list.append(cluster_de_location_list[index][:int(args.proportion * clip)])
        valid_label_convert_list.append(cluster_label_convert_list[index][:int(args.proportion * clip)])
        valid_mask_convert_list.append(cluster_mask_convert_list[index][:int(args.proportion * clip)])

        test_en_feat_list.append(cluster_en_feat_list[index][int(args.proportion * clip):])
        test_en_ts_list.append(cluster_en_ts_list[index][int(args.proportion * clip):])
        test_en_location_list.append(cluster_en_location_list[index][int(args.proportion * clip):])
        test_de_feat_list.append(cluster_de_feat_list[index][int(args.proportion * clip):])
        test_de_ts_list.append(cluster_de_ts_list[index][int(args.proportion * clip):])
        test_de_location_list.append(cluster_de_location_list[index][int(args.proportion * clip):])
        test_label_convert_list.append(cluster_label_convert_list[index][int(args.proportion * clip):])
        test_mask_convert_list.append(cluster_mask_convert_list[index][int(args.proportion * clip):])

    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []

    for index in range(len(clusters_list)):
        train_loader = DataLoader(data_loader_1(train_en_feat_list[index],
                                                train_en_ts_list[index],
                                                train_en_location_list[index],
                                                train_de_feat_list[index],
                                                train_de_ts_list[index],
                                                train_de_location_list[index],
                                                train_label_convert_list[index],
                                                train_mask_convert_list[index]), batch_size=batch_size[0], shuffle=True,
                                  drop_last=True)

        valid_loader = DataLoader(data_loader_1(valid_en_feat_list[index],
                                                valid_en_ts_list[index],
                                                valid_en_location_list[index],
                                                valid_de_feat_list[index],
                                                valid_de_ts_list[index],
                                                valid_de_location_list[index],
                                                valid_label_convert_list[index],
                                                valid_mask_convert_list[index]), batch_size=batch_size[1], shuffle=True,
                                  drop_last=True)

        test_loader = DataLoader(data_loader_1(test_en_feat_list[index],
                                               test_en_ts_list[index],
                                               test_en_location_list[index],
                                               test_de_feat_list[index],
                                               test_de_ts_list[index],
                                               test_de_location_list[index],
                                               test_label_convert_list[index],
                                               test_mask_convert_list[index]), batch_size=batch_size[2], shuffle=True,
                                 drop_last=True)

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)

    train_loader_all = DataLoader(data_loader_1(np.concatenate(train_en_feat_list),
                                                np.concatenate(train_en_ts_list),
                                                np.concatenate(train_en_location_list),
                                                np.concatenate(train_de_feat_list),
                                                np.concatenate(train_de_ts_list),
                                                np.concatenate(train_de_location_list),
                                                np.concatenate(train_label_convert_list),
                                                np.concatenate(train_mask_convert_list)), batch_size=batch_size[0],
                                  shuffle=True)

    valid_loader_all = DataLoader(data_loader_1(np.concatenate(valid_en_feat_list),
                                                np.concatenate(valid_en_ts_list),
                                                np.concatenate(valid_en_location_list),
                                                np.concatenate(valid_de_feat_list),
                                                np.concatenate(valid_de_ts_list),
                                                np.concatenate(valid_de_location_list),
                                                np.concatenate(valid_label_convert_list),
                                                np.concatenate(valid_mask_convert_list)), batch_size=batch_size[1],
                                  shuffle=True)

    test_loader_all = DataLoader(data_loader_1(np.concatenate(test_en_feat_list),
                                               np.concatenate(test_en_ts_list),
                                               np.concatenate(test_en_location_list),
                                               np.concatenate(test_de_feat_list),
                                               np.concatenate(test_de_ts_list),
                                               np.concatenate(test_de_location_list),
                                               np.concatenate(test_label_convert_list),
                                               np.concatenate(test_mask_convert_list)), batch_size=batch_size[2],
                                 shuffle=True)

    return train_loader_list, valid_loader_list, test_loader_list, \
        train_loader_all, valid_loader_all, test_loader_all

def stack_loader_no_fold(en_feat_list, en_ts_list, en_location_list,
                               de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list,
                               clusters_list, batch_size=[128, 128, 1]):
    train_en_feat_list = []
    train_en_ts_list = []
    train_en_location_list = []
    train_de_feat_list = []
    train_de_ts_list = []
    train_de_location_list = []
    train_label_convert_list = []
    train_mask_convert_list = []

    valid_en_feat_list = []
    valid_en_ts_list = []
    valid_en_location_list = []
    valid_de_feat_list = []
    valid_de_ts_list = []
    valid_de_location_list = []
    valid_label_convert_list = []
    valid_mask_convert_list = []

    test_en_feat_list = []
    test_en_ts_list = []
    test_en_location_list = []
    test_de_feat_list = []
    test_de_ts_list = []
    test_de_location_list = []
    test_label_convert_list = []
    test_mask_convert_list = []

    cluster_en_feat_list = []
    cluster_en_ts_list = []
    cluster_en_location_list = []
    cluster_de_feat_list = []
    cluster_de_ts_list = []
    cluster_de_location_list = []
    cluster_label_convert_list = []
    cluster_mask_convert_list = []

    for clusters in clusters_list:
        middle_en_feat_list = []
        middle_en_ts_list = []
        middle_en_location_list = []
        middle_de_feat_list = []
        middle_de_ts_list = []
        middle_de_location_list = []
        middle_label_convert_list = []
        middle_mask_convert_list = []
        for index in clusters:
            middle_en_feat_list.append(en_feat_list[index])
            middle_en_ts_list.append(en_ts_list[index])
            middle_en_location_list.append(en_location_list[index])
            middle_de_feat_list.append(de_feat_list[index])
            middle_de_ts_list.append(de_ts_list[index])
            middle_de_location_list.append(de_location_list[index])
            middle_label_convert_list.append(label_convert_list[index])
            middle_mask_convert_list.append(mask_convert_list[index])
        cluster_en_feat_list.append(np.concatenate(tuple(middle_en_feat_list), axis=0))
        cluster_en_ts_list.append(np.concatenate(tuple(middle_en_ts_list), axis=0))
        cluster_en_location_list.append(np.concatenate(tuple(middle_en_location_list), axis=0))
        cluster_de_feat_list.append(np.concatenate(tuple(middle_de_feat_list), axis=0))
        cluster_de_ts_list.append(np.concatenate(tuple(middle_de_ts_list), axis=0))
        cluster_de_location_list.append(np.concatenate(tuple(middle_de_location_list), axis=0))
        cluster_label_convert_list.append(np.concatenate(middle_label_convert_list))
        cluster_mask_convert_list.append(np.concatenate(middle_mask_convert_list))

    for index in range(len(clusters_list)):
        shuffle_list = np.random.permutation(np.arange(len(cluster_en_feat_list[index])))
        cluster_en_feat_list[index] = cluster_en_feat_list[index][shuffle_list]
        cluster_en_ts_list[index] = cluster_en_ts_list[index][shuffle_list]
        cluster_en_location_list[index] = cluster_en_location_list[index]
        cluster_de_feat_list[index] = cluster_de_feat_list[index]
        cluster_de_ts_list[index] = cluster_de_ts_list[index]
        cluster_de_location_list[index] = cluster_de_location_list[index]
        cluster_label_convert_list[index] = cluster_label_convert_list[index]
        cluster_mask_convert_list[index] = cluster_mask_convert_list[index]

        clip = len(cluster_en_feat_list[index]) / 100

        train_en_feat_list.append(cluster_en_feat_list[index][:int(80 * clip)])
        train_en_ts_list.append(cluster_en_ts_list[index][:int(80 * clip)])
        train_en_location_list.append(cluster_en_location_list[index][:int(80 * clip)])
        train_de_feat_list.append(cluster_de_feat_list[index][:int(80 * clip)])
        train_de_ts_list.append(cluster_de_ts_list[index][:int(80 * clip)])
        train_de_location_list.append(cluster_de_location_list[index][:int(80 * clip)])
        train_label_convert_list.append(cluster_label_convert_list[index][:int(80 * clip)])
        train_mask_convert_list.append(cluster_mask_convert_list[index][:int(80 * clip)])

        valid_en_feat_list.append(cluster_en_feat_list[index][:int(80 * clip)])
        valid_en_ts_list.append(cluster_en_ts_list[index][:int(80 * clip)])
        valid_en_location_list.append(cluster_en_location_list[index][:int(80 * clip)])
        valid_de_feat_list.append(cluster_de_feat_list[index][:int(80 * clip)])
        valid_de_ts_list.append(cluster_de_ts_list[index][:int(80 * clip)])
        valid_de_location_list.append(cluster_de_location_list[index][:int(80 * clip)])
        valid_label_convert_list.append(cluster_label_convert_list[index][:int(80 * clip)])
        valid_mask_convert_list.append(cluster_mask_convert_list[index][:int(80 * clip)])

        test_en_feat_list.append(cluster_en_feat_list[index][int(args.proportion * clip):])
        test_en_ts_list.append(cluster_en_ts_list[index][int(args.proportion * clip):])
        test_en_location_list.append(cluster_en_location_list[index][int(args.proportion * clip):])
        test_de_feat_list.append(cluster_de_feat_list[index][int(args.proportion * clip):])
        test_de_ts_list.append(cluster_de_ts_list[index][int(args.proportion * clip):])
        test_de_location_list.append(cluster_de_location_list[index][int(args.proportion * clip):])
        test_label_convert_list.append(cluster_label_convert_list[index][int(args.proportion * clip):])
        test_mask_convert_list.append(cluster_mask_convert_list[index][int(args.proportion * clip):])

    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []

    for index in range(len(clusters_list)):
        train_loader = DataLoader(data_loader_1(train_en_feat_list[index],
                                                train_en_ts_list[index],
                                                train_en_location_list[index],
                                                train_de_feat_list[index],
                                                train_de_ts_list[index],
                                                train_de_location_list[index],
                                                train_label_convert_list[index],
                                                train_mask_convert_list[index]), batch_size=batch_size[0], shuffle=True,
                                  drop_last=True)

        valid_loader = DataLoader(data_loader_1(valid_en_feat_list[index],
                                                valid_en_ts_list[index],
                                                valid_en_location_list[index],
                                                valid_de_feat_list[index],
                                                valid_de_ts_list[index],
                                                valid_de_location_list[index],
                                                valid_label_convert_list[index],
                                                valid_mask_convert_list[index]), batch_size=batch_size[1], shuffle=True,
                                  drop_last=True)

        test_loader = DataLoader(data_loader_1(test_en_feat_list[index],
                                               test_en_ts_list[index],
                                               test_en_location_list[index],
                                               test_de_feat_list[index],
                                               test_de_ts_list[index],
                                               test_de_location_list[index],
                                               test_label_convert_list[index],
                                               test_mask_convert_list[index]), batch_size=batch_size[2], shuffle=True,
                                 drop_last=True)

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)

    train_loader_all = DataLoader(data_loader_1(np.concatenate(train_en_feat_list),
                                                np.concatenate(train_en_ts_list),
                                                np.concatenate(train_en_location_list),
                                                np.concatenate(train_de_feat_list),
                                                np.concatenate(train_de_ts_list),
                                                np.concatenate(train_de_location_list),
                                                np.concatenate(train_label_convert_list),
                                                np.concatenate(train_mask_convert_list)), batch_size=batch_size[0],
                                  shuffle=True)

    valid_loader_all = DataLoader(data_loader_1(np.concatenate(valid_en_feat_list),
                                                np.concatenate(valid_en_ts_list),
                                                np.concatenate(valid_en_location_list),
                                                np.concatenate(valid_de_feat_list),
                                                np.concatenate(valid_de_ts_list),
                                                np.concatenate(valid_de_location_list),
                                                np.concatenate(valid_label_convert_list),
                                                np.concatenate(valid_mask_convert_list)), batch_size=batch_size[1],
                                  shuffle=True)

    test_loader_all = DataLoader(data_loader_1(np.concatenate(test_en_feat_list),
                                               np.concatenate(test_en_ts_list),
                                               np.concatenate(test_en_location_list),
                                               np.concatenate(test_de_feat_list),
                                               np.concatenate(test_de_ts_list),
                                               np.concatenate(test_de_location_list),
                                               np.concatenate(test_label_convert_list),
                                               np.concatenate(test_mask_convert_list)), batch_size=batch_size[2],
                                 shuffle=True)

    return train_loader_list, valid_loader_list, test_loader_list, \
        train_loader_all, valid_loader_all, test_loader_all

def stack_loader_k_medoid_2(en_feat_list, en_ts_list, en_location_list,
                               de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list,
                               clusters_list, batch_size=[128, 128, 1]):
    train_en_feat_list = []
    train_en_ts_list = []
    train_en_location_list = []
    train_de_feat_list = []
    train_de_ts_list = []
    train_de_location_list = []
    train_label_convert_list = []
    train_mask_convert_list = []

    valid_en_feat_list = []
    valid_en_ts_list = []
    valid_en_location_list = []
    valid_de_feat_list = []
    valid_de_ts_list = []
    valid_de_location_list = []
    valid_label_convert_list = []
    valid_mask_convert_list = []

    test_en_feat_list = []
    test_en_ts_list = []
    test_en_location_list = []
    test_de_feat_list = []
    test_de_ts_list = []
    test_de_location_list = []
    test_label_convert_list = []
    test_mask_convert_list = []

    cluster_en_feat_list = []
    cluster_en_ts_list = []
    cluster_en_location_list = []
    cluster_de_feat_list = []
    cluster_de_ts_list = []
    cluster_de_location_list = []
    cluster_label_convert_list = []
    cluster_mask_convert_list = []

    for clusters in clusters_list:
        middle_en_feat_list = []
        middle_en_ts_list = []
        middle_en_location_list = []
        middle_de_feat_list = []
        middle_de_ts_list = []
        middle_de_location_list = []
        middle_label_convert_list = []
        middle_mask_convert_list = []
        for index in clusters:
            middle_en_feat_list.append(en_feat_list[index])
            middle_en_ts_list.append(en_ts_list[index])
            middle_en_location_list.append(en_location_list[index])
            middle_de_feat_list.append(de_feat_list[index])
            middle_de_ts_list.append(de_ts_list[index])
            middle_de_location_list.append(de_location_list[index])
            middle_label_convert_list.append(label_convert_list[index])
            middle_mask_convert_list.append(mask_convert_list[index])
        cluster_en_feat_list.append(np.concatenate(tuple(middle_en_feat_list), axis=0))
        cluster_en_ts_list.append(np.concatenate(tuple(middle_en_ts_list), axis=0))
        cluster_en_location_list.append(np.concatenate(tuple(middle_en_location_list), axis=0))
        cluster_de_feat_list.append(np.concatenate(tuple(middle_de_feat_list), axis=0))
        cluster_de_ts_list.append(np.concatenate(tuple(middle_de_ts_list), axis=0))
        cluster_de_location_list.append(np.concatenate(tuple(middle_de_location_list), axis=0))
        cluster_label_convert_list.append(np.concatenate(middle_label_convert_list))
        cluster_mask_convert_list.append(np.concatenate(middle_mask_convert_list))

    for index in range(len(clusters_list)):
        shuffle_list = np.random.permutation(np.arange(len(cluster_en_feat_list[index])))
        cluster_en_feat_list[index] = cluster_en_feat_list[index][shuffle_list]
        cluster_en_ts_list[index] = cluster_en_ts_list[index][shuffle_list]
        cluster_en_location_list[index] = cluster_en_location_list[index]
        cluster_de_feat_list[index] = cluster_de_feat_list[index]
        cluster_de_ts_list[index] = cluster_de_ts_list[index]
        cluster_de_location_list[index] = cluster_de_location_list[index]
        cluster_label_convert_list[index] = cluster_label_convert_list[index]
        cluster_mask_convert_list[index] = cluster_mask_convert_list[index]

        clip = len(cluster_en_feat_list[index]) / 100

        train_en_feat_list.append(cluster_en_feat_list[index][0:int(60 * clip)])
        train_en_ts_list.append(cluster_en_ts_list[index][0:int(60 * clip)])
        train_en_location_list.append(cluster_en_location_list[index][0:int(60 * clip)])
        train_de_feat_list.append(cluster_de_feat_list[index][0:int(60 * clip)])
        train_de_ts_list.append(cluster_de_ts_list[index][0:int(60 * clip)])
        train_de_location_list.append(cluster_de_location_list[index][0:int(60 * clip)])
        train_label_convert_list.append(cluster_label_convert_list[index][0:int(60 * clip)])
        train_mask_convert_list.append(cluster_mask_convert_list[index][0:int(60 * clip)])

        valid_en_feat_list.append(cluster_en_feat_list[index][int(60 * clip):int(80 * clip)])
        valid_en_ts_list.append(cluster_en_ts_list[index][int(60 * clip) :int(80 * clip)])
        valid_en_location_list.append(cluster_en_location_list[index][int(60 * clip):int(80 * clip)])
        valid_de_feat_list.append(cluster_de_feat_list[index][int(60 * clip):int(80 * clip)])
        valid_de_ts_list.append(cluster_de_ts_list[index][int(60 * clip):int(80 * clip)])
        valid_de_location_list.append(cluster_de_location_list[index][int(60 * clip):int(80 * clip)])
        valid_label_convert_list.append(cluster_label_convert_list[index][int(60 * clip):int(80 * clip)])
        valid_mask_convert_list.append(cluster_mask_convert_list[index][int(60 * clip):int(80 * clip)])

        test_en_feat_list.append(cluster_en_feat_list[index][int(80 * clip):])
        test_en_ts_list.append(cluster_en_ts_list[index][int(80 * clip):])
        test_en_location_list.append(cluster_en_location_list[index][int(80 * clip):])
        test_de_feat_list.append(cluster_de_feat_list[index][int(80 * clip):])
        test_de_ts_list.append(cluster_de_ts_list[index][int(80 * clip):])
        test_de_location_list.append(cluster_de_location_list[index][int(80 * clip):])
        test_label_convert_list.append(cluster_label_convert_list[index][int(80 * clip):])
        test_mask_convert_list.append(cluster_mask_convert_list[index][int(80 * clip):])

    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []

    for index in range(len(clusters_list)):
        train_loader = DataLoader(data_loader_1(train_en_feat_list[index],
                                                train_en_ts_list[index],
                                                train_en_location_list[index],
                                                train_de_feat_list[index],
                                                train_de_ts_list[index],
                                                train_de_location_list[index],
                                                train_label_convert_list[index],
                                                train_mask_convert_list[index]), batch_size=batch_size[0], shuffle=True)

        valid_loader = DataLoader(data_loader_1(valid_en_feat_list[index],
                                                valid_en_ts_list[index],
                                                valid_en_location_list[index],
                                                valid_de_feat_list[index],
                                                valid_de_ts_list[index],
                                                valid_de_location_list[index],
                                                valid_label_convert_list[index],
                                                valid_mask_convert_list[index]), batch_size=batch_size[1], shuffle=True)

        test_loader = DataLoader(data_loader_1(test_en_feat_list[index],
                                               test_en_ts_list[index],
                                               test_en_location_list[index],
                                               test_de_feat_list[index],
                                               test_de_ts_list[index],
                                               test_de_location_list[index],
                                               test_label_convert_list[index],
                                               test_mask_convert_list[index]), batch_size=batch_size[2], shuffle=True)

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)

    train_loader_all = DataLoader(data_loader_1(np.concatenate(train_en_feat_list),
                                                np.concatenate(train_en_ts_list),
                                                np.concatenate(train_en_location_list),
                                                np.concatenate(train_de_feat_list),
                                                np.concatenate(train_de_ts_list),
                                                np.concatenate(train_de_location_list),
                                                np.concatenate(train_label_convert_list),
                                                np.concatenate(train_mask_convert_list)), batch_size=batch_size[0], shuffle=True)

    valid_loader_all = DataLoader(data_loader_1(np.concatenate(valid_en_feat_list),
                                                np.concatenate(valid_en_ts_list),
                                                np.concatenate(valid_en_location_list),
                                                np.concatenate(valid_de_feat_list),
                                                np.concatenate(valid_de_ts_list),
                                                np.concatenate(valid_de_location_list),
                                                np.concatenate(valid_label_convert_list),
                                                np.concatenate(valid_mask_convert_list)), batch_size=batch_size[1], shuffle=True)

    test_loader_all = DataLoader(data_loader_1(np.concatenate(test_en_feat_list),
                                               np.concatenate(test_en_ts_list),
                                               np.concatenate(test_en_location_list),
                                               np.concatenate(test_de_feat_list),
                                               np.concatenate(test_de_ts_list),
                                               np.concatenate(test_de_location_list),
                                               np.concatenate(test_label_convert_list),
                                               np.concatenate(test_mask_convert_list)), batch_size=batch_size[2], shuffle=True)

    return train_loader_list, valid_loader_list, test_loader_list, \
           train_loader_all, valid_loader_all, test_loader_all


def stack_loader_DANN(feat_list, time_list, location_list, label_list, label_mask_list, clusters_list, batch_size=[128, 128, 1]):
    en_in_list = []
    de_in_list = []
    en_timestamp_list = []
    de_timestamp_list = []
    en_de_location_list = []
    label = []
    label_mask = []

    for index in range(len(feat_list)):
        index_list = []
        for i in range(label_list[index].shape[0]):
            if label_mask_list[index][i].sum() != 0:
                index_list.append(i)
        random.shuffle(index_list)
        en_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        de_in_list.append(feat_list[index].reshape(-1, 5, 34)[index_list])
        en_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        de_timestamp_list.append(time_list[index].reshape(-1, 5, 4)[index_list])
        en_de_location_list.append(location_list[index].reshape(-1, 5, 4)[index_list])
        label.append(label_list[index][index_list])
        label_mask.append(label_mask_list[index][index_list])

        # shuffle_list = list(range(en_in_list[-1].shape[0]))
        # random.shuffle(shuffle_list)
        pass
    train_en_in_list = []
    train_de_in_list = []
    train_en_timestamp_list = []
    train_de_timestamp_list = []
    train_en_de_location_list = []
    train_label_list = []
    train_label_mask_list = []

    valid_en_in_list = []
    valid_de_in_list = []
    valid_en_timestamp_list = []
    valid_de_timestamp_list = []
    valid_en_de_location_list = []
    valid_label_list = []
    valid_label_mask_list = []

    test_en_in_list = []
    test_de_in_list = []
    test_en_timestamp_list = []
    test_de_timestamp_list = []
    test_en_de_location_list = []
    test_label_list = []
    test_label_mask_list = []
    train_list_loader = []
    for index in range(len(label)):
        clip = int(len(label[index]) / 10)
        train_en_in_list.append(en_in_list[index][:8 * clip])
        train_de_in_list.append(de_in_list[index][:8 * clip])
        train_en_timestamp_list.append(en_timestamp_list[index][:8 * clip])
        train_de_timestamp_list.append(de_timestamp_list[index][:8 * clip])
        train_en_de_location_list.append(en_de_location_list[index][:8 * clip])
        train_label_list.append(label[index][:8 * clip])
        train_label_mask_list.append(label_mask[index][:8 * clip])

        '''train_list_loader.append(DataLoader(data_loader(en_in_list[index][:8 * clip],
                               de_in_list[index][:8 * clip],
                               en_timestamp_list[index][:8 * clip],
                               de_timestamp_list[index][:8 * clip],
                               en_de_location_list[index][:8 * clip],
                               label[index][:8 * clip],
                               label_mask[index][:8 * clip]), batch_size = batch_size[0], shuffle=True))'''

        valid_en_in_list.append(en_in_list[index][8 * clip:9 * clip])
        valid_de_in_list.append(de_in_list[index][8 * clip:9 * clip])
        valid_en_timestamp_list.append(en_timestamp_list[index][8 * clip:9 * clip])
        valid_de_timestamp_list.append(de_timestamp_list[index][8 * clip:9 * clip])
        valid_en_de_location_list.append(en_de_location_list[index][8 * clip:9 * clip])
        valid_label_list.append(label[index][8 * clip:9 * clip])
        valid_label_mask_list.append(label_mask[index][8 * clip:9 * clip])

        test_en_in_list.append(en_in_list[index][9 * clip:])
        test_de_in_list.append(de_in_list[index][9 * clip:])
        test_en_timestamp_list.append(en_timestamp_list[index][9 * clip:])
        test_de_timestamp_list.append(de_timestamp_list[index][9 * clip:])
        test_en_de_location_list.append(en_de_location_list[index][9 * clip:])
        test_label_list.append(label[index][9 * clip:])
        test_label_mask_list.append(label_mask[index][9 * clip:])

    train_en_in_all = np.concatenate(train_en_in_list)
    train_de_in_all = np.concatenate(train_de_in_list)
    train_en_timestamp_all = np.concatenate(train_en_timestamp_list)
    train_de_timestamp_all = np.concatenate(train_de_timestamp_list)
    train_en_de_location_all = np.concatenate(train_en_de_location_list)
    train_label_all = np.concatenate(train_label_list)
    train_label_mask_all = np.concatenate(train_label_mask_list)

    train_all_loader = DataLoader(data_loader(train_en_in_all,
                                              train_de_in_all,
                                              train_en_timestamp_all,
                                              train_de_timestamp_all,
                                              train_en_de_location_all,
                                              train_label_all,
                                              train_label_mask_all), batch_size=batch_size[0], shuffle=True)

    valid_loader = DataLoader(data_loader(np.concatenate(valid_en_in_list),
                                          np.concatenate(valid_de_in_list),
                                          np.concatenate(valid_en_timestamp_list),
                                          np.concatenate(valid_de_timestamp_list),
                                          np.concatenate(valid_en_de_location_list),
                                          np.concatenate(valid_label_list),
                                          np.concatenate(valid_label_mask_list)), batch_size=batch_size[1], shuffle=False)

    test_loader = DataLoader(data_loader(np.concatenate(test_en_in_list),
                                         np.concatenate(test_de_in_list),
                                         np.concatenate(test_en_timestamp_list),
                                         np.concatenate(test_de_timestamp_list),
                                         np.concatenate(test_en_de_location_list),
                                         np.concatenate(test_label_list),
                                         np.concatenate(test_label_mask_list)), batch_size=batch_size[2], shuffle=False)

    for i, cluster_list_i in enumerate(clusters_list):
        list_en_in = []
        list_de_in = []
        list_en_ts = []
        list_de_ts = []
        list_locat = []
        list_label = []
        list_mask = []
        for ii in cluster_list_i:
            list_en_in.append(train_en_in_list[ii])
            list_de_in.append(train_de_in_list[ii])
            list_en_ts.append(train_en_timestamp_list[ii])
            list_de_ts.append(train_de_timestamp_list[ii])
            list_locat.append(train_en_de_location_list[ii])
            list_label.append(train_label_list[ii])
            list_mask.append(train_label_mask_list[ii])
        domain = np.zeros((1, len(clusters_list)))
        domain[0, i] = 1
        domain = domain.repeat(np.concatenate(list_en_in).shape[0], axis=0)
        train_list_loader.append(DataLoader(data_loader(np.concatenate(list_en_in),
                                                        np.concatenate(list_de_in),
                                                        np.concatenate(list_en_ts),
                                                        np.concatenate(list_de_ts),
                                                        np.concatenate(list_locat),
                                                        np.concatenate(list_label),
                                                        np.concatenate(list_mask),
                                                        domain), batch_size=batch_size[0], shuffle=True))

    return train_list_loader, train_all_loader, valid_loader, test_loader

def labelprocess(label_dir):
    triangular_std = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]])
    dt = pd.read_csv(label_dir, header=None, dtype=str)
    label = dt.values.reshape(dt.values.shape)
    label_reshape = label.reshape(-1, 5, 5)
    label_mask = np.zeros(label.shape, dtype='int')
    label_int = np.ones(label.shape, dtype='int')
    for i in range(dt.shape[0]):
        mask = np.copy(triangular_std)
        for index in range(5):
            if label_reshape[i, index, 4 - index] is 'N':
                mask[:, 4 - index] = np.zeros((1, 5))
        label_mask[i, :] = mask.reshape(1, -1).squeeze()
        for j in range(label.shape[1]):
            if label[i, j] is not '1':
                label_int[i, j] = 0
    return label_int, label_mask


def seq_convert(seq_dir, vocab_list):
    vec_dim = 0
    for vocab in vocab_list:
        vec_dim += len(vocab)
    dt = pd.read_csv(seq_dir, dtype=str)
    data_index = []
    for i in range(dt.values.shape[0]):
        if dt.values[i, 0] != '-1':
            data_index.append(i)
    if dt.values.shape[1] < 19:
        data = dt.values[data_index, 4:9]
        time = dt.values[data_index, 3]
        location = dt.values[data_index, 14:18]
        feat_array = np.zeros((data.shape[0], vec_dim))
        in_time_array = np.zeros((data.shape[0], 4))
        location_array = np.zeros((data.shape[0], 4))
    else:
        data = dt.values[data_index, 7:12]
        time = dt.values[data_index, 6]
        location = dt.values[data_index, 17:21]
        feat_array = np.zeros((data.shape[0], vec_dim))
        in_time_array = np.zeros((data.shape[0], 4))
        location_array = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):
        index = 0
        for j in range(data.shape[1]):
            a = np.nan
            b = a is not np.nan
            if data[i, j] is not np.nan:
                if j is 3:
                    if int(data[i, 3]) < 10000:
                        data[i, j] = '1'
                    elif 10000 <= int(data[i, 3]) and int(data[i, 3]) < 20000:
                        data[i, 3] = '2'
                    elif 20000 <= int(data[i, 3]) and int(data[i, 3]) < 30000:
                        data[i, 3] = '3'
                    else:
                        data[i, 3] = '4'
                feat_array[i, index + vocab_list[j][str.strip(data[i, j])]] = 1
                index += len(vocab_list[j])
    for i in range(data.shape[0]):
        if time[i] is not np.nan:
            time1 = datetime.datetime.strptime(str.lstrip(time[i]), '%Y/%m/%d %H:%M')
            in_time_array[i, 0] = time1.minute / 59.0
            in_time_array[i, 1] = time1.hour / 23.0
            in_time_array[i, 2] = (time1.day - 1) / 30.0
            in_time_array[i, 3] = time1.month / 11.0
        else:
            in_time_array[i, 0] = 0.0
            in_time_array[i, 1] = 0.0
            in_time_array[i, 2] = 0.0
            in_time_array[i, 3] = 0.0

    for i in range(data.shape[0]):
        if data[i, 0] is not np.nan:

            location_array[i, 0] = int(location[i, 0])
            location_array[i, 1] = int(location[i, 1])
            location_array[i, 2] = int(location[i, 2])
            location_array[i, 3] = int(location[i, 3])
            '''
            location_array[i, 0] = (float(location[i, 0])-1)/60
            location_array[i, 1] = (float(location[i, 1])-1)/179
            location_array[i, 2] = (float(location[i, 2])-1)/9
            location_array[i, 3] = (float(location[i, 3])-1)/4
            '''
        else:
            location_array[i, 0] = 0.0
            location_array[i, 1] = 0.0
            location_array[i, 2] = 0.0
            location_array[i, 3] = 0.0

    '''
    ##########bay###############
    feat_array = feat_array.reshape(-1, 10, 5, feat_array.shape[-1])
    in_time_array = in_time_array.reshape(-1, 10, 5, in_time_array.shape[-1])
    location_array = location_array.reshape(-1, 10, 5, location.shape[-1])
    ############################
    '''
    ##########stack##############
    feat_array = feat_array.reshape(-1, 5, feat_array.shape[-1])
    in_time_array = in_time_array.reshape(-1, 5, in_time_array.shape[-1])
    location_array = location_array.reshape(-1, 5, location.shape[-1])
    ############################
    return feat_array, in_time_array, location_array

def container_data(file_dir):
    file_list = os.listdir(file_dir)
    area_file_dir = file_dir + '/' + file_list[2]
    area_dir_list_old = os.listdir(area_file_dir)
    area_dir_list = []
    for i in range(1, len(area_dir_list_old) + 1):
        for j in range(0, len(area_dir_list_old)):
            if int(area_dir_list_old[j][:-4][10:]) == i:
                area_dir_list.append(area_dir_list_old[j])

    dict = {}
    container_list = []
    index_list = []

    for area_dir in area_dir_list:
        feat, time, location = seq_convert(area_file_dir + '/' + area_dir, vocab)
        feat = feat.reshape(-1, feat.shape[-1])
        for index in range(feat.shape[0]):
            if feat[index].sum() > 0:
                index_list.append(index)
        container_list.append(feat[index_list])
        index_list.clear()
    return container_list

def area_dis(container_list, loss_type='mmd'):
    heat_matrix = torch.zeros((len(container_list), len(container_list)), dtype=float)
    criterion_transder = TransferLoss(loss_type=loss_type, input_dim=container_list[0].shape[-1])
    for i in range(len(container_list)):
        for j in range(len(container_list)):
            heat_matrix[i, j] = criterion_transder.compute(torch.tensor(container_list[i], dtype=torch.float32), torch.tensor(container_list[j], dtype=torch.float32))
    return heat_matrix

def heatmap(matrix=None):
    sns.set_theme()
    mask = np.zeros_like(matrix)
    mask[np.tril_indices_from(mask)] = True
    ax = sns.heatmap(matrix, mask=mask)
    ax = plt.gca()
    plt.show()

def container_embedding(container_list, embedding_dir='C:/Users/Administrator/Desktop/partData/embeddingmatrix/embedding_matrix_128.csv'):
    embedding_matrix = pd.read_csv(embedding_dir, header=None, dtype=float)
    embedding_matrix = torch.from_numpy(embedding_matrix.values)
    for i in range(len(container_list)):
        container_list[i] = torch.mm(torch.tensor(container_list[i]).float(), embedding_matrix.float())
    return container_list

class data_loader(Dataset):
    def __init__(self, en_in, de_in, en_ts, de_ts, en_de_location, label, label_mask, domain=None):
        self.en_in = en_in
        self.de_in = de_in
        self.en_ts = en_ts
        self.de_ts = de_ts
        self.en_de_location = en_de_location
        self.label = label
        self.label_mask = label_mask
        self.domain = domain

        self.en_in = torch.tensor(
            self.en_in, dtype=torch.float)
        self.de_in = torch.tensor(
            self.de_in, dtype=torch.float)
        self.en_ts = torch.tensor(
            self.en_ts, dtype=torch.float)
        self.de_ts = torch.tensor(
            self.de_ts, dtype=torch.float)
        self.en_de_location = torch.tensor(
            self.en_de_location, dtype=torch.float)
        self.label = torch.tensor(
            self.label, dtype=torch.float)
        self.label_mask = torch.tensor(
            self.label_mask, dtype=torch.float)

        for i in range(len(self.en_de_location)):
            for ii in range(5):
                if self.en_de_location[i, ii].sum() > 0:
                    '''
                    self.en_de_location[i, ii, 0] = (self.en_de_location[i, ii, 0] - 1) / 60
                    self.en_de_location[i, ii, 1] = (self.en_de_location[i, ii, 1] - 1) / 179
                    self.en_de_location[i, ii, 2] = (self.en_de_location[i, ii, 2] - 1) / 9
                    self.en_de_location[i, ii, 3] = (self.en_de_location[i, ii, 3] - 1) / 4
                    '''
                    pass

        if domain is not None:
            self.domain = torch.tensor(
                self.domain, dtype=torch.float)

    def __getitem__(self, index):
        en_in = self.en_in[index]
        de_in = self.de_in[index]
        en_ts = self.en_ts[index]
        de_ts = self.de_ts[index]
        en_de_location = self.en_de_location[index]
        label = self.label[index]
        label_mask = self.label_mask[index]
        if self.domain is None:
            return en_in, de_in, en_ts, de_ts, en_de_location, label, label_mask
        else:
            domain = self.domain[index]
            return en_in, de_in, en_ts, de_ts, en_de_location, label, label_mask, domain



    def __len__(self):
        return len(self.label)

class data_loader_1(Dataset):
    def __init__(self, en_in, en_ts, en_loc, de_in, de_ts, de_loc, label, label_mask):
        self.en_in = en_in
        self.de_in = de_in
        self.en_ts = en_ts
        self.de_ts = de_ts
        self.en_loc = en_loc
        self.de_loc = de_loc
        self.label = label
        self.label_mask = label_mask

        self.en_in = torch.tensor(
            self.en_in, dtype=torch.float)
        self.de_in = torch.tensor(
            self.de_in, dtype=torch.float)
        self.en_ts = torch.tensor(
            self.en_ts, dtype=torch.float)
        self.de_ts = torch.tensor(
            self.de_ts, dtype=torch.float)
        self.en_loc = torch.tensor(
            self.en_loc, dtype=torch.float)
        self.de_loc = torch.tensor(
            self.de_loc, dtype=torch.float)
        self.label = torch.tensor(
            self.label, dtype=torch.float)
        self.label_mask = torch.tensor(
            self.label_mask, dtype=torch.float)

    def __getitem__(self, index):
        en_in = self.en_in[index]
        de_in = self.de_in[index]
        en_ts = self.en_ts[index]
        de_ts = self.de_ts[index]
        en_loc = self.en_loc[index]
        de_loc = self.de_loc[index]
        label = self.label[index]
        label_mask = self.label_mask[index]
        return en_in, en_ts, en_loc, de_in, de_ts, de_loc, label, label_mask

    def __len__(self):
        return len(self.label)

def sample_select(target_feat, target_time, taget_location, target_label, target_label_mask, \
                  other_feat, other_time, other_location, other_label, other_label_mask, \
                  dropout_rate=0.9, loss_type='mmd'):
    other_feat_select = other_feat.reshape(other_feat.shape[0], -1)
    other_time_select = other_time.reshape(other_time.shape[0], -1)
    other_location_select = other_location.reshape(other_location.shape[0], -1)
    other_label_select = other_label.reshape(other_time.shape[0], -1)
    other_label_mask_select = other_label_mask.reshape(other_label_mask.shape[0], -1)
    target_feat = target_feat.reshape(target_feat.shape[0], -1)
    num_dropout = int(len(other_feat)*dropout_rate)
    criterion_transder = TransferLoss(loss_type=loss_type, input_dim=other_feat_select.shape[-1])
    div_std = criterion_transder.compute(torch.tensor(target_feat, dtype=torch.float32), \
                                         torch.tensor(other_feat_select, dtype=torch.float32)).item()
    for i in range(num_dropout):
        max_div = 0
        delete = None
        for flag in range(other_feat_select.shape[0]):
            div = criterion_transder.compute(torch.tensor(target_feat, dtype=torch.float32), \
                                         torch.tensor(np.delete(other_feat_select, flag, axis=0), dtype=torch.float32)).item()
            if div_std - div > max_div:
                max_div = div_std - div
                delete = flag
        if max_div == 0:
            break
        else:
            other_feat_select = np.delete(other_feat_select, delete, axis=0)
            other_time_select = np.delete(other_time_select, delete, axis=0)
            other_location_select = np.delete(other_location_select, delete, axis=0)
            other_label_select = np.delete(other_label_select, delete, axis=0)
            other_label_mask_select = np.delete(other_label_mask_select, delete, axis=0)
    pass
    div_new = criterion_transder.compute(torch.tensor(target_feat, dtype=torch.float32), \
                                         torch.tensor(other_feat_select, dtype=torch.float32)).item()
    return other_feat_select.reshape(-1, 5, other_feat.shape[-1]), \
        other_time_select.reshape(-1, 5, other_time.shape[-1]), \
        other_location_select.reshape(-1, 5, other_location.shape[-1]), \
        other_label_select.reshape(-1, other_label.shape[-1]), \
        other_label_mask_select.reshape(-1, other_label_mask.shape[-1])
