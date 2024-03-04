import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

IYC_CSZ_CSIZECD = {'20': 0, '40': 1, '45': 2}
IYC_CTYPECD = {'BU': 0, 'FR': 1, 'GP': 2, 'HC': 3, 'OT': 4, 'PF': 5, 'TK': 6, 'TU': 7, 'RF': 8}
IYC_CHEIGHTCD = {'HQ': 0, 'LQ': 1, 'MQ': 2, 'PQ': 3}
IYC_CWEIGHT = {'1': 0, '2': 1, '3': 2}
IYC_STS_CSTATUSCD = {'CF': 0, 'EE': 1, 'IE': 2, 'IF': 3, 'IZ': 4, 'NF': 5, 'OE': 6, 'OF': 7, 'OZ': 8, 'RE': 9, 'RF': 10, 'T': 11, 'TE': 12, 'ZE': 13, 'TF':14}
vocab = [IYC_CSZ_CSIZECD, IYC_CTYPECD, IYC_CHEIGHTCD, IYC_CWEIGHT, IYC_STS_CSTATUSCD]

def data(file_dir):
    seq_dir = file_dir + '/' + 'container storage sequence.csv'
    label_dir = file_dir + '/' + 'label.csv'
    label, label_mask = labelprocess(label_dir)

    feat, time, location = seq_convert(seq_dir, vocab)
    for ii in range(len(feat)):
        mid_feature = np.zeros((feat.shape[1], feat.shape[2]))
        mid_time = np.zeros((time.shape[1], time.shape[2]))
        mid_location = np.zeros((location.shape[1], location.shape[2]))
        for iii in range(5):
            mid_feature[5 - iii - 1] = feat[ii, iii]
            mid_time[5 - iii - 1] = time[ii, iii]
            mid_location[5 - iii - 1] = location[ii, iii]
        feat[ii] = mid_feature
        time[ii] = mid_time
        location[ii] = mid_location

    return feat, time, location, label, label_mask

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
                    # elif 20000 <= int(data[i, 3]) and int(data[i, 3]) < 30000:
                    #     data[i, 3] = '3'
                    else:
                        data[i, 3] = '3'
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

def data_covert(feat_list, time_list, location_list, label_list, label_mask_list):#en:[000123] de:[00012] [000123]
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

    return en_feat_list[0], en_ts_list[0], en_location_list[0], de_feat_list[0], de_ts_list[0], de_location_list[0], label_convert_list[0], mask_convert_list[0]

def data_convert_loader(en_feat, en_ts, en_location, de_feat, de_ts, de_location, label_convert, mask_convert):
    loader = DataLoader(data_loader_1(en_feat, en_ts, en_location, de_feat, de_ts, de_location, label_convert, mask_convert), batch_size=1, shuffle=False)

    return loader

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

