import random
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
IYC_CSZ_CSIZECD = {'20': 0, '40': 1, '45': 2}
IYC_CTYPECD = {'BU': 0, 'FR': 1, 'GP': 2, 'HC': 3, 'OT': 4, 'PF': 5, 'TK': 6, 'TU': 7, 'RF': 8}
IYC_CHEIGHTCD = {'HQ': 0, 'LQ': 1, 'MQ': 2, 'PQ': 3}
IYC_CWEIGHT = {'1': 0, '2': 1, '3': 2, '4': 3}
IYC_STS_CSTATUSCD = {'CF': 0, 'EE': 1, 'IE': 2, 'IF': 3, 'IZ': 4, 'NF': 5, 'OE': 6, 'OF': 7, 'OZ': 8, 'RE': 9, 'RF': 10, 'T': 11, 'TE': 12, 'ZE': 13}
vocab = [IYC_CSZ_CSIZECD, IYC_CTYPECD, IYC_CHEIGHTCD, IYC_CWEIGHT, IYC_STS_CSTATUSCD]

def calculate_acc(Model, data_loader):
    flag_acc = 0
    flag_all = 0
    for data in data_loader:
        en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(Model.device)
        de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(Model.device)
        en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(Model.device)
        de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(Model.device)
        location = data[4].reshape(data[4].shape[0], -1, data[4].shape[-1]).to(Model.device)
        label = data[5].to(Model.device)
        label_mask = data[6].to(Model.device)

        output = Model.get_output(en_in, en_ts)
        output = output.squeeze()
        for i in [4, 8, 12, 16, 20]:
            if label_mask[0, i] != 0.0:
                flag_all += 1
                if output[i] > 0.5:
                    output[i] = 1
                else:
                    output[i] = 0

                if output[i] - label[0, i] == 0:
                    flag_acc += 1
    return flag_acc / flag_all

def calculate_rehandle_acc_2(Model, data_loader):
    flag_acc_1 = 0
    flag_acc_2 = 0
    flag_all = 0
    mae_list_1 = []
    mae_list_2 = []
    mse_list_1 = []
    mse_list_2 = []
    retrieval_mae_list = []
    retrieval_mse_list = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for data in data_loader:
        en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(Model.device)
        de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(Model.device)
        en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(Model.device)
        de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(Model.device)
        location = data[4].reshape(data[4].shape[0], -1, data[4].shape[-1]).to(Model.device)
        label = data[5].to(Model.device)
        label_mask = data[6].to(Model.device)

        output, _ = Model.get_output(en_in, en_ts)
        pred= output.reshape(-1, 5, 5)
        mask = label_mask.reshape(-1, 5, 5)

        p_1, p_2 = torch.zeros([1, 5]).to('cuda:0'), torch.zeros([1, 5]).to('cuda:0')
        p_1[0, 1] = pred[0, 4, 0] * pred[0, 4, 1] * mask[0, 3, 1]
        p_1[0, 2] = pred[0, 4, 0] * pred[0, 4, 2] + pred[0, 3, 1] * pred[0, 3, 2] * mask[0, 2, 2]
        p_1[0, 3] = pred[0, 4, 0] * pred[0, 4, 3] + pred[0, 3, 1] * pred[0, 3, 3] + pred[0, 2, 2] * pred[0, 2, 3] * mask[0, 1, 3]
        p_1[0, 4] = pred[0, 4, 0] * pred[0, 4, 4] + pred[0, 3, 1] * pred[0, 3, 4] + pred[0, 2, 2] * pred[0, 2, 4] + pred[0, 1, 3] * pred[0, 1, 4] * mask[0, 0, 4]

        p_2[0, 1] = (1 - pred[0, 3, 1]) * (1 - (1 - pred[0, 4, 0])) * mask[0, 3, 1]
        p_2[0, 2] = (1 - pred[0, 2, 2]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1])) * mask[0, 2, 2]
        p_2[0, 3] = (1 - pred[0, 1, 3]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1]) * (1 - pred[0, 2, 2])) * mask[0, 1, 3]
        p_2[0, 4] = (1 - pred[0, 0, 4]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1]) * (1 - pred[0, 2, 2]) * (1 - pred[0, 1, 3])) * mask[0, 0, 4]

        loss_mae_1 = 0
        loss_mae_2 = 0
        loss_mse_1 = 0
        loss_mse_2 = 0
        loss_retrieval_mse = 0
        loss_retrieval_mae = 0
        flag = 0
        for i, value in enumerate([20, 16, 12, 8, 4]):
            if label_mask[0, value] != 0.0:
                loss_mse_1 += (p_1[0, i].item()+label[0, value].item()-1)**2
                loss_mse_2 += (p_2[0, i].item()+label[0, value].item()-1)**2
                loss_retrieval_mse += (output[value].item()-label[:, value].item())**2

                loss_mae_1 += abs(p_1[0, i].item()+label[0, value].item()-1)
                loss_mae_2 += abs(p_2[0, i].item()+label[0, value].item()-1)
                loss_retrieval_mae += abs(output[value].item()-label[:, value].item())

                flag_all += 1
                flag += 1
                if p_1[0, i] > 0.5:
                    p_1[0, i] = 1
                else:
                    p_1[0, i] = 0

                if p_2[0, i] > 0.5:
                    p_2[0, i] = 1
                else:
                    p_2[0, i] = 0

                if 1 - p_1[0, i] - label[0, value] == 0:
                    flag_acc_1 += 1
                if p_2[0, i] == 1 - label[0, value]:
                    if p_2[0, i] == 1:
                        TP += 1
                    else:
                        TN += 1
                    flag_acc_2 += 1
                else:
                    if p_2[0, i] == 1:
                        FP += 1
                    else:
                        FN += 1
        mae_list_1.append(loss_mae_1/flag)
        mae_list_2.append(loss_mae_2/flag)
        mse_list_1.append(loss_mse_1/flag)
        mse_list_2.append(loss_mse_2/flag)
        retrieval_mae_list.append(loss_retrieval_mae/flag)
        retrieval_mse_list.append(loss_retrieval_mse/flag)

    print('acc_1 =  %.5f              acc_2 = %.5f' % (flag_acc_1 / flag_all, flag_acc_2 / flag_all))
    print('mae_1 = %.5f' % (sum(mae_list_1)/len(mae_list_1)))
    print('mae_2 = %.5f' % (sum(mae_list_2) / len(mae_list_2)))
    print('mse_1 = %.5f' % (sum(mse_list_1) / len(mse_list_1)))
    print('mse_2 = %.5f' % (sum(mse_list_2) / len(mse_list_2)))
    print('retrieval_mae = %.5f' % (sum(retrieval_mae_list) / len(retrieval_mae_list)))
    print('retrieval_mse = %.5f' % (sum(retrieval_mse_list) / len(retrieval_mse_list)))
    print('TP = %d     FP = %d' % (TP, FP))
    print('FN = %d     TN = %d' % (FN, TN))

#data_loader = get_embeddingLoader("C:/Users/Administrator/Desktop/partData/contain_sequence.csv")
#vocab = get_vocab("C:/Users/Administrator/Desktop/partData/17/3Area_48_zdxqdatatest.csv")
#data_convert("C:/Users/Administrator/Desktop/partData/17/3Area_48_zdxqdatatest.csv", vocab, 'C:/Users/Administrator/Desktop/partData/17/1Area_48_zdxqdatatest.csv')
#data_convert_shuffle("C:/Users/Administrator/Desktop/partData/17/3Area_48_zdxqdatatest.csv", vocab, 'C:/Users/Administrator/Desktop/partData/17/1Area_48_zdxqdatatest.csv')
#visual_data("C:/Users/Administrator/Desktop/partData/1Area_48_zdxqdatatest(4).csv")
#process_label('C:/Users/Administrator/Desktop/partData/17/1Area_48_zdxqdatatest.csv')
pass
