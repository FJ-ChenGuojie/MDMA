import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import time
import pandas as pd
import datetime
from torch.utils.data import Dataset, DataLoader
from base.loss_transfer import TransferLoss

class Basic_Model(nn.Module):
    def __init__(self, device='cuda:0'):
        super(Basic_Model, self).__init__()
        IYC_CSZ_CSIZECD = {'20': 0, '40': 1, '45': 2}
        IYC_CTYPECD = {'BU': 0, 'FR': 1, 'GP': 2, 'HC': 3, 'OT': 4, 'PF': 5, 'TK': 6, 'TU': 7, 'RF': 8}
        IYC_CHEIGHTCD = {'HQ': 0, 'LQ': 1, 'MQ': 2, 'PQ': 3}
        IYC_CWEIGHT = {'1': 0, '2': 1, '3': 2, '4': 3}
        IYC_STS_CSTATUSCD = {'CF': 0, 'EE': 1, 'IE': 2, 'IF': 3, 'IZ': 4, 'NF': 5, 'OE': 6, 'OF': 7, 'OZ': 8, 'RE': 9, 'RF': 10, 'T': 11, 'TE': 12, 'ZE': 13}
        self.vocab = [IYC_CSZ_CSIZECD, IYC_CTYPECD, IYC_CHEIGHTCD, IYC_CWEIGHT, IYC_STS_CSTATUSCD]
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.retrieval_loss = []
        self.rehandle_loss = []
        self.device = device

    def get_output(self, input):
        embedding_out = self.embedding_layer(input.reshape(-1, self.n_input))
        embedding_out = embedding_out.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            embedding_out = gru_feature

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.sigmoid(self.fc_out(gru_out))
        return output

    def train_epoch(self, train_loader, valid_loader, optimizer, criterion, epoch):
        self.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
            de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
            en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
            de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
            location = data[4].reshape(data[4].shape[0], -1, data[4].shape[-1]).to(self.device)
            label = data[5].to(self.device)
            label_mask = data[6].to(self.device)

            rehandle_label = torch.ones((label.shape[0], 5)).to(self.device) - label[:, [20, 16, 12, 8, 4]]
            output = self.get_output(en_in, en_ts)

            p1, p2 = self.returning_rate(output, label_mask)

            loss_retrieval = criterion[0]((output * label_mask)[:, [20, 16, 12, 8, 4]], (label * label_mask)[:, [20, 16, 12, 8, 4]])
            #loss_rehandle = criterion[0]((output * d_label_mask)[:, [9, 13, 14, 17, 18, 19, 21, 22, 23, 24]], (d_label * d_label_mask)[:, [9, 13, 14, 17, 18, 19, 21, 22, 23, 24]])
            #loss_retrieval = criterion[0]((p2 * label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * label_mask[:, [20, 16, 12, 8, 4]]))
            loss_rehandle = criterion[0]((p1 * label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * label_mask[:, [20, 16, 12, 8, 4]]))
            loss_p = criterion[0](p1[:, 1:], p2[:, 1:])

            if epoch >= 0:
                all_loss = self.trade_off[0]*loss_retrieval + self.trade_off[1]*loss_rehandle + self.trade_off[2] * loss_p
            else:
                all_loss = self.trade_off[0]*loss_retrieval + self.trade_off[1]*loss_rehandle
            #loss = self.trade_off[0]*loss_retrieval + self.trade_off[1]*loss_rehandle
            optimizer.zero_grad()
            all_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.parameters(), 3.)
            optimizer.step()

            if i % 100 == 0:
                loss_valid = self.pred(valid_loader)

                self.valid_loss_list.append(loss_valid.item())
                self.train_loss_list.append(all_loss.item())
                self.p_loss_list.append(loss_p.item())
                self.rehandle_loss.append(loss_rehandle.item())
                self.retrieval_loss.append(loss_retrieval.item())

    def returning_rate(self, output, label_mask):
        p_1, p_2 = torch.zeros([output.shape[0], 5]).to(self.device), torch.zeros([output.shape[0], 5]).to(self.device)
        pred = output.reshape(-1, 5, 5)
        mask = label_mask.reshape(-1, 5, 5)

        for i in range(pred.shape[0]):
            p_1[i, 1] = pred[i, 4, 0] * pred[i, 4, 1] * mask[i, 3, 1]
            p_1[i, 2] = pred[i, 4, 0] * pred[i, 4, 2] + pred[i, 3, 1] * pred[i, 3, 2] * mask[i, 2, 2]
            p_1[i, 3] = pred[i, 4, 0] * pred[i, 4, 3] + pred[i, 3, 1] * pred[i, 3, 3] + pred[i, 2, 2] * pred[i, 2, 3] * mask[i, 1, 3]
            p_1[i, 4] = pred[i, 4, 0] * pred[i, 4, 4] + pred[i, 3, 1] * pred[i, 3, 4] + pred[i, 2, 2] * pred[i, 2, 4] + pred[i, 1, 3] * pred[i, 1, 4]*mask[i, 0, 4]
        for i in range(pred.shape[0]):
            p_2[i, 1] = (1 - pred[i, 3, 1]) * (1 - (1 - pred[i, 4, 0])) * mask[i, 3, 1]
            p_2[i, 2] = (1 - pred[i, 2, 2]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1])) * mask[i, 2, 2]
            p_2[i, 3] = (1 - pred[i, 1, 3]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1]) * (1-pred[i, 2, 2])) * mask[i, 1, 3]
            p_2[i, 4] = (1 - pred[i, 0, 4]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1]) * (1-pred[i, 2, 2]) * (1-pred[i, 1, 3])) * mask[i, 0, 4]
        return p_1, p_2

    def train_model(self, train_epoch, train_loader, valid_loader, test_loader):
        optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = [nn.MSELoss(), nn.L1Loss()]
        lr = self.lr
        min_loss = np.Inf
        stop_round = 0
        for epoch in range(train_epoch):
            lr = self.update_lr(epoch, lr)
            optimizer.defaults['lr'] = lr
            run_start = time.time()
            self.train_epoch(train_loader, valid_loader, optimizer, criterion, epoch)
            run_end = time.time()
            loss = self.pred(valid_loader)
            if loss < min_loss:
                min_loss = loss
                stop_round = 0
                torch.save(self, self.path)
            else:
                stop_round += 1
                if stop_round >= 5:
                    print('early stop')
                    break
            #self.calculate_acc(test_loader)
            print("########################################")
            self.calculate_rehandle_acc_2(test_loader)
            print('runtime = %f ####epoch = %d #####train_loss = %.5f' % (run_end-run_start, epoch, loss))
            print('stop_round = %d#### min_loss = %.5f' % (stop_round, min_loss))
            print("########################################")

    def pred(self, data_loader):
        total_loss = torch.zeros(1).to(self.device)
        criterion = [nn.MSELoss(), nn.L1Loss()]

        with torch.no_grad():
            for data in data_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].reshape(data[4].shape[0], -1, data[4].shape[-1]).to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)

                rehandle_label = torch.ones((label.shape[0], 5)).to(self.device) - label[:, [20, 16, 12, 8, 4]]
                output = self.get_output(en_in, en_ts)
                p1, p2 = self.returning_rate(output, label_mask)
                loss_p = criterion[0](p1[:, 1:], p2[:, 1:])
                loss_retrieval = criterion[0]((output * label_mask)[:, [20, 16, 12, 8, 4]], (label * label_mask)[:, [20, 16, 12, 8, 4]])
                #loss_rehandle = criterion[0]((output * label_mask)[:, [9, 13, 14, 17, 18, 19, 21, 22, 23, 24]], (label * label_mask)[:, [9, 13, 14, 17, 18, 19, 21, 22, 23, 24]])
                #loss_retrieval = criterion[0]((p2 * label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * label_mask[:, [20, 16, 12, 8, 4]]))
                loss_rehandle = criterion[0]((p1 * label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * label_mask[:, [20, 16, 12, 8, 4]]))
                #total_loss += criterion(output * label_mask, label * label_mask)
                total_loss = total_loss + self.trade_off[0] * loss_retrieval + self.trade_off[1] * loss_rehandle + self.trade_off[2] * loss_p
        self.train()
        return total_loss/len(data_loader)

    def update_lr(self, epoch, lr):
        lr_adjust = {
            5: 1e-3, 10: 5e-4, 20: 1e-4,
            30: 5e-5, 40: 1e-5, 50: 5e-6
        }
        if epoch in lr_adjust.keys():
            lr_new = lr_adjust[epoch]
        else:
            lr_new = lr
        return lr_new

    class forecasting_data_loader(Dataset):
        def __init__(self, df_feature, df_time):

            self.df_feature = df_feature
            self.df_time = df_time

            self.df_feature = torch.tensor(
                self.df_feature, dtype=torch.float)
            self.df_time = torch.tensor(
                self.df_time, dtype=torch.float)

        def __getitem__(self, index):
            sample, timestamp = self.df_feature[index], self.df_time[index]
            return sample, timestamp

        def __len__(self):
            return int(len(self.df_time)/5)

    def model_test(self, container_file, out_file):

        nan_flag = []
        ################data convert##################
        vec_dim = 0
        for vocab in self.vocab:
            vec_dim += len(vocab)
        dt = pd.read_csv(container_file, dtype=str)
        data = dt.values[:, 4:9]
        time = dt.values[:, 3]
        feat_array = np.zeros((data.shape[0], vec_dim))
        in_time_array = np.zeros((data.shape[0], 4))
        for i in range(data.shape[0]):
            index = 0
            for j in range(data.shape[1]):
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
                    feat_array[i, index + self.vocab[j][data[i, j]]] = 1
                    index += len(self.vocab[j])
        for i in range(data.shape[0]):
            if time[i] is not np.nan:
                time1 = datetime.datetime.strptime(time[i], '%Y/%m/%d %H:%M')
                in_time_array[i, 0] = time1.minute / 59.0
                in_time_array[i, 1] = time1.hour / 23.0
                in_time_array[i, 2] = (time1.day - 1) / 30.0
                in_time_array[i, 3] = time1.month / 11.0
            else:
                in_time_array[i, 0] = 0.0
                in_time_array[i, 1] = 0.0
                in_time_array[i, 2] = 0.0
                in_time_array[i, 3] = 0.0
                nan_flag.append(i)
    ##############################################
        len_seq = 5

        feature = feat_array.reshape(-1, len_seq, feat_array.shape[-1])
        time = in_time_array.reshape(-1, len_seq, in_time_array.shape[-1])

        ############reserve squence#####################################
        for i in range(feature.shape[0]):
            mid_feature = np.zeros((feature.shape[1], feature.shape[2]))
            mid_time = np.zeros((time.shape[1], time.shape[2]))
            for j in range(len_seq):
                mid_feature[len_seq - j - 1] = feature[i, j]
                mid_time[len_seq - j - 1] = time[i, j]
            feature[i] = mid_feature
            time[i] = mid_time
        #############################################################

        pred_out = np.zeros(feature.shape[0]*feature.shape[1])
        data_row = self.forecasting_data_loader(feature, time)
        data_loader = DataLoader(data_row, batch_size=1, shuffle=False)

        for i, data in enumerate(data_loader):
            feat = data[0].to(self.device)
            timestamp = data[1].to(self.device)
            output = self.get_output(feat, timestamp)

            pred = output.reshape(-1, 5, 5)
            p_1 = (1 - pred[0, 3, 1]) * (1 - (1 - pred[0, 4, 0]))
            p_2 = (1 - pred[0, 2, 2]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1]))
            p_3 = (1 - pred[0, 1, 3]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1]) * (1 - pred[0, 2, 2]))
            p_4 = (1 - pred[0, 0, 4]) * (1 - (1 - pred[0, 4, 0]) * (1 - pred[0, 3, 1]) * (1 - pred[0, 2, 2]) * (1 - pred[0, 1, 3]))

            pred_out[i*5] = 1 if p_4 > 0.5 else 0
            pred_out[i*5+1] = 1 if p_3 > 0.5 else 0
            pred_out[i*5+2] = 1 if p_2 > 0.5 else 0
            pred_out[i*5+3] = 1 if p_1 > 0.5 else 0


        pred_out[nan_flag] = np.nan
        pred_out = pred_out.reshape(-1, 1)
        np.savetxt(out_file, pred_out, delimiter=',', fmt='%s')

    def calculate_acc(self, data_loader):
        flag_acc = 0
        flag_all = 0
        for data in data_loader:
            feat = data[0].to(self.device)
            label = data[1].to(self.device)
            label_mask = data[2].to(self.device)
            timestamp = data[3].to(self.device)
            output = self.get_output(feat, timestamp)
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
        print('acc = %.5f' % (flag_acc / flag_all))

    def calculate_rehandle_acc_2(self, data_loader):
        self.eval()
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

        totall_time = 0
        with torch.no_grad():
            for data in data_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].reshape(data[4].shape[0], -1, data[4].shape[-1]).to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)
                statr_time = time.time()
                output = self.get_output(en_in, en_ts)
                end_time = time.time()
                totall_time += end_time-statr_time
                pred = output.reshape(-1, 5, 5)
                mask = label_mask.reshape(-1, 5, 5)

                p_1, p_2 = torch.zeros([1, 5]).to(self.device), torch.zeros([1, 5]).to(self.device)
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
                        loss_mse_1 += (p_1[0, i].item() + label[0, value].item() - 1) ** 2
                        loss_mse_2 += (p_2[0, i].item() + label[0, value].item() - 1) ** 2
                        loss_retrieval_mse += (output[value].item() - label[:, value].item()) ** 2

                        loss_mae_1 += abs(p_1[0, i].item() + label[0, value].item() - 1)
                        loss_mae_2 += abs(p_2[0, i].item() + label[0, value].item() - 1)
                        loss_retrieval_mae += abs(output[value].item() - label[:, value].item())

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

                        if 1 - p_2[0, i] - label[0, value] == 0:
                            flag_acc_2 += 1

                        if 1 - p_1[0, i] - label[0, value] == 0:
                            if p_1[0, i] == 1:
                                TP += 1
                            else:
                                TN += 1
                            flag_acc_1 += 1
                        else:
                            if p_1[0, i] == 1:
                                FP += 1
                            else:
                                FN += 1
                mae_list_1.append(loss_mae_1 / flag)
                mae_list_2.append(loss_mae_2 / flag)
                mse_list_1.append(loss_mse_1 / flag)
                mse_list_2.append(loss_mse_2 / flag)
                retrieval_mae_list.append(loss_retrieval_mae / flag)
                retrieval_mse_list.append(loss_retrieval_mse / flag)

        print('acc_1 =  %.5f              acc_2 = %.5f' % (flag_acc_1 / flag_all, flag_acc_2 / flag_all))
        print('mae_1 = %.5f' % (sum(mae_list_1) / len(mae_list_1)))
        print('mae_2 = %.5f' % (sum(mae_list_2) / len(mae_list_2)))
        print('mse_1 = %.5f' % (sum(mse_list_1) / len(mse_list_1)))
        print('mse_2 = %.5f' % (sum(mse_list_2) / len(mse_list_2)))
        print('retrieval_mae = %.5f' % (sum(retrieval_mae_list) / len(retrieval_mae_list)))
        print('retrieval_mse = %.5f' % (sum(retrieval_mse_list) / len(retrieval_mse_list)))
        print('TP = %d     FP = %d' % (TP, FP))
        print('FN = %d     TN = %d' % (FN, TN))
        print('inference time = %.5f' % (len(data_loader) / totall_time))
        return flag_acc_1 / flag_all, flag_acc_2 / flag_all

    def get_transfer(self, encoding, dis_type, weight_mat=None):

        loss_all = torch.zeros(1).to(self.device)

        len_seq = encoding.shape[1]
        if weight_mat is None:
            weight = (1.0 / len_seq * torch.ones(1, len_seq)).to(self.device)
        else:
            weight = weight_mat
        dist_mat = torch.zeros(1, len_seq).to(self.device)
        data = encoding
        data_s = data[0:len(data) // 2]
        data_t = data[len(data) // 2:]
        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=data_s.shape[2])
        for j in range(data_s.shape[1]):
            loss_transfer = criterion_transder.compute(data_s[:, j, :], data_t[:, j, :])
            loss_all = loss_all + weight[0, j] * loss_transfer
            dist_mat[0, j] = loss_transfer
        return loss_all, dist_mat, weight


