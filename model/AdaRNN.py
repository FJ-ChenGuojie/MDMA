import copy

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import datetime
from model import basicmodel
import torch.nn.functional as F
from utils.tools import get_index
from base.loss_transfer import TransferLoss
import time

class AdaRNN(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout_rate=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0', pre_epoch=20,
                 dis_type='mmd'):
        super(AdaRNN, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.transfer_loss_list = []
        self.trade_off = trade_off
        self.path = model_path
        self.lr = lr
        self.device = device
        self.pre_epoch = pre_epoch
        self.dis_type = dis_type
        weight, dist, dist_old = [], [], []
        for i in range(self.num_layers):
            weight.append((1.0 / self.len_seq * torch.ones(1, self.len_seq)).to(self.device))
            dist.append(torch.zeros(1, self.len_seq).to(self.device))
            dist_old.append(torch.zeros(1, self.len_seq).to(self.device))
        self.weight_mat = weight
        self.dist_mat = dist
        self.dist_old = dist_old
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        features = nn.ModuleList()
        dropout = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                bidirectional=False
            )
            dropout.append(nn.Dropout(dropout_rate))
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)
        self.dropout = nn.Sequential(*dropout)

        self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.sigmoid = nn.Sigmoid()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding)+embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        out_list = []
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            out_list.append(gru_feature)
            embedding_out = self.dropout(gru_feature)

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.sigmoid(self.fc_out(gru_out))
        return output, out_list

    def train_model(self, train_epoch, train_list_loader, valid_loader, test_loader):
        optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = nn.MSELoss()
        min_loss = np.Inf
        stop_round = 0
        indexs = get_index(len(train_list_loader) - 1)
        for epoch in range(train_epoch):
            optimizer.defaults['lr'] = self.update_lr(epoch, self.lr)
            run_start = time.time()
            self.train_epoch(train_list_loader, valid_loader, optimizer, criterion, epoch, indexs)
            run_end = time.time()

            if epoch >= self.pre_epoch:
                for e_layer in range(self.num_layers):
                    self.weight_mat[e_layer] = self.update_weight_Boosting(self.weight_mat[e_layer], self.dist_old[e_layer], self.dist_mat[e_layer])
            for e_layer in range(len(self.dist_mat)):
                self.dist_old[e_layer] = self.dist_mat[e_layer]

            loss = self.pred(valid_loader)

            if epoch >= self.pre_epoch:
                if loss < min_loss:
                    min_loss = loss
                    stop_round = 0
                    torch.save(self, self.path)
                else:
                    stop_round += 1
                    if stop_round >= 20:
                        print('early stop')
                        break
            #self.calculate_acc(test_loader)
            print("########################################")
            self.calculate_rehandle_acc_2(test_loader)
            print('runtime = %f ####epoch = %d #####train_loss = %.5f' % (run_end-run_start, epoch, loss))
            print('stop_round = %d#### min_loss = %.5f' % (stop_round, min_loss))
            print("########################################")

    def train_epoch(self, train_list_loader, valid_loader, optimizer, criterion, epoch, indexs):
        self.train()
        torch.cuda.empty_cache()
        iter_list = []
        num_iter = 0

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) > num_iter:
                num_iter = len(train_list_loader[i])
        
        for i in range(num_iter):
            en_in_list = []
            de_in_list = []
            en_ts_list = []
            de_ts_list = []
            location_list = []
            label_list = []
            label_mask_list = []
            out_list_list = []

            for ii in range(len(train_list_loader)):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                en_in_list.append(batch_en_in.to(self.device))
                de_in_list.append(batch_de_in.to(self.device))
                en_ts_list.append(batch_en_ts.to(self.device))
                de_ts_list.append(batch_de_ts.to(self.device))
                location_list.append(batch_location.to(self.device))
                label_list.append(batch_label.to(self.device))
                label_mask_list.append(batch_label_mask.to(self.device))

                if i % len(train_list_loader[ii]) == 0:
                    iter_list[ii] = iter(train_list_loader[ii])

            loss_all = torch.zeros(1).cuda(self.device)
            loss_transfer_all = torch.zeros(1).cuda(self.device)

            for ii in range(len(train_list_loader)):
                batch_en_in = en_in_list[ii].reshape(en_in_list[ii].shape[0], -1, en_in_list[ii].shape[-1])
                batch_de_in = de_in_list[ii].reshape(de_in_list[ii].shape[0], -1, de_in_list[ii].shape[-1])
                batch_en_ts = en_ts_list[ii].reshape(en_ts_list[ii].shape[0], -1, en_ts_list[ii].shape[-1])
                batch_de_ts = de_ts_list[ii].reshape(de_ts_list[ii].shape[0], -1, de_ts_list[ii].shape[-1])
                batch_location = location_list[ii].reshape(location_list[ii].shape[0], -1, location_list[ii].shape[-1])
                batch_label = label_list[ii]
                batch_label_mask = label_mask_list[ii]

                rehandle_label = torch.ones((batch_label.shape[0], 5)).to(self.device) - batch_label[:, [20, 16, 12, 8, 4]]
                output, out_list = self.get_output(batch_en_in, batch_en_ts)
                out_list_list.append(out_list)

                p1, p2 = self.returning_rate(output, batch_label_mask)
                loss_retrieval = criterion((output * batch_label_mask)[:, [20, 16, 12, 8, 4]], (batch_label * batch_label_mask)[:, [20, 16, 12, 8, 4]])
                loss_rehandle = criterion((p1 * batch_label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * batch_label_mask[:, [20, 16, 12, 8, 4]]))
                loss_p = criterion(p1[:, 1:], p2[:, 1:])

                loss_all = loss_all + self.trade_off[0] * loss_retrieval + self.trade_off[1] * loss_rehandle + self.trade_off[2] * loss_p

            for ii in range(self.num_layers):
                for index in indexs:
                    feat = torch.cat((out_list_list[index[0]][ii], out_list_list[index[1]][ii]), 0)
                    loss_transfer, dist, self.weight_mat[ii] = self.get_transfer(feat, self.dis_type, self.weight_mat[ii])
                    loss_transfer_all += loss_transfer
                    self.dist_mat[ii] += dist

            if epoch >= self.pre_epoch:
                loss_all += self.trade_off[3] * loss_transfer_all

            optimizer.zero_grad()
            loss_all.backward()
            #torch.nn.utils.clip_grad_value_(self.parameters(), 3.)
            optimizer.step()

            if i % 100 == 0:
                loss_valid = self.pred(valid_loader)

                self.valid_loss_list.append(loss_valid.item())
                self.train_loss_list.append(loss_all.item())
                self.p_loss_list.append(loss_p.item())
                self.rehandle_loss.append(loss_rehandle.item())
                self.retrieval_loss.append(loss_retrieval.item())
                self.transfer_loss_list.append(loss_transfer_all)

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-5
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, len(weight_mat[0]))
        return weight_mat

    def datashift_evaluation_2(self, data_loader):
        self.eval()
        area_mae = []
        area_mse = []
        area_acc = []
        mae_list = []
        mse_list = []
        flag_list = []
        flag_all_list = []
        for area_num in range(0, 60):
            mae_list.append(0)
            mse_list.append(0)
            flag_list.append(0)
            flag_all_list.append(0)

        with torch.no_grad():
            for data in data_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)
                output, _ = self.get_output(en_in, en_ts)
                pred = output.reshape(-1, 5, 5)
                mask = label_mask.reshape(-1, 5, 5)

                p_1 = torch.zeros([1, 5]).to(self.device)
                p_1[0, 1] = pred[0, 4, 0] * pred[0, 4, 1] * mask[0, 3, 1]
                p_1[0, 2] = pred[0, 4, 0] * pred[0, 4, 2] + pred[0, 3, 1] * pred[0, 3, 2] * mask[0, 2, 2]
                p_1[0, 3] = pred[0, 4, 0] * pred[0, 4, 3] + pred[0, 3, 1] * pred[0, 3, 3] + pred[0, 2, 2] * pred[0, 2, 3] * mask[0, 1, 3]
                p_1[0, 4] = pred[0, 4, 0] * pred[0, 4, 4] + pred[0, 3, 1] * pred[0, 3, 4] + pred[0, 2, 2] * pred[0, 2, 4] + pred[0, 1, 3] * pred[0, 1, 4] * mask[0, 0, 4]

                #print(int(location[0, 0, 0].item()))

                for i, value in enumerate([20, 16, 12, 8, 4]):
                    if label_mask[0, value] != 0.0:
                        mse_list[int(location[0, 0, 0].item())-2] += (p_1[0, i].item() + label[0, value].item() - 1) ** 2
                        mae_list[int(location[0, 0, 0].item())-2] += abs(p_1[0, i].item() + label[0, value].item() - 1)

                        flag_all_list[int(location[0, 0, 0].item())-2] += 1
                        if p_1[0, i] > 0.5:
                            p_1[0, i] = 1
                        else:
                            p_1[0, i] = 0

                        if 1 - p_1[0, i] - label[0, value] == 0:
                            flag_list[int(location[0, 0, 0].item())-2] += 1

        for area_num in range(0, 60):
            if flag_list[area_num] > 0:
                mae = mae_list[area_num] / flag_all_list[area_num]
                mse = mse_list[area_num] / flag_all_list[area_num]
                acc = flag_list[area_num] / flag_all_list[area_num]
                area_mae.append(mae)
                area_mse.append(mse)
                area_acc.append(acc)

            else:
                area_mae.append(0)
                area_mse.append(0)
                area_acc.append(0)

        return area_mae, area_mse, area_acc, flag_list, flag_all_list

    def update_lr(self, epoch, lr):
        lr_adjust = {
            5: 5e-5, 10: 1e-4, 40: 1e-4,
            50: 5e-5, 60: 1e-5, 70: 5e-6
        }
        if epoch in lr_adjust.keys():
            lr_new = lr_adjust[epoch]
        else:
            lr_new = lr
        return lr_new

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
                output, _ = self.get_output(en_in, en_ts)
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
                output, _ = self.get_output(en_in, en_ts)
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

class AdaRNN_v2(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout_rate=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0', pre_epoch=20,
                 dis_type='mmd'):
        super(AdaRNN_v2, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.transfer_loss_list = []
        self.trade_off = trade_off
        self.path = model_path
        self.lr = lr
        self.device = device
        self.pre_epoch = pre_epoch
        self.dis_type = dis_type
        self.weight_list = []
        for i in range(self.num_layers):
            self.weight_list.append(None)
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=False)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        features = nn.ModuleList()
        dropout = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                bidirectional=False
            )
            dropout.append(nn.Dropout(dropout_rate))
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)
        self.dropout = nn.Sequential(*dropout)

        gate = nn.ModuleList()
        for i in range(len(n_hiddens)):
            gate_weight = nn.Linear(
                len_seq * self.n_hiddens[i] * 2, len_seq)
            gate.append(gate_weight)
        self.gate = gate

        bnlst = nn.ModuleList()
        for i in range(len(n_hiddens)):
            bnlst.append(nn.BatchNorm1d(len_seq))
        self.bn_lst = bnlst
        self.softmax = torch.nn.Softmax(dim=0)

        self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), self.lr)

    def train_model(self, train_epoch, train_list_loader, valid_loader, test_loader):
        #optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = nn.MSELoss()
        min_loss = np.Inf
        stop_round = 0
        indexs = get_index(len(train_list_loader) - 1)
        for epoch in range(train_epoch):
            if epoch >= self.pre_epoch:
                self.optimizer = optim.Adam([{'params': self.gate.parameters(), 'lr': self.lr*10}], lr=self.lr)
            self.optimizer.defaults['lr'] = self.update_lr(epoch, self.lr)
            run_start = time.time()
            self.train_epoch(train_list_loader, valid_loader, self.optimizer, criterion, epoch, indexs)
            run_end = time.time()

            loss = self.pred(valid_loader)

            if epoch >= self.pre_epoch:
                if loss < min_loss:
                    min_loss = loss
                    stop_round = 0
                    torch.save(self, self.path)
                else:
                    stop_round += 1
                    if stop_round >= 20:
                        print('early stop')
                        break
            #self.calculate_acc(test_loader)
            print("########################################")
            self.calculate_rehandle_acc_2(test_loader)
            print('runtime = %f ####epoch = %d #####train_loss = %.5f' % (run_end-run_start, epoch, loss))
            print('stop_round = %d#### min_loss = %.5f' % (stop_round, min_loss))
            print("########################################")

    def train_epoch(self, train_list_loader, valid_loader, optimizer, criterion, epoch, indexs):
        self.train()
        torch.cuda.empty_cache()
        iter_list = []
        num_iter = 0

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) > num_iter:
                num_iter = len(train_list_loader[i])

        for i in range(num_iter):
            en_in_list = []
            de_in_list = []
            en_ts_list = []
            de_ts_list = []
            location_list = []
            label_list = []
            label_mask_list = []
            out_list_list = []

            for ii in range(len(train_list_loader)):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                en_in_list.append(batch_en_in.to(self.device))
                de_in_list.append(batch_de_in.to(self.device))
                en_ts_list.append(batch_en_ts.to(self.device))
                de_ts_list.append(batch_de_ts.to(self.device))
                location_list.append(batch_location.to(self.device))
                label_list.append(batch_label.to(self.device))
                label_mask_list.append(batch_label_mask.to(self.device))

                if i % len(train_list_loader[ii]) == 0:
                    iter_list[ii] = iter(train_list_loader[ii])

            loss_all = torch.zeros(1).cuda(self.device)
            loss_transfer_all = torch.zeros(1).cuda(self.device)

            for ii in range(len(train_list_loader)):
                batch_en_in = en_in_list[ii].reshape(en_in_list[ii].shape[0], -1, en_in_list[ii].shape[-1])
                batch_de_in = de_in_list[ii].reshape(de_in_list[ii].shape[0], -1, de_in_list[ii].shape[-1])
                batch_en_ts = en_ts_list[ii].reshape(en_ts_list[ii].shape[0], -1, en_ts_list[ii].shape[-1])
                batch_de_ts = de_ts_list[ii].reshape(de_ts_list[ii].shape[0], -1, de_ts_list[ii].shape[-1])
                batch_location = location_list[ii].reshape(location_list[ii].shape[0], -1, location_list[ii].shape[-1])
                batch_label = label_list[ii]
                batch_label_mask = label_mask_list[ii]

                rehandle_label = torch.ones((batch_label.shape[0], 5)).to(self.device) - batch_label[:, [20, 16, 12, 8, 4]]
                output, out_list = self.get_output(batch_en_in, batch_en_ts)
                out_list_list.append(out_list)

                p1, p2 = self.returning_rate(output, batch_label_mask)
                loss_retrieval = criterion((output * batch_label_mask)[:, [20, 16, 12, 8, 4]], (batch_label * batch_label_mask)[:, [20, 16, 12, 8, 4]])
                loss_rehandle = criterion((p1 * batch_label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * batch_label_mask[:, [20, 16, 12, 8, 4]]))
                loss_p = criterion(p1[:, 1:], p2[:, 1:])

                loss_all = loss_all + self.trade_off[0] * loss_retrieval + self.trade_off[1] * loss_rehandle + self.trade_off[2] * loss_p

            for ii in range(self.num_layers):
                for index in indexs:
                    feat_all = torch.cat((out_list_list[index[0]][ii], out_list_list[index[1]][ii]), 2)
                    feat_all = feat_all.view(feat_all.shape[0], -1)
                    weight = torch.sigmoid(self.bn_lst[ii](self.gate[ii](feat_all.float())))
                    weight = torch.mean(weight, dim=0)
                    self.weight_list[ii] = self.softmax(weight)
                    loss_transfer = self.get_transfer(out_list_list[index[0]][ii], out_list_list[index[1]][ii], self.dis_type, self.weight_list[ii])
                    loss_transfer_all += loss_transfer

            if epoch >= self.pre_epoch:
                loss_all += self.trade_off[3] * loss_transfer_all

            optimizer.zero_grad()
            loss_all.backward()
            # torch.nn.utils.clip_grad_value_(self.parameters(), 3.)
            optimizer.step()

            if i % 100 == 0:
                loss_valid = self.pred(valid_loader)

                self.valid_loss_list.append(loss_valid.item())
                self.train_loss_list.append(loss_all.item())
                self.p_loss_list.append(loss_p.item())
                self.rehandle_loss.append(loss_rehandle.item())
                self.retrieval_loss.append(loss_retrieval.item())
                self.transfer_loss_list.append(loss_transfer_all)

    def get_transfer(self, feat_s, feat_t, dis_type, weight=None):
        loss_all = torch.zeros(1).to(self.device)
        if weight is None:
            weight = (1.0 / self.len_seq * torch.ones(1, self.len_seq)).to(self.device)

        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_s.shape[2])
        for j in range(feat_s.shape[1]):
            loss_transfer = criterion_transder.compute(feat_s[:, j, :], feat_t[:, j, :])
            loss_all = loss_all + weight[j] * loss_transfer
        return loss_all

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
                output, _ = self.get_output(en_in, en_ts)
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

    def datashift_evaluation_2(self, data_loader):
        self.eval()
        area_mae = []
        area_mse = []
        area_acc = []
        mae_list = []
        mse_list = []
        flag_list = []
        flag_all_list = []
        for area_num in range(0, 60):
            mae_list.append(0)
            mse_list.append(0)
            flag_list.append(0)
            flag_all_list.append(0)

        with torch.no_grad():
            for data in data_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)
                output, _ = self.get_output(en_in, en_ts)
                pred = output.reshape(-1, 5, 5)
                mask = label_mask.reshape(-1, 5, 5)

                p_1 = torch.zeros([1, 5]).to(self.device)
                p_1[0, 1] = pred[0, 4, 0] * pred[0, 4, 1] * mask[0, 3, 1]
                p_1[0, 2] = pred[0, 4, 0] * pred[0, 4, 2] + pred[0, 3, 1] * pred[0, 3, 2] * mask[0, 2, 2]
                p_1[0, 3] = pred[0, 4, 0] * pred[0, 4, 3] + pred[0, 3, 1] * pred[0, 3, 3] + pred[0, 2, 2] * pred[0, 2, 3] * mask[0, 1, 3]
                p_1[0, 4] = pred[0, 4, 0] * pred[0, 4, 4] + pred[0, 3, 1] * pred[0, 3, 4] + pred[0, 2, 2] * pred[0, 2, 4] + pred[0, 1, 3] * pred[0, 1, 4] * mask[0, 0, 4]

                print(int(location[0, 0, 0].item()))

                for i, value in enumerate([20, 16, 12, 8, 4]):
                    if label_mask[0, value] != 0.0:
                        mse_list[int(location[0, 0, 0].item())-2] += (p_1[0, i].item() + label[0, value].item() - 1) ** 2
                        mae_list[int(location[0, 0, 0].item())-2] += abs(p_1[0, i].item() + label[0, value].item() - 1)

                        flag_all_list[int(location[0, 0, 0].item())-2] += 1
                        if p_1[0, i] > 0.5:
                            p_1[0, i] = 1
                        else:
                            p_1[0, i] = 0

                        if 1 - p_1[0, i] - label[0, value] == 0:
                            flag_list[int(location[0, 0, 0].item())-2] += 1

        for area_num in range(0, 60):
            if flag_list[area_num] > 0:
                mae = mae_list[area_num] / flag_all_list[area_num]
                mse = mse_list[area_num] / flag_all_list[area_num]
                acc = flag_list[area_num] / flag_all_list[area_num]
                area_mae.append(mae)
                area_mse.append(mse)
                area_acc.append(acc)

            else:
                area_mae.append(0)
                area_mse.append(0)
                area_acc.append(0)

        return area_mae, area_mse, area_acc, flag_list, flag_all_list

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
                output, _ = self.get_output(en_in, en_ts)
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

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding)+embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        out_list = []
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            out_list.append(gru_feature)
            embedding_out = self.dropout(gru_feature)

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.sigmoid(self.fc_out(gru_out))
        return output, out_list