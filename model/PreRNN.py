import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import datetime
from model import basicmodel
import torch.nn.functional as F
import time

class Pre_RNN(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0'):
        super(Pre_RNN, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.trade_off = trade_off
        self.path = model_path
        self.lr = lr
        self.device = device
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
            dropout.append(nn.Dropout(self.dropout))
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
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            embedding_out = self.dropout(gru_feature)

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.sigmoid(self.fc_out(gru_out))
        return output

    def train_model(self, train_epoch, train_loader, valid_loader, test_loader):
        optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = [nn.MSELoss(), nn.L1Loss()]
        lr = self.lr
        min_loss = np.Inf
        stop_round = 0
        for epoch in range(train_epoch):
            lr = self.update_lr(epoch, lr)
            optimizer.defaults['lr'] = lr * 10-2
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
                output = self.get_output(en_in, en_ts)
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
                        mse_list[int(location[0, 0, 0].item())-1] += (p_1[0, i].item() + label[0, value].item() - 1) ** 2
                        mae_list[int(location[0, 0, 0].item())-1] += abs(p_1[0, i].item() + label[0, value].item() - 1)

                        flag_all_list[int(location[0, 0, 0].item())-1] += 1
                        if p_1[0, i] > 0.5:
                            p_1[0, i] = 1
                        else:
                            p_1[0, i] = 0

                        if 1 - p_1[0, i] - label[0, value] == 0:
                            flag_list[int(location[0, 0, 0].item())-1] += 1

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

