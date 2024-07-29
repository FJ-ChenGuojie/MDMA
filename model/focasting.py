import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import datetime
from model import basicmodel
import torch.nn.functional as F
import time

class Seq2Seq_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, en_hiddens=[128, 128, 64], device='cuda:0',
                 de_hiddens=[64, 64], n_output=25, dropout=0.1, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
            super(Seq2Seq_Model, self).__init__()
            self.n_input = n_input
            self.n_embedding = n_embedding
            self.en_hiddens = en_hiddens
            self.de_hiddens = de_hiddens
            self.dropout = dropout
            self.num_en_layers = len(en_hiddens)
            self.num_de_layers = len(de_hiddens)
            self.len_seq = len_seq
            self.n_output = n_output
            self.train_loss_list = []
            self.valid_loss_list = []
            self.p_loss_list = []
            self.acc1_list = []
            self.acc2_list = []
            self.path = model_path
            self.trade_off = trade_off
            self.device = device
            en_size = n_embedding
            de_size = n_embedding
            self.lr = lr

            self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
            self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

            en_features = nn.ModuleList()
            en_dropout = nn.ModuleList()
            for hidden in en_hiddens:
                rnn = nn.GRU(
                    input_size=en_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                en_features.append(rnn)
                en_dropout.append(nn.Dropout(self.dropout))
                en_size = hidden
            self.en_features = nn.Sequential(*en_features)
            self.en_dropout = nn.Sequential(*en_dropout)

            de_features = nn.ModuleList()
            de_dropout = nn.ModuleList()
            for hidden in de_hiddens:
                rnn = nn.GRU(
                    input_size=de_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                de_features.append(rnn)
                de_dropout.append(nn.Dropout(self.dropout))
                de_size = hidden
            self.de_features = nn.Sequential(*de_features)
            self.de_dropout = nn.Sequential(*de_dropout)

            self.fc_out = nn.Linear(self.len_seq*de_hiddens[-1], self.n_output)
            torch.nn.init.xavier_normal_(self.fc_out.weight)
            self.act = nn.GELU()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        hx = []
        for i in range(self.num_de_layers):
            hx.append(None)
        de_out = torch.zeros((embedding_out.shape[0], self.len_seq, embedding_out.shape[-1])).to(self.device)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_gru_feature, _ = self.en_features[i](en_input.float())
            en_gru_feature = self.en_dropout[i](en_gru_feature)
            en_input = en_gru_feature

        context = _.transpose(0, 1)
        for seq in range(self.len_seq):
            if seq == 0:
                de_input = context
            else:
                de_input = context + de_out[:, seq-1, :].reshape(de_out.shape[0], 1, de_out.shape[2])
            for i in range(self.num_de_layers):
                de_gru_feature, hx[i] = self.de_features[i](de_input.float(), hx[i])
                de_gru_feature = self.de_dropout[i](de_gru_feature)
                de_input = de_gru_feature
            de_out[:, seq, :] = de_input[:, 0, :].clone()

        gru_out = de_out.reshape(de_out.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()

    def train_epoch(self, train_loader, valid_loader, test_loader, optimizer, criterion, epoch):
        self.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            d_feat = data[0].to(self.device)
            d_label = data[1].to(self.device)
            d_label_mask = data[2].to(self.device)
            d_timestamp = data[3].to(self.device)

            rehandle_label = torch.ones((d_label.shape[0], 5)).to(self.device)-d_label[:, [20, 16, 12, 8, 4]]
            output = self.get_output(d_feat, d_timestamp)

            p1, p2 = self.returning_rate(output, d_label_mask)

            loss_retrieval = criterion[0]((output * d_label_mask)[:, [20, 16, 12, 8, 4]], (d_label * d_label_mask)[:, [20, 16, 12, 8, 4]])
            loss_rehandle = criterion[0]((p1 * d_label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * d_label_mask[:, [20, 16, 12, 8, 4]]))
            loss_p = criterion[0](p1[:, 1:], p2[:, 1:])

            all_loss = self.trade_off[0]*loss_retrieval + self.trade_off[1]*loss_rehandle + self.trade_off[2] * loss_p

            optimizer.zero_grad()
            all_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.parameters(), 3.)
            optimizer.step()

            if i % 20 == 0:
                loss_valid = self.pred(valid_loader)

                #acc1, acc2 = self.calculate_rehandle_acc_2(test_loader)

                self.valid_loss_list.append(loss_valid.item())
                self.train_loss_list.append(all_loss.item())
                self.p_loss_list.append(loss_p.item())
                self.rehandle_loss.append(loss_rehandle.item())
                self.retrieval_loss.append(loss_retrieval.item())
                #self.acc1_list.append(acc1)
                #self.acc2_list.append(acc2)

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
            self.train_epoch(train_loader, valid_loader, test_loader, optimizer, criterion, epoch)
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
            self.calculate_rehandle_acc_2(test_loader)
            print('runtime = %f ####epoch = %d #####train_loss = %.5f' % (run_end-run_start, epoch, loss))
            print('stop_round = %d#### min_loss = %.5f' % (stop_round, min_loss))

class MLP_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, n_hiddens=[128, 512, 256, 64], n_output=25, dropout=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
        super(MLP_Model, self).__init__()
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.len_seq = len_seq
        self.dropout = dropout
        self.n_embedding = n_embedding
        self.num_layers = len(n_hiddens)
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.path = model_path
        self.trade_off = trade_off
        self.lr = lr
        input = self.n_embedding * self.len_seq

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        features = nn.ModuleList()
        for hidden in n_hiddens:
            dnn = nn.Linear(input, hidden)
            features.append(dnn)
            features.append(nn.Dropout())
            features.append(nn.ReLU())
            input = hidden
        self.features = nn.Sequential(*features)

        self.fc_out = nn.Linear(input, self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.sigmoid = nn.Sigmoid()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property + embedding_timestamp
        for layer in self.features:
            x = layer(x)
        output = self.sigmoid(self.fc_out(x))
        return output

class RNN_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0'):
        super(RNN_Model, self).__init__()
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
                        mse_list[int(location[0, 0, 0].item()) - 2] += (p_1[0, i].item() + label[0, value].item() - 1) ** 2
                        mae_list[int(location[0, 0, 0].item()) - 2] += abs(p_1[0, i].item() + label[0, value].item() - 1)

                        flag_all_list[int(location[0, 0, 0].item()) - 2] += 1
                        if p_1[0, i] > 0.5:
                            p_1[0, i] = 1
                        else:
                            p_1[0, i] = 0

                        if 1 - p_1[0, i] - label[0, value] == 0:
                            flag_list[int(location[0, 0, 0].item()) - 2] += 1

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

class New_RNN_cell(nn.Module):
    def __init__(self, input, hidden, convd, dropout):
        super(New_RNN_cell, self).__init__()
        self.rnn_cell = nn.GRU(
                input_size=input,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True
            )
        self.linear1 = nn.Linear(hidden, input)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(input)
        self.covd1 = nn.Conv1d(in_channels=input, out_channels=convd, kernel_size=1)

        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.covd2 = nn.Conv1d(in_channels=convd, out_channels=input, kernel_size=1)

        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(input)

    def forward(self, input, hx=None):
        x = input
        xx = input
        x, hx = self.rnn_cell(x, hx)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = x + xx
        x = self.layer_norm1(x)
        xx = x
        x = self.covd1(x.transpose(1, 2))
        x = self.act1(x)
        x = self.dropout2(x)
        x = self.covd2(x)
        x = x.transpose(1, 2) + xx
        out = self.layer_norm2(x)

        return out, hx

class New_RNN_cell_2(nn.Module):
    def __init__(self, input, hidden, convd, dropout):
        super(New_RNN_cell_2, self).__init__()
        self.rnn_cell = nn.GRU(
                input_size=input,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True
            )
        self.linear1 = nn.Linear(hidden, input)
        self.dropout1 = nn.Dropout(dropout)

        self.batch_norm1 = nn.BatchNorm1d(input)
        self.covd1 = nn.Conv1d(in_channels=input, out_channels=convd, kernel_size=1)

        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.covd2 = nn.Conv1d(in_channels=convd, out_channels=input, kernel_size=1)

        self.dropout3 = nn.Dropout(dropout)
        self.batch_norm2 = nn.BatchNorm1d(input)

    def forward(self, input, hx=None):
        x = input
        xx = input
        x, hx_new = self.rnn_cell(x, hx)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = x + xx
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        xx = x
        x = self.covd1(x.transpose(1, 2))
        x = self.act1(x)
        x = self.dropout2(x)
        x = self.covd2(x)
        x = x.transpose(1, 2) + xx
        out = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)

        return out, hx_new

class New_RNN(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=512, n_hiddens=[64, 64, 64], n_covds=[2048, 2048, 2048],
                 n_output=25, dropout=0.0, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0'):
        super(New_RNN, self).__init__()

        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.path = model_path
        self.trade_off = trade_off
        self.lr = lr
        self.device = device
        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=False)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        new_rnn_list = nn.ModuleList()
        for i, hidden in enumerate(n_hiddens):
            new_rnn_list.append(New_RNN_cell(input=n_embedding, hidden=hidden, convd=n_covds[i], dropout=dropout))

        self.features = nn.Sequential(*new_rnn_list)
        self.fc_out = nn.Linear(n_embedding, n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.act = nn.Sigmoid()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            embedding_out = gru_feature

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.act(self.fc_out(gru_out))
        return output

class RNN_Model_Muti_loss(RNN_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout=0.0, len_seq=5, model_path=None, trade_off=[1, 0.001]):
        super(RNN_Model, self).__init__()
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
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.sigmoid = nn.Sigmoid()

    def train_epoch(self, train_loader, valid_loader, optimizer, criterion):
        self.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            d_feat = data[0].to(self.device)
            d_label = data[1].to(self.device)
            d_label_mask = data[2].to(self.device)
            d_timestamp = data[3].to(self.device)

            output = self.get_output(d_feat, d_timestamp)

            p1, p2 = self.returning_rate(output, d_label_mask)

            #loss = criterion(output * d_label_mask, d_label * d_label_mask)
            loss = self.get_loss(output, d_label, d_label_mask)
            loss_p = criterion(p1[:, 1:], p2[:, 1:])

            all_loss = self.trade_off[0]*loss + self.trade_off[1] * loss_p
            optimizer.zero_grad()
            all_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.parameters(), 3.)
            optimizer.step()

            if i % 20 == 0:
                total_loss = torch.zeros(1).to(self.device)
                for valid_data in valid_loader:
                    valid_feat = valid_data[0].to(self.device)
                    valid_label = valid_data[1].to(self.device)
                    valid_label_mask = valid_data[2].to(self.device)
                    valid_timestamp = valid_data[3].to(self.device)

                    output = self.get_output(valid_feat, valid_timestamp)

                    total_loss += criterion(output * valid_label_mask, valid_label * valid_label_mask)

                self.valid_loss_list.append(total_loss.item() / len(valid_loader))
                self.train_loss_list.append(loss.item())
                self.p_loss_list.append(loss_p.item())

    def get_loss(self, matrix=None, label=None, label_mask=None, criterion=nn.CrossEntropyLoss()):
        slot = matrix[:, [4, 8, 12, 16, 20]].reshape(-1).to(self.device)
        not_empty_list = []
        for i in range(slot.shape[0]):
            if label_mask[:, [4, 8, 12, 16, 20]].reshape(-1)[i] != 0:
                not_empty_list.append(i)

        mid = slot[not_empty_list].to(self.device)
        pick_up = torch.zeros(len(not_empty_list), 2).to(self.device)
        pick_up[:, 0] = torch.ones(len(not_empty_list)).to(self.device) - mid
        pick_up[:, 1] = mid
        pick_up_label = label[:, [4, 8, 12, 16, 20]].reshape(-1)[not_empty_list].to(self.device)
        loss = criterion(pick_up, pick_up_label.long())
        return loss.to(self.device)

class LSTNet(basicmodel.Basic_Model):
    def __init__(self, n_input, n_output, len_seq, n_embedding, hidRNN, hidCNN, hidSkip, CNN_kernel, skip, highway_window, dropout,
                 output_fun='sigmoid', model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0'):
        super(LSTNet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.P = len_seq
        self.len_seq = len_seq
        self.m = n_embedding
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip;
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = highway_window
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.trade_off = trade_off
        self.path = model_path
        self.lr = lr
        self.device = device
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=dropout);

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (output_fun == 'sigmoid'):
            self.output = nn.Sigmoid();
        if (output_fun == 'tanh'):
            self.output = F.tanh;

        self.outlayer = nn.Linear(self.m, self.n_output, bias=True)

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.P, self.m) + embedding_timestamp.reshape(-1, self.P, self.m)
        batch_size = x.size(0);

        # CNN
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);

        # RNN
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r, 0));

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2, 0, 3, 1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r, s), 1);

        res = self.linear1(r);

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.m);
            res = res + z;

        if (self.output):
            res = self.output(self.outlayer(res))
        return res.squeeze()

class Seq2Seq_Model_2(Seq2Seq_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, en_hiddens=[128, 128, 64],
                 de_hiddens=[64, 64], n_output=25, dropout=0.0, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
        super(Seq2Seq_Model, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.en_hiddens = en_hiddens
        self.de_hiddens = de_hiddens
        self.dropout = dropout
        self.num_en_layers = len(en_hiddens)
        self.num_de_layers = len(de_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.path = model_path
        self.trade_off = trade_off
        en_size = n_embedding
        de_size = n_embedding
        self.lr = lr

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        en_features = nn.ModuleList()
        for hidden in en_hiddens:
            rnn = nn.GRU(
                input_size=en_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            en_features.append(rnn)
            en_size = hidden
        self.en_features = nn.Sequential(*en_features)

        de_features = nn.ModuleList()
        for hidden in de_hiddens:
            rnn = nn.GRU(
                input_size=de_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            de_features.append(rnn)
            de_size = hidden
        self.de_features = nn.Sequential(*de_features)

        self.fc_out = nn.Linear(self.len_seq * de_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.act = nn.GELU()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        hx_list = []
        de_out = torch.zeros((embedding_out.shape[0], self.len_seq, embedding_out.shape[-1])).to(self.device)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_gru_feature, hx = self.en_features[i](en_input.float())
            hx_list.append(hx)
            en_input = en_gru_feature

        de_input = embedding_out

        for i in range(self.num_de_layers):
            de_gru_feature, _ = self.de_features[i](de_input.float(), hx_list[i])
            de_input = de_gru_feature

        gru_out = de_input.reshape(de_input.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()

class RNN_Conv_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, n_conv=256):
        super(RNN_Conv_Model).__init__()
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
        self.n_conv = n_conv
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_conv, n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.cov1d = nn.Conv1d(n_input[0], n_conv, kernel_size=1, stride=1)

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.sigmoid = nn.Sigmoid()

    def get_output(self, input_property, input_timestamp):
        conv_property = self.cov1d(input_property.permute(0, 2, 1).float())
        embedding_property = self.property_embedding(conv_property.permute(0, 2, 1).reshape(-1, self.n_conv))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding)+embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](embedding_out.float())
            embedding_out = gru_feature

        gru_out = gru_feature[:, -1, :].squeeze()
        output = self.sigmoid(self.fc_out(gru_out))
        return output

class GLG_RNN_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
        super(GLG_RNN_Model, self).__init__()
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
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.cond1_1 = nn.Conv1d(n_embedding, n_embedding, kernel_size=1, stride=1)
        self.cond1_2 = nn.Conv1d(n_embedding, n_embedding, kernel_size=2, stride=1)
        self.cond1_3 = nn.Conv1d(n_embedding, n_embedding, kernel_size=3, stride=1)
        self.cond1_4 = nn.Conv1d(n_embedding, n_embedding, kernel_size=4, stride=1)
        self.cond1_5 = nn.Conv1d(n_embedding, n_embedding, kernel_size=5, stride=1)

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

    def get_output(self, input_property, input_timestamp, granularity=None):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding)+embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)

        granularity_1 = self.cond1_1(embedding_out.transpose(1, 2))
        granularity_2 = self.cond1_2(embedding_out.transpose(1, 2))
        granularity_3 = self.cond1_3(embedding_out.transpose(1, 2))
        granularity_4 = self.cond1_4(embedding_out.transpose(1, 2))
        granularity_5 = self.cond1_5(embedding_out.transpose(1, 2))

        x_combine = torch.cat([granularity_1, granularity_2, granularity_3, granularity_4, granularity_5], dim=2)
        x_combine = x_combine.transpose(1, 2)
        for i in range(self.num_layers):
            gru_feature, _ = self.features[i](x_combine.float())
            x_combine = self.dropout(gru_feature)

        gru_out = gru_feature[:, -1, :]
        output = self.sigmoid(self.fc_out(gru_out))
        return output.squeeze()

class Seq2Seq_Model_3(Seq2Seq_Model):
    def __init__(self, n_input=[34, 4], n_embedding=128, en_hiddens=[128, 128, 128],
                 de_hiddens=[128, 128], n_output=25, dropout=0.0, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
        super(Seq2Seq_Model_3, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.en_hiddens = en_hiddens
        self.de_hiddens = de_hiddens
        self.dropout = dropout
        self.num_en_layers = len(en_hiddens)
        self.num_de_layers = len(de_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.path = model_path
        self.trade_off = trade_off
        en_size = n_embedding
        de_size = n_embedding
        self.lr = lr

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        en_features = nn.ModuleList()
        for hidden in en_hiddens:
            rnn = nn.GRU(
                input_size=en_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            en_features.append(rnn)
            en_size = hidden
        self.en_features = nn.Sequential(*en_features)

        self.batchnorm = nn.BatchNorm1d(n_embedding)

        de_features = nn.ModuleList()
        for hidden in de_hiddens:
            rnn = nn.GRU(
                input_size=de_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            de_features.append(rnn)
            de_size = hidden
        self.de_features = nn.Sequential(*de_features)

        self.fc_out = nn.Linear(self.len_seq * de_hiddens[-1], self.n_output)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        self.act = nn.Sigmoid()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_gru_feature, _ = self.en_features[i](en_input.float())
            en_input = en_gru_feature

        de_input = self.batchnorm(en_input.transpose(1, 2)+embedding_out.transpose(1, 2))
        de_input = de_input.transpose(1, 2)


        for i in range(self.num_de_layers):
            de_gru_feature, _ = self.de_features[i](de_input.float())
            de_input = de_gru_feature

        gru_out = de_input.reshape(de_input.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()

class Seq2Seq_Model_4(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, en_hiddens=[128, 128, 128],
                 de_hiddens=[64, 64], n_output=25, dropout=0.1, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
            super(Seq2Seq_Model_4, self).__init__()
            self.n_input = n_input
            self.n_embedding = n_embedding
            self.en_hiddens = en_hiddens
            self.de_hiddens = de_hiddens
            self.dropout = dropout
            self.num_en_layers = len(en_hiddens)
            self.num_de_layers = len(de_hiddens)
            self.len_seq = len_seq
            self.n_output = n_output
            self.train_loss_list = []
            self.valid_loss_list = []
            self.p_loss_list = []
            self.path = model_path
            self.trade_off = trade_off
            en_size = n_embedding
            de_size = n_embedding
            self.lr = lr

            self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
            self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

            en_features = nn.ModuleList()
            en_dropout = nn.ModuleList()
            for hidden in en_hiddens:
                rnn = nn.GRU(
                    input_size=en_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                en_features.append(rnn)
                en_dropout.append(nn.Dropout(self.dropout))
                en_size = hidden
            self.en_features = nn.Sequential(*en_features)
            self.en_dropout = nn.Sequential(*en_dropout)

            de_features = nn.ModuleList()
            de_dropout = nn.ModuleList()
            for hidden in de_hiddens:
                rnn = nn.GRU(
                    input_size=de_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                de_features.append(rnn)
                de_dropout.append(nn.Dropout(self.dropout))
                de_size = hidden
            self.de_features = nn.Sequential(*de_features)
            self.de_dropout = nn.Sequential(*de_dropout)

            self.fc_out = nn.Linear(self.len_seq*de_hiddens[-1], self.n_output)
            torch.nn.init.xavier_normal_(self.fc_out.weight)
            self.act = nn.GELU()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        hx = []
        for i in range(self.num_de_layers):
            hx.append(None)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_gru_feature, _ = self.en_features[i](en_input.float())
            hx[i] = _
            en_gru_feature = self.en_dropout[i](en_gru_feature)
            en_input = en_gru_feature

        de_input = en_input

        for i in range(self.num_de_layers):
            de_gru_feature, _ = self.de_features[i](de_input.float(), hx[i])
            de_gru_feature = self.de_dropout[i](de_gru_feature)
            de_input = de_gru_feature

        gru_out = de_input.reshape(de_input.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()

class New_Seq2Seq_Model(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, en_hiddens=[128, 128, 64], en_conv=[512, 512, 512], device='cuda:0',
                 de_hiddens=[64, 64], de_conv=[512, 512], n_output=25, dropout=0.1, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
            super(New_Seq2Seq_Model, self).__init__()
            self.n_input = n_input
            self.n_embedding = n_embedding
            self.en_hiddens = en_hiddens
            self.de_hiddens = de_hiddens
            self.dropout = dropout
            self.num_en_layers = len(en_hiddens)
            self.num_de_layers = len(de_hiddens)
            self.len_seq = len_seq
            self.n_output = n_output
            self.train_loss_list = []
            self.valid_loss_list = []
            self.p_loss_list = []
            self.path = model_path
            self.trade_off = trade_off
            en_size = n_embedding
            de_size = n_embedding
            self.lr = lr
            self.device = device

            self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
            self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

            en_features = nn.ModuleList()
            for i, hidden in enumerate(en_hiddens):
                en_features.append(New_RNN_cell(input=hidden, hidden=hidden, convd=en_conv[i], dropout=dropout))
            self.en_features = nn.Sequential(*en_features)

            de_features = nn.ModuleList()
            de_dropout = nn.ModuleList()
            for hidden in de_hiddens:
                rnn = nn.GRU(
                    input_size=de_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                de_features.append(rnn)
                de_dropout.append(nn.Dropout(self.dropout))
                de_size = hidden
            self.de_features = nn.Sequential(*de_features)
            self.de_dropout = nn.Sequential(*de_dropout)

            self.fc_out = nn.Linear(self.len_seq*de_hiddens[-1], self.n_output)
            torch.nn.init.xavier_normal_(self.fc_out.weight)
            self.act = nn.GELU()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        hx = []
        for i in range(self.num_de_layers):
            hx.append(None)
        de_out = torch.zeros((embedding_out.shape[0], self.len_seq, embedding_out.shape[-1])).to(self.device)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_gru_feature, _ = self.en_features[i](en_input.float())
            en_input = en_gru_feature

        context = en_input[:, -1, :].unsqueeze(dim=1)
        for seq in range(self.len_seq):
            if seq == 0:
                de_input = context
            else:
                de_input = context + de_out[:, seq - 1, :].reshape(de_out.shape[0], 1, de_out.shape[2])
            for i in range(self.num_de_layers):
                de_gru_feature, hx[i] = self.de_features[i](de_input.float(), hx[i])
                de_gru_feature = self.de_dropout[i](de_gru_feature)
                de_input = de_gru_feature
            de_out[:, seq, :] = de_input[:, 0, :].clone()

        gru_out = de_out.reshape(de_out.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()

class Seq2Seq_Model_5(basicmodel.Basic_Model):
    def __init__(self, n_input=[161, 4], n_embedding=128, en_hiddens=[128, 128, 64], device='cuda:0',
                 de_hiddens=[64, 64], n_output=25, dropout=0.1, len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4):
            super(Seq2Seq_Model_5, self).__init__()
            self.n_input = n_input
            self.n_embedding = n_embedding
            self.en_hiddens = en_hiddens
            self.de_hiddens = de_hiddens
            self.dropout = dropout
            self.num_en_layers = len(en_hiddens)
            self.num_de_layers = len(de_hiddens)
            self.len_seq = len_seq
            self.n_output = n_output
            self.train_loss_list = []
            self.valid_loss_list = []
            self.p_loss_list = []
            self.acc1_list = []
            self.acc2_list = []
            self.path = model_path
            self.trade_off = trade_off
            self.device = device
            en_size = n_embedding
            de_size = en_hiddens[-1]
            self.lr = lr

            self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
            self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

            en_conv = nn.ModuleList()
            conv_dropout = nn.ModuleList()
            en_features = nn.ModuleList()
            en_dropout = nn.ModuleList()
            for hidden in en_hiddens:
                cnn = nn.Conv2d(1, 2*en_size, kernel_size=(1, en_size))
                en_conv.append(cnn)
                conv_dropout.append(nn.Dropout(self.dropout))
                rnn = nn.GRU(
                    input_size=2*en_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                en_features.append(rnn)
                en_dropout.append(nn.Dropout(self.dropout))
                en_size = hidden
            self.en_conv = nn.Sequential(*en_conv)
            self.conv_dropout = nn.Sequential(*conv_dropout)
            self.en_features = nn.Sequential(*en_features)
            self.en_dropout = nn.Sequential(*en_dropout)

            de_features = nn.ModuleList()
            de_dropout = nn.ModuleList()
            for hidden in de_hiddens:
                rnn = nn.GRU(
                    input_size=de_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True
                )
                de_features.append(rnn)
                de_dropout.append(nn.Dropout(self.dropout))
                de_size = hidden
            self.de_features = nn.Sequential(*de_features)
            self.de_dropout = nn.Sequential(*de_dropout)

            self.fc_out = nn.Linear(self.len_seq*de_hiddens[-1], self.n_output)
            torch.nn.init.xavier_normal_(self.fc_out.weight)
            self.act = nn.GELU()

    def get_output(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        embedding_out = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        hx = []
        for i in range(self.num_de_layers):
            hx.append(None)
        de_out = torch.zeros((embedding_out.shape[0], self.len_seq, embedding_out.shape[-1])).to(self.device)
        en_input = embedding_out
        for i in range(self.num_en_layers):
            en_input = en_input.view(-1, 1, self.len_seq, en_input.shape[-1])
            en_input = F.relu(self.en_conv[i](en_input))
            en_input = self.conv_dropout[i](en_input)
            en_input = torch.squeeze(en_input, 3)
            en_input = en_input.permute(0, 2, 1)

            en_gru_feature, _ = self.en_features[i](en_input.float())
            en_gru_feature = self.en_dropout[i](en_gru_feature)
            en_input = en_gru_feature

        context = _.transpose(0, 1)
        for seq in range(self.len_seq):
            if seq == 0:
                de_input = context
            else:
                de_input = context + de_out[:, seq-1, :].reshape(de_out.shape[0], 1, de_out.shape[2])
            for i in range(self.num_de_layers):
                de_gru_feature, hx[i] = self.de_features[i](de_input.float(), hx[i])
                de_gru_feature = self.de_dropout[i](de_gru_feature)
                de_input = de_gru_feature
            de_out[:, seq, :] = de_input[:, 0, :].clone()

        gru_out = de_out.reshape(de_out.shape[0], -1)
        output = self.act(self.fc_out(gru_out))

        return output.squeeze()