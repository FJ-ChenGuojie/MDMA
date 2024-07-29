import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from utils.functions import ReverseLayerF
import datetime
from model import basicmodel
import torch.nn.functional as F
import time
from utils.tools import get_index

class ADDA_RNN(nn.Module):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout_rate=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0', num_cluster=None):
        super(ADDA_RNN, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.dropout_rate = dropout_rate
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.n_output = n_output
        self.num_cluster = num_cluster
        self.train_loss_list = []
        self.valid_loss_list = []
        self.p_loss_list = []
        self.generator_loss = []
        self.discriminator_loss = []
        self.trade_off = trade_off
        self.path = model_path
        self.lr = lr
        self.device = device
        self.specific_predictor_list = nn.ModuleList()
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=False)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.s_feature = nn.Sequential()
        self.s_feature.add_module('f_rnn1', nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.s_feature.add_module('f_ln1', nn.LayerNorm(64))
        self.s_feature.add_module('f_relu1', nn.ReLU(True))
        self.s_feature.add_module('f_drop1', nn.Dropout(dropout_rate))
        self.s_feature.add_module('f_rnn2', nn.GRU(input_size=64, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.s_feature.add_module('f_ln2', nn.LayerNorm(64))
        self.s_feature.add_module('f_relu2', nn.ReLU(True))
        self.s_feature.add_module('f_drop2', nn.Dropout(dropout_rate))

        self.t_feature = nn.Sequential()
        self.t_feature.add_module('f_rnn1', nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.t_feature.add_module('f_ln1', nn.LayerNorm(64))
        self.t_feature.add_module('f_relu1', nn.ReLU(True))
        self.t_feature.add_module('f_drop1', nn.Dropout(dropout_rate))
        self.t_feature.add_module('f_rnn2', nn.GRU(input_size=64, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.t_feature.add_module('f_ln2', nn.LayerNorm(64))
        self.t_feature.add_module('f_relu2', nn.ReLU(True))
        self.t_feature.add_module('f_drop2', nn.Dropout(dropout_rate))#x=batch*5*64

        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(64*self.len_seq, 128))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_out', nn.Linear(128, 2 * self.len_seq))#x=batch*10

        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_fc1', nn.Linear(64*self.len_seq, 128))
        self.discriminator.add_module('d_bn1', nn.BatchNorm1d(128))
        self.discriminator.add_module('d_relu1', nn.ReLU(True))
        self.discriminator.add_module('d_fc2', nn.Linear(128, self.num_cluster))
        #self.discriminator.add_module('d_fc2', nn.Linear(128, 2))

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        specific_predictor = nn.Sequential()
        specific_predictor.add_module('s_fc1', nn.Linear(64*self.len_seq, 128))
        specific_predictor.add_module('s_bn1', nn.BatchNorm1d(128))
        specific_predictor.add_module('s_relu1', nn.ReLU(True))
        #specific_predictor.add_module('s_fc2', nn.Linear(128, 25))
        specific_predictor.add_module('s_fc2', nn.Linear(128, 10))
        #specific_predictor.add_module('s_sigmoid', nn.Sigmoid())

        for i in range(self.num_cluster):
            self.specific_predictor_list.append(specific_predictor)

        self.weight = nn.Sequential()
        self.weight.add_module('w_fc1', nn.Linear(25, self.num_cluster))
        self.weight.add_module('w_softmax', nn.Softmax(dim=1))

    def train_model_stage1(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer = optim.Adam([{'params': self.s_feature.parameters(), 'lr': self.lr},
                                {'params': self.classifier.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        criterion = torch.nn.NLLLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        indexs = get_index(len(train_list_loader))

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()

            for i in range(num_iter):
                error_class_all = torch.zeros(1).to(self.device)
                error_class_align = torch.zeros(1).to(self.device)
                error_class_list = []

                for ii in range(len(iter_list)):
                    batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                    batch_en_in = batch_en_in.to(self.device)
                    batch_en_ts = batch_en_ts.to(self.device)
                    batch_label = batch_label.to(self.device)
                    batch_label_mask = batch_label_mask.to(self.device)
                    batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long().to(self.device)
                    batch_label_class_mask = batch_label_mask[:, 20:25].to(self.device)
                    class_output = self.get_output_stage1(batch_en_in, batch_en_ts)

                    if i % len(train_list_loader[ii]) == 0:
                        iter_list[ii] = iter(train_list_loader[ii])

                    nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                    error_class = criterion(class_output.reshape(-1, 2)[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                    error_class_all += error_class
                    error_class_list.append(error_class)

                for index in indexs:
                    error_class_align += torch.abs((error_class_list[index[0]]-error_class_list[index[1]]))
                err = error_class_all/len(train_list_loader)
                optimizer.zero_grad()
                err.backward()
                optimizer.step()

                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f, error_class_align : %f   '\
                                 % (epoch, i + 1, num_iter, error_class_all.data.cpu().numpy(), error_class_align.data.cpu().numpy()))
                sys.stdout.flush()
            print('    ')
            for i in range(len(iter_list)):
                print(' err_class %d : %f' % (i + 1, error_class_list[i].data.cpu().numpy()), end='    ')
            print('\n')

            area_mae, area_mse, area_acc, flag_list, flag_all_list = self.test_model_stage1(valid_loader)
            if sum(flag_list) > best_acc:
                best_acc = sum(flag_list) / sum(flag_all_list)
                torch.save(self.state_dict(), 'C:/Users/Administrator/Desktop/partData/best_adda_parameter.pkl')
                torch.save(self, 'C:/Users/Administrator/Desktop/partData/best_adda_model.pkl')

            scheduler.step()

    def train_model_stage1_all(self, train_epoch=20, train_all_loader=None, valid_loader=None):
        optimizer = optim.Adam([{'params': self.s_feature.parameters(), 'lr': self.lr},
                                {'params': self.classifier.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        criterion = torch.nn.NLLLoss()
        best_acc = 0
        iter_data = iter(train_all_loader)
        num_iter = len(train_all_loader)
        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for i in range(num_iter):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_data.next()
                batch_en_in = batch_en_in.to(self.device)
                batch_en_ts = batch_en_ts.to(self.device)
                batch_label = batch_label.to(self.device)
                batch_label_mask = batch_label_mask.to(self.device)
                batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long().to(self.device)
                batch_label_class_mask = batch_label_mask[:, 20:25].to(self.device)

                if i % len(train_all_loader) == 0:
                    iter_data = iter(train_all_loader)

                class_output = self.get_output_stage1(batch_en_in, batch_en_ts)
                nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                err = criterion(class_output.reshape(-1, 2)[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f   ' % (epoch, i + 1, num_iter, err.data.cpu().numpy()))
                sys.stdout.flush()
            print('    ')

            area_mae, area_mse, area_acc, flag_list, flag_all_list = self.test_model_stage1(valid_loader)
            if sum(flag_list) > best_acc:
                best_acc = sum(flag_list) / sum(flag_all_list)
                torch.save(self.state_dict(), 'C:/Users/Administrator/Desktop/partData/best_adda_parameter_'+str(self.num_cluster)+'.pkl')
                torch.save(self, 'C:/Users/Administrator/Desktop/partData/best_adda_model_' + str(self.num_cluster) + '.pkl')

            scheduler.step()

    def get_output_stage1(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.s_feature)):
            if i % 4 == 0:
                x, _ = self.s_feature[i](x.float())
            else:
                x = self.s_feature[i](x)

        x = x.view(len(x), -1)
        x = self.classifier(x)
        class_output = self.logsoftmax(x.view(-1, 2)).view(-1, self.len_seq, 2)
        return class_output

    def test_model_stage1(self, data_loader):
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
        criterion = nn.CrossEntropyLoss()
        iter_data = iter(data_loader)
        num_iter = len(data_loader)
        err = torch.zeros(1).to(self.device)

        with torch.no_grad():
            for i in range(num_iter):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_data.next()
                batch_en_in = batch_en_in.to(self.device)
                batch_en_ts = batch_en_ts.to(self.device)
                batch_label = batch_label.to(self.device)
                batch_label_mask = batch_label_mask.to(self.device)
                batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long()
                batch_label_class_mask = batch_label_mask[:, 20:25]

                class_output = self.get_output_stage1(batch_en_in, batch_en_ts)
                nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                err += criterion(class_output.reshape(-1, 2)[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                class_output = self.softmax(class_output.view(-1, 2)).view(-1, self.len_seq, 2)
                p_1 = class_output[:, :, 1]

                for ii, value in enumerate([20, 16, 12, 8, 4]):
                    if batch_label_mask[0, value] != 0.0:
                        mse_list[int(batch_location[0, 0, 0].item()) - 2] += (p_1[0, ii].item() + batch_label[0, value].item() - 1) ** 2
                        mae_list[int(batch_location[0, 0, 0].item()) - 2] += abs(p_1[0, ii].item() + batch_label[0, value].item() - 1)
                        flag_all_list[int(batch_location[0, 0, 0].item())-2] += 1
                        if p_1[0, ii] > 0.5:
                            p_1[0, ii] = 1
                        else:
                            p_1[0, ii] = 0

                        if 1 - p_1[0, ii] - batch_label[0, value] == 0:
                            flag_list[int(batch_location[0, 0, 0].item())-2] += 1
                sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, num_iter))
                sys.stdout.flush()
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

        err /= len(data_loader)
        acc = sum(flag_list)/sum(flag_all_list)

        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err.data.cpu().numpy(), acc))
        self.train()
        return area_mae, area_mse, area_acc, flag_list, flag_all_list

    def train_model_stage2(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr*10}], betas=(0.5, 0.9))
        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr*0.1}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=500, gamma=0.1)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=200, gamma=0.1)
        criterion = torch.nn.NLLLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []
        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                for i in range(len(data_list)):
                    s_data = data_list[i]
                    t_data = []
                    for ii in range(len(s_data)):
                        t_data.append(torch.cat([data_list[iii][ii] for iii in range(len(data_list)) if iii != i], dim=0))
                        t_label_domain = torch.cat([iii * torch.ones(len(s_data[0])) for iii in range(len(data_list)) if iii != i], dim=0)

                    s_en_in = s_data[0].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_in = s_data[1].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_en_ts = s_data[2].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_ts = s_data[3].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_location = s_data[4].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    # s_label = s_data[5].repeat(len(train_list_loader)-1, 1).to(self.device)
                    # s_label_mask = s_data[6].repeat(len(train_list_loader)-1, 1).to(self.device)

                    t_en_in = t_data[0].to(self.device)
                    t_de_in = t_data[1].to(self.device)
                    t_en_ts = t_data[2].to(self.device)
                    t_de_ts = t_data[3].to(self.device)
                    t_location = t_data[4].to(self.device)
                    # t_label = t_data[5].to(self.device)
                    # t_label_mask = t_data[6].to(self.device)

                    feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                    s_label_domain = i*torch.ones(len(feat_src)).long().to(self.device)
                    t_label_domain = t_label_domain.long().to(self.device)
                    label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)

                    discriminator_output_concat = self.logsoftmax(self.discriminator(feat_concat.detach()))
                    discriminator_loss = criterion(discriminator_output_concat, label_concat) / len(train_list_loader)

                    optimizer_d.zero_grad()
                    discriminator_loss.backward()
                    optimizer_d.step()

                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    t_fake_label = i*torch.ones(len(feat_tgt)).long().to(self.device)
                    discriminator_output_tgt = self.logsoftmax(self.discriminator(feat_tgt))
                    discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label) / len(train_list_loader)

                    optimizer_t_feat.zero_grad()
                    discriminator_loss_fake.backward()
                    optimizer_t_feat.step()

                    sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f   ' \
                                     % (epoch, step + 1, num_iter, discriminator_loss.data.cpu().numpy(), discriminator_loss_fake.data.cpu().numpy()))
                    sys.stdout.flush()

            print('\n')

            if epoch % 50 == 0:
                pass
                #_ = self.test_model_stage2(valid_loader)
            self.generator_loss.append(discriminator_loss_fake.data.cpu().numpy())
            self.discriminator_loss.append(discriminator_loss.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def train_model_stage2_v2(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr*10}], betas=(0.5, 0.9))
        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr*0.005}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=100, gamma=0.8)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.1)
        criterion = torch.nn.NLLLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []
        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                for i in range(len(data_list)):
                    s_data = data_list[i]
                    for ii in [iii for iii in range(len(data_list)) if iii != i]:
                        t_data = data_list[ii]
                        s_en_in = s_data[0].to(self.device)
                        s_de_in = s_data[1].to(self.device)
                        s_en_ts = s_data[2].to(self.device)
                        s_de_ts = s_data[3].to(self.device)
                        s_location = s_data[4].to(self.device)
                        # s_label = s_data[5].to(self.device)
                        # s_label_mask = s_data[6].to(self.device)

                        t_en_in = t_data[0].to(self.device)
                        t_de_in = t_data[1].to(self.device)
                        t_en_ts = t_data[2].to(self.device)
                        t_de_ts = t_data[3].to(self.device)
                        t_location = t_data[4].to(self.device)
                        # t_label = t_data[5].to(self.device)
                        # t_label_mask = t_data[6].to(self.device)

                        feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                        feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                        feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                        s_label_domain = torch.ones(len(feat_src)).long().to(self.device)
                        t_label_domain = torch.zeros(len(feat_tgt)).long().to(self.device)
                        label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)

                        discriminator_output_concat = self.logsoftmax(self.discriminator(feat_concat.detach()))
                        discriminator_loss = criterion(discriminator_output_concat, label_concat)

                        optimizer_d.zero_grad()
                        discriminator_loss.backward()
                        optimizer_d.step()

                        feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                        t_fake_label = torch.ones(len(feat_tgt)).long().to(self.device)
                        discriminator_output_tgt = self.logsoftmax(self.discriminator(feat_tgt))
                        discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label)

                        optimizer_t_feat.zero_grad()
                        discriminator_loss_fake.backward()
                        optimizer_t_feat.step()

                        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f   ' \
                                         % (epoch, step + 1, num_iter, discriminator_loss.data.cpu().numpy(), discriminator_loss_fake.data.cpu().numpy()))
                        sys.stdout.flush()

            print('\n')
            if epoch % 50 == 0:
                pass
                #_ = self.test_model_stage2(valid_loader)
            self.generator_loss.append(discriminator_loss_fake.data.cpu().numpy())
            self.discriminator_loss.append(discriminator_loss.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def train_model_stage2_v3(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr*10}], betas=(0.5, 0.9))
        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr*0.01}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=100, gamma=0.8)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)
        criterion = torch.nn.NLLLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []
        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                for i in range(len(data_list)):
                    s_data = data_list[i]
                    t_data = []
                    for ii in range(len(s_data)):
                        t_data.append(torch.cat([data_list[iii][ii] for iii in range(len(data_list)) if iii != i], dim=0))
                    s_en_in = s_data[0].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_in = s_data[1].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_en_ts = s_data[2].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_ts = s_data[3].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_location = s_data[4].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    # s_label = s_data[5].repeat(len(train_list_loader)-1, 1).to(self.device)
                    # s_label_mask = s_data[6].repeat(len(train_list_loader)-1, 1).to(self.device)

                    t_en_in = t_data[0].to(self.device)
                    t_de_in = t_data[1].to(self.device)
                    t_en_ts = t_data[2].to(self.device)
                    t_de_ts = t_data[3].to(self.device)
                    t_location = t_data[4].to(self.device)
                    # t_label = t_data[5].to(self.device)
                    # t_label_mask = t_data[6].to(self.device)

                    feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                    s_label_domain = torch.ones(len(feat_src)).long().to(self.device)
                    t_label_domain = torch.zeros(len(feat_tgt)).long().to(self.device)
                    label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)

                    discriminator_output_concat = self.logsoftmax(self.discriminator(feat_concat.detach()))
                    discriminator_loss = criterion(discriminator_output_concat, label_concat)

                    optimizer_d.zero_grad()
                    discriminator_loss.backward()
                    optimizer_d.step()

                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    t_fake_label = torch.ones(len(feat_tgt)).long().to(self.device)
                    discriminator_output_tgt = self.logsoftmax(self.discriminator(feat_tgt))
                    discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label)

                    optimizer_t_feat.zero_grad()
                    discriminator_loss_fake.backward()
                    optimizer_t_feat.step()

                    sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f   ' \
                                     % (epoch, step + 1, num_iter, discriminator_loss.data.cpu().numpy(), discriminator_loss_fake.data.cpu().numpy()))
                    sys.stdout.flush()

            print('\n')
            if epoch % 50 == 0:
                pass
                #_ = self.test_model_stage2(valid_loader)
            self.generator_loss.append(discriminator_loss_fake.data.cpu().numpy())
            self.discriminator_loss.append(discriminator_loss.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def train_model_stage2_v4(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))
        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=100, gamma=0.8)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.8)
        criterion = torch.nn.CrossEntropyLoss()
        num_iter = np.Inf
        iter_list = []

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                data = []
                t_data = []
                for i in range(len(data_list[0])):
                    data.append(torch.cat([data_list[ii][i] for ii in range(len(data_list))], dim=0))

                s_en_in = data[0].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                s_de_in = data[1].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                s_en_ts = data[2].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                s_de_ts = data[3].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                s_location = data[4].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                #s_label = data[5].repeat(len(train_list_loader)-1, 1).to(self.device)
                #s_label_mask = data[6].repeat(len(train_list_loader)-1, 1).to(self.device)

                t_en_in = data[0].to(self.device)
                t_de_in = data[1].to(self.device)
                t_en_ts = data[2].to(self.device)
                t_de_ts = data[3].to(self.device)
                t_location = data[4].to(self.device)
                t_label = data[5].to(self.device)
                t_label_mask = data[6].to(self.device)

                feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                s_label_domain = torch.ones(len(feat_src)).long().to(self.device)
                t_label_domain = torch.zeros(len(feat_tgt)).long().to(self.device)
                label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)

                discriminator_output_concat = self.softmax(self.discriminator(feat_concat.detach()))
                discriminator_loss = criterion(discriminator_output_concat, label_concat)

                optimizer_d.zero_grad()
                discriminator_loss.backward()
                optimizer_d.step()

                feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                t_fake_label = torch.ones(len(feat_tgt)).long().to(self.device)
                discriminator_output_tgt = self.softmax(self.discriminator(feat_tgt))
                discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label)

                class_output_tgt = self.softmax(self.classifier(feat_tgt).view(-1, 2))  # batch*10
                label_class = torch.ones(len(t_label), 5).long().to(self.device) - t_label[:, [20, 16, 12, 8, 4]].long().to(self.device)
                label_class_mask = t_label_mask[:, 20:25].to(self.device)
                nonzero_position = torch.where(label_class_mask.reshape(-1))[0].tolist()
                class_output_loss = criterion(class_output_tgt.reshape(-1, 2)[nonzero_position], label_class.reshape(-1, 1).squeeze()[nonzero_position])

                err = 0 * discriminator_loss_fake + 0 * class_output_loss

                optimizer_t_feat.zero_grad()
                err.backward()
                optimizer_t_feat.step()

                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f, classifier : %f   ' \
                                 % (epoch, step + 1, num_iter, discriminator_loss.data.cpu().numpy(), discriminator_loss_fake.data.cpu().numpy(), \
                                    class_output_loss))
                sys.stdout.flush()

            print('\n')
            if epoch % 50 == 0:
                pass
                _ = self.test_model_stage2(valid_loader)
            #self.generator_loss.append(discriminator_loss_fake.data.cpu().numpy())
            self.generator_loss.append(err.data.cpu().numpy())
            self.discriminator_loss.append(discriminator_loss.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def train_model_stage2_v5(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))

        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr},
                                  {'params': self.t_feature.parameters(), 'lr': 0},
                                  {'params': self.s_feature.parameters(), 'lr': 0},
                                  {'params': self.property_embedding.parameters(), 'lr': 0},
                                  {'params': self.time_embedding.parameters(), 'lr': 0}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=100, gamma=0.8)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []
        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                for i in range(len(data_list)):
                    s_data = data_list[i]
                    for ii in [iii for iii in range(len(data_list)) if iii != i]:
                        t_data = data_list[ii]
                        s_en_in = s_data[0].to(self.device)
                        s_de_in = s_data[1].to(self.device)
                        s_en_ts = s_data[2].to(self.device)
                        s_de_ts = s_data[3].to(self.device)
                        s_location = s_data[4].to(self.device)
                        # s_label = s_data[5].to(self.device)
                        # s_label_mask = s_data[6].to(self.device)

                        t_en_in = t_data[0].to(self.device)
                        t_de_in = t_data[1].to(self.device)
                        t_en_ts = t_data[2].to(self.device)
                        t_de_ts = t_data[3].to(self.device)
                        t_location = t_data[4].to(self.device)
                        t_label = t_data[5].to(self.device)
                        t_label_mask = t_data[6].to(self.device)

                        feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                        feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                        feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                        s_label_domain = i*torch.ones(len(feat_src)).long().to(self.device)
                        t_label_domain = ii*torch.ones(len(feat_tgt)).long().to(self.device)
                        label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)
                        pass
                        #discriminator_output_concat = self.logsoftmax(self.discriminator(feat_concat.detach()))
                        discriminator_output_concat = self.softmax(self.discriminator(feat_concat.detach()))
                        discriminator_loss = criterion(discriminator_output_concat, label_concat)

                        optimizer_d.zero_grad()
                        discriminator_loss.backward()
                        optimizer_d.step()

                        feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                        t_fake_label = i*torch.ones(len(feat_tgt)).long().to(self.device)
                        discriminator_output_tgt = self.softmax(self.discriminator(feat_tgt))
                        discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label)

                        class_output_tgt = self.softmax(self.classifier(feat_tgt).view(-1, 2))  # batch*10
                        label_class = torch.ones(len(t_label), 5).long().to(self.device) - t_label[:, [20, 16, 12, 8, 4]].long().to(self.device)
                        label_class_mask = t_label_mask[:, 20:25].to(self.device)
                        nonzero_position = torch.where(label_class_mask.reshape(-1))[0].tolist()
                        class_output_loss = criterion(class_output_tgt.reshape(-1, 2)[nonzero_position], label_class.reshape(-1, 1).squeeze()[nonzero_position])

                        err = discriminator_loss_fake + 0*class_output_loss

                        optimizer_t_feat.zero_grad()
                        err.backward()
                        optimizer_t_feat.step()

                        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f, classifier : %f   ' \
                                         % (epoch, step + 1, num_iter, discriminator_loss.data.cpu().numpy(), discriminator_loss_fake.data.cpu().numpy(), \
                                            class_output_loss))
                        sys.stdout.flush()

            print('\n')
            if epoch % 50 == 0:
                pass
                _ = self.test_model_stage2(valid_loader)
            self.generator_loss.append(discriminator_loss_fake.data.cpu().numpy())
            #self.generator_loss.append(err.data.cpu().numpy())
            self.discriminator_loss.append(discriminator_loss.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def train_model_stage2_v6(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer_t_feat = optim.Adam([{'params': self.t_feature.parameters(), 'lr': self.lr*100}], betas=(0.5, 0.9))

        optimizer_d = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.lr},
                                  {'params': self.t_feature.parameters(), 'lr': 0},
                                  {'params': self.s_feature.parameters(), 'lr': 0},
                                  {'params': self.property_embedding.parameters(), 'lr': 0},
                                  {'params': self.time_embedding.parameters(), 'lr': 0}], betas=(0.5, 0.9))

        scheduler_t_feat = torch.optim.lr_scheduler.StepLR(optimizer_t_feat, step_size=100, gamma=0.8)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []
        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()
            for step, data_list in enumerate(zip(*train_list_loader)):
                err_discriminator = torch.zeros(1).to(self.device)
                err_f_feature = torch.zeros(1).to(self.device)
                for i in range(len(data_list)):
                    s_data = data_list[i]
                    t_data = []
                    for ii in range(len(s_data)):
                        t_data.append(torch.cat([data_list[iii][ii] for iii in range(len(data_list)) if iii != i], dim=0))
                        t_label_domain = torch.cat([iii*torch.ones(len(s_data[0])) for iii in range(len(data_list)) if iii != i], dim=0)

                    s_en_in = s_data[0].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_in = s_data[1].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_en_ts = s_data[2].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_de_ts = s_data[3].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    s_location = s_data[4].repeat(len(train_list_loader) - 1, 1, 1).to(self.device)
                    #s_label = s_data[5].repeat(len(train_list_loader)-1, 1).to(self.device)
                    #s_label_mask = s_data[6].repeat(len(train_list_loader)-1, 1).to(self.device)

                    t_en_in = t_data[0].to(self.device)
                    t_de_in = t_data[1].to(self.device)
                    t_en_ts = t_data[2].to(self.device)
                    t_de_ts = t_data[3].to(self.device)
                    t_location = t_data[4].to(self.device)
                    #t_label = t_data[5].to(self.device)
                    #t_label_mask = t_data[6].to(self.device)

                    feat_src = self.get_output_s_feature(s_en_in, s_en_ts)
                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                    s_label_domain = i*torch.ones(len(feat_src)).long().to(self.device)
                    t_label_domain = t_label_domain.long().to(self.device)
                    label_concat = torch.cat((s_label_domain, t_label_domain), dim=0)

                    discriminator_output_concat = self.softmax(self.discriminator(feat_concat.detach()))
                    discriminator_loss = criterion(discriminator_output_concat, label_concat) / len(train_list_loader)
                    err_discriminator += discriminator_loss
                optimizer_d.zero_grad()
                err_discriminator.backward()
                optimizer_d.step()

                for i in range(len(data_list)):
                    s_data = data_list[i]
                    t_data = []
                    for ii in range(len(s_data)):
                        t_data.append(torch.cat([data_list[iii][ii] for iii in range(len(data_list)) if iii != i], dim=0))
                        t_label_domain = torch.cat([iii * torch.ones(len(s_data[0])) for iii in range(len(data_list)) if iii != i], dim=0)

                    t_en_in = t_data[0].to(self.device)
                    t_de_in = t_data[1].to(self.device)
                    t_en_ts = t_data[2].to(self.device)
                    t_de_ts = t_data[3].to(self.device)
                    t_location = t_data[4].to(self.device)
                    # t_label = t_data[5].to(self.device)
                    # t_label_mask = t_data[6].to(self.device)

                    feat_tgt = self.get_output_t_feature(t_en_in, t_en_ts)
                    t_fake_label = i*torch.ones(len(feat_tgt)).long().to(self.device)
                    discriminator_output_tgt = self.logsoftmax(self.discriminator(feat_tgt))
                    discriminator_loss_fake = criterion(discriminator_output_tgt, t_fake_label) / len(train_list_loader)

                    err_f_feature += discriminator_loss_fake
                optimizer_t_feat.zero_grad()
                discriminator_loss_fake.backward()
                optimizer_t_feat.step()

                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], discriminator: %f, t_feature : %f   ' \
                                 % (epoch, step + 1, num_iter, err_discriminator.data.cpu().numpy(), err_f_feature.data.cpu().numpy()))
                sys.stdout.flush()
            print('\n')
            if epoch % 50 == 0:
                pass
                #_ = self.test_model_stage2(valid_loader)
            self.generator_loss.append(err_f_feature.data.cpu().numpy())
            self.discriminator_loss.append(err_discriminator.data.cpu().numpy())
            scheduler_t_feat.step()
            scheduler_d.step()

    def get_output_s_feature(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.s_feature)):
            if i % 4 == 0:
                x, _ = self.s_feature[i](x.float())
            else:
                x = self.s_feature[i](x)

        x = x.view(len(x), -1)

        return x
    
    def get_output_t_feature(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.t_feature)):
            if i % 4 == 0:
                x, _ = self.t_feature[i](x.float())
            else:
                x = self.t_feature[i](x)

        x = x.view(len(x), -1)
        return x

    def test_model_stage2(self, data_loader):
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
        criterion = nn.CrossEntropyLoss()
        iter_data = iter(data_loader)
        num_iter = len(data_loader)
        err = torch.zeros(1).to(self.device)

        with torch.no_grad():
            for i in range(num_iter):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_data.next()
                batch_en_in = batch_en_in.to(self.device)
                batch_en_ts = batch_en_ts.to(self.device)
                batch_label = batch_label.to(self.device)
                batch_label_mask = batch_label_mask.to(self.device)
                batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long()
                batch_label_class_mask = batch_label_mask[:, 20:25]

                a = self.get_output_t_feature(batch_en_in, batch_en_ts)
                feat = self.classifier(a)
                class_output = self.logsoftmax(feat.view(-1, 2)).view(-1, self.len_seq, 2)

                nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                err += criterion(class_output.reshape(-1, 2)[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                class_output = self.softmax(class_output.view(-1, 2)).view(-1, self.len_seq, 2)
                p_1 = class_output[:, :, 1]

                for ii, value in enumerate([20, 16, 12, 8, 4]):
                    if batch_label_mask[0, value] != 0.0:
                        mse_list[int(batch_location[0, 0, 0].item()) - 2] += (p_1[0, ii].item() + batch_label[0, value].item() - 1) ** 2
                        mae_list[int(batch_location[0, 0, 0].item()) - 2] += abs(p_1[0, ii].item() + batch_label[0, value].item() - 1)
                        flag_all_list[int(batch_location[0, 0, 0].item()) - 2] += 1
                        if p_1[0, ii] > 0.5:
                            p_1[0, ii] = 1
                        else:
                            p_1[0, ii] = 0

                        if 1 - p_1[0, ii] - batch_label[0, value] == 0:
                            flag_list[int(batch_location[0, 0, 0].item()) - 2] += 1
                sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, num_iter))
                sys.stdout.flush()
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

        err /= len(data_loader)
        acc = sum(flag_list) / sum(flag_all_list)

        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err.data.cpu().numpy(), acc))
        self.train()
        return area_mae, area_mse, area_acc, flag_list, flag_all_list

    def train_model_stage3(self, train_epoch=20, train_list_loader=None, valid_loader=None, cluster_lists=None):
        optimizer = optim.Adam([{'params': self.specific_predictor_list.parameters(), 'lr': self.lr}], betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        criterion = torch.nn.CrossEntropyLoss()
        best_acc = 0
        num_iter = np.Inf
        iter_list = []

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            torch.cuda.empty_cache()

            for i in range(num_iter):
                error_class_all = torch.zeros(1).to(self.device)
                error_class_list = []

                for ii in range(len(iter_list)):
                    batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                    batch_en_in = batch_en_in.to(self.device)
                    batch_en_ts = batch_en_ts.to(self.device)
                    batch_label = batch_label.to(self.device)
                    batch_label_mask = batch_label_mask.to(self.device)
                    batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long().to(self.device)
                    batch_label_class_mask = batch_label_mask[:, 20:25].to(self.device)

                    feature = self.get_output_t_feature(batch_en_in, batch_en_ts)
                    class_output = self.specific_predictor_list[ii](feature.detach())
                    class_output = self.softmax(class_output.reshape(-1, 2))

                    if i % len(train_list_loader[ii]) == 0:
                        iter_list[ii] = iter(train_list_loader[ii])

                    nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                    error_class = criterion(class_output[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                    error_class_all += error_class
                    error_class_list.append(error_class)

                    optimizer.zero_grad()
                    error_class.backward()
                    optimizer.step()

                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f   '\
                                 % (epoch, i + 1, num_iter, error_class_all.data.cpu().numpy()))
                sys.stdout.flush()
            print('    ')
            for i in range(len(iter_list)):
                print(' err_class %d : %f' % (i + 1, error_class_list[i].data.cpu().numpy()), end='    ')
            print('\n')

            area_mae, area_mse, area_acc, flag_list, flag_all_list = self.test_model_stage3(valid_loader, cluster_lists)
            if sum(flag_list) > best_acc:
                best_acc = sum(flag_list) / sum(flag_all_list)
                torch.save(self.state_dict(), 'C:/Users/Administrator/Desktop/partData/best_adda_stage3_parameter_' + str(self.num_cluster) + '.pkl')
                torch.save(self, 'C:/Users/Administrator/Desktop/partData/best_adda_model_stage3_' + str(self.num_cluster) + '.pkl')

            scheduler.step()

    def test_model_stage3(self, data_loader, cluster_lists):
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
        criterion = nn.CrossEntropyLoss()
        iter_data = iter(data_loader)
        num_iter = len(data_loader)
        err = torch.zeros(1).to(self.device)

        with torch.no_grad():
            for i in range(num_iter):
                batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_data.next()
                batch_en_in = batch_en_in.to(self.device)
                batch_en_ts = batch_en_ts.to(self.device)
                batch_label = batch_label.to(self.device)
                batch_label_mask = batch_label_mask.to(self.device)
                batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long()
                batch_label_class_mask = batch_label_mask[:, 20:25]

                for ii in range(self.num_cluster):
                    if int(batch_location[0, 0, 0]) in cluster_lists[ii]:
                        feature = self.get_output_t_feature(batch_en_in, batch_en_ts)
                        class_output = self.specific_predictor_list[ii](feature.detach())

                nonzero_position = torch.where(batch_label_class_mask.reshape(-1))[0].tolist()
                err += criterion(class_output.reshape(-1, 2)[nonzero_position], batch_label_class.reshape(-1, 1).squeeze()[nonzero_position])
                class_output = self.softmax(class_output.view(-1, 2)).view(-1, self.len_seq, 2)
                p_1 = class_output[:, :, 1]

                for ii, value in enumerate([20, 16, 12, 8, 4]):
                    if batch_label_mask[0, value] != 0.0:
                        mse_list[int(batch_location[0, 0, 0].item()) - 2] += (p_1[0, ii].item() + batch_label[0, value].item() - 1) ** 2
                        mae_list[int(batch_location[0, 0, 0].item()) - 2] += abs(p_1[0, ii].item() + batch_label[0, value].item() - 1)
                        flag_all_list[int(batch_location[0, 0, 0].item()) - 2] += 1
                        if p_1[0, ii] > 0.5:
                            p_1[0, ii] = 1
                        else:
                            p_1[0, ii] = 0

                        if 1 - p_1[0, ii] - batch_label[0, value] == 0:
                            flag_list[int(batch_location[0, 0, 0].item()) - 2] += 1
                sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, num_iter))
                sys.stdout.flush()
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

        err /= len(data_loader)
        acc = sum(flag_list) / sum(flag_all_list)

        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err.data.cpu().numpy(), acc))
        self.train()
        return area_mae, area_mse, area_acc, flag_list, flag_all_list
