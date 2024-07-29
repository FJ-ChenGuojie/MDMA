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

class DANN_RNN(nn.Module):
    def __init__(self, n_input=[77, 4], n_embedding=128, n_hiddens=[64, 64], n_output=25, dropout_rate=0.0,
                 len_seq=5, model_path=None, trade_off=[1, 0.001], lr=1e-4, device='cuda:0', num_cluster=2):
        super(DANN_RNN, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hiddens = n_hiddens
        self.dropout_rate = dropout_rate
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
        self.specific_predictor_list = nn.ModuleList()
        in_size = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=False)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.feature = nn.Sequential()
        self.feature.add_module('f_rnn1', nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln1', nn.LayerNorm(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn2', nn.GRU(input_size=64, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln2', nn.LayerNorm(64))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop2', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn3', nn.GRU(input_size=64, num_layers=1, hidden_size=64, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln3', nn.LayerNorm(64))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_drop3', nn.Dropout(dropout_rate))

        self.predictor = nn.Sequential()
        #self.predictor.add_module('p_rnn1', nn.GRU(input_size=64, num_layers=1, hidden_size=128, batch_first=True, bidirectional=False))
        self.predictor.add_module('p_fc1', nn.Linear(64*self.len_seq, 128))
        self.predictor.add_module('p_bn1', nn.BatchNorm1d(128))
        self.predictor.add_module('p_relu1', nn.ReLU(True))
        self.predictor.add_module('p_out', nn.Linear(128, 2 * self.len_seq))
        #self.predictor.add_module('p_softmax', nn.LogSoftmax(dim=1))

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential()
        #self.classifier.add_module('c_rnn1', nn.GRU(input_size=64, num_layers=1, hidden_size=128, batch_first=True, bidirectional=False))
        self.classifier.add_module('c_fc1', nn.Linear(64*self.len_seq, 128))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_fc2', nn.Linear(128, 512))
        self.classifier.add_module('c_bn2', nn.BatchNorm1d(512))
        self.classifier.add_module('c_relu2', nn.ReLU(True))
        self.classifier.add_module('c_fc3', nn.Linear(512, 128))
        self.classifier.add_module('c_bn3', nn.BatchNorm1d(128))
        self.classifier.add_module('c_relu3', nn.ReLU(True))
        self.classifier.add_module('c_fc4', nn.Linear(128, num_cluster))
        self.classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        specific_predictor = nn.Sequential()
        specific_predictor.add_module('s_fc1', nn.Linear(64*self.len_seq, 128))
        specific_predictor.add_module('s_bn1', nn.BatchNorm1d(128))
        specific_predictor.add_module('s_relu1', nn.ReLU(True))
        #specific_predictor.add_module('s_fc2', nn.Linear(128, 25))
        specific_predictor.add_module('s_fc2', nn.Linear(128, 5))
        specific_predictor.add_module('s_sigmoid', nn.Sigmoid())

        for i in range(num_cluster):
            self.specific_predictor_list.append(specific_predictor)

        self.weight = nn.Sequential()
        self.weight.add_module('w_fc1', nn.Linear(25, num_cluster))
        self.weight.add_module('w_softmax', nn.Softmax(dim=1))

    def train_model_stage1(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        #optimizer = optim.Adam(self.parameters(), self.lr)

        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()
        iter_list = []
        num_iter = np.Inf
        best_nll = 0

        indexs = get_index(len(train_list_loader))

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):
            if epoch < 100:
                optimizer = optim.Adam([{'params': self.classifier.parameters(), 'lr': self.lr},
                                        {'params': self.feature.parameters(), 'lr': self.lr*0}])
            else:
                optimizer = optim.Adam([{'params': self.classifier.parameters(), 'lr': self.lr},
                                        {'params': self.predictor.parameters(), 'lr': self.lr * 0.1},
                                        {'params': self.feature.parameters(), 'lr': self.lr}])
            torch.cuda.empty_cache()

            lr = self.update_lr(epoch, self.lr)
            optimizer.defaults['lr'] = self.lr
            self.lr = lr

            #optimizer.defaults['lr'] = self.update_lr(epoch, self.lr)

            for i in range(num_iter):
                error_class_all = torch.zeros(1).to(self.device)
                error_class_align = torch.zeros(1).to(self.device)
                error_class_list = []
                error_domain_list = []
                error_domain_all = torch.zeros(1).to(self.device)
                p = float(i + epoch * num_iter) / train_epoch / num_iter
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                for ii in range(len(iter_list)):
                    batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                    batch_label_domain = ii * torch.ones(len(batch_en_in)).long().to(self.device)
                    batch_en_in = batch_en_in.to(self.device)
                    batch_en_ts = batch_en_ts.to(self.device)
                    batch_label = batch_label.to(self.device)
                    batch_label_mask = batch_label_mask.to(self.device)
                    batch_label_class = torch.ones(len(batch_en_in), 5).long().to(self.device) - batch_label[:, [20, 16, 12, 8, 4]].long()
                    batch_label_class_mask = batch_label_mask[:, 20:25]
                    class_output, domain_output = self.get_output_stage1(batch_en_in, batch_en_ts, alpha=alpha)

                    if i % len(train_list_loader[ii]) == 0:
                        iter_list[ii] = iter(train_list_loader[ii])

                    error_class = torch.zeros(1).to(self.device)
                    for iii in range(len(batch_en_in)):
                        error_class += loss_class(class_output[iii, 0:len(torch.nonzero(batch_label_class_mask[iii])), :],
                                                  batch_label_class[iii, 0:len(torch.nonzero(batch_label_class_mask[iii]))]) / len(batch_en_in)

                    error_class_all += error_class / len(train_list_loader)
                    error_class_list.append(error_class)
                    error_domain = loss_domain(domain_output, batch_label_domain)
                    error_domain_all += error_domain / len(train_list_loader)
                    error_domain_list.append(error_domain)

                for index in indexs:
                    error_class_align += torch.abs((error_class_list[index[0]]-error_class_list[index[1]]))
                err = error_class_all + 0*error_class_align + 0*error_domain_all
                self.zero_grad()
                err.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
                optimizer.step()

                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f, err_domain: %f, error_class_align : %f   '\
                                 % (epoch, i + 1, num_iter, error_class_all.data.cpu().numpy(), error_domain_all.data.cpu().numpy(), error_class_align.data.cpu().numpy()))
                sys.stdout.flush()
            print('    ')
            for ii in range(len(iter_list)):
                print(' err_class %d : %f' % (ii + 1, error_class_list[ii].data.cpu().numpy()), end='    ')
            for ii in range(len(iter_list)):
                print('err_domain %d : %f' % (ii + 1, error_domain_list[ii].data.cpu().numpy()), end='    ')
            '''
            print('    ')
            area_mae, area_mse, nll_list, area_acc, flag_list, flag_all_list = self.test_model_stage1(valid_loader)
            acc = np.array(flag_list).sum() / np.array(flag_all_list).sum()
            print('Accuracy of the dataset: %f, NLL_Loss of the dataset : %f' % (acc, sum(nll_list)/len(nll_list)))
            if sum(nll_list)/len(nll_list) < best_nll:
                best_nll = nll_list
                print('Best epoch: %d' % (epoch))
                torch.save(self.state_dict(), 'C:/Users/Administrator/Desktop/partData/best_dann_parameter.pkl')
            '''

    def get_output_stage1(self, input_property, input_timestamp, alpha):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.feature)):
            if i % 4 == 0:
                x, _ = self.feature[i](x.float())
            else:
                x = self.feature[i](x)

        x = x.view(len(x), -1)
        reverse_x = ReverseLayerF.apply(x, alpha)
        #####predictor###########
        x = self.predictor(x)
        class_output = self.logsoftmax(x.view(-1, 2)).view(-1, self.len_seq, 2)#class_output(batch, 5, 2)

        #######dimian classifier####
        domain_output = self.classifier(reverse_x)

        return class_output, domain_output

    def test_model_stage1(self, test_loader):
        self.eval()
        area_mae = []
        area_mse = []
        area_acc = []
        mae_list = []
        mse_list = []
        nll_list = []
        flag_list = []
        flag_all_list = []
        for area_num in range(0, 60):
            mae_list.append(0)
            mse_list.append(0)
            flag_list.append(0)
            flag_all_list.append(0)

        loss_class = torch.nn.NLLLoss()

        with torch.no_grad():
            for data in test_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)
                label_class = torch.ones(len(en_in), 5).long().to(self.device) - label[:, [20, 16, 12, 8, 4]].long()
                label_class_mask = label_mask[:, 20:25]
                class_output, _ = self.get_output_stage1(en_in, en_ts, alpha=0)

                error_class = torch.zeros(1).to(self.device)
                for iii in range(len(en_in)):
                    error_class += loss_class(class_output[iii, 0:len(torch.nonzero(label_class_mask[iii]))],
                                              label_class[iii, 0:len(torch.nonzero(label_class_mask[iii]))]) / len(en_in)

                nll_list.append(error_class.data.cpu().numpy())
                class_output = self.softmax(class_output.view(-1, 2)).view(-1, self.len_seq, 2)
                p_1 = class_output[:, :, 1]

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

        self.train()

        return area_mae, area_mse, nll_list, area_acc, flag_list, flag_all_list

    def train_model_stage2(self, train_epoch=20, train_list_loader=None, valid_loader=None):
        optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = torch.nn.MSELoss()
        iter_list = []
        num_iter = np.Inf

        for i in range(len(train_list_loader)):
            iter_list.append(iter(train_list_loader[i]))
            if len(train_list_loader[i]) < num_iter:
                num_iter = len(train_list_loader[i])

        for epoch in range(train_epoch):

            torch.cuda.empty_cache()
            optimizer.defaults['lr'] = self.update_lr(epoch, self.lr)

            for i in range(num_iter):
                for ii in range(len(iter_list)):
                    #error_specific = torch.zeros(1).to(self.device)
                    batch_en_in, batch_de_in, batch_en_ts, batch_de_ts, batch_location, batch_label, batch_label_mask = iter_list[ii].next()
                    batch_en_in = batch_en_in.to(self.device)
                    batch_en_ts = batch_en_ts.to(self.device)
                    batch_label = batch_label.to(self.device)
                    batch_label_mask = batch_label_mask.to(self.device)

                    if i % len(train_list_loader[ii]) == 0:
                        iter_list[ii] = iter(train_list_loader[ii])

                    rehandle_label = torch.ones((batch_label.shape[0], 5)).to(self.device) - batch_label[:, [20, 16, 12, 8, 4]]
                    specific_output = self.get_output_stage2(batch_en_in, batch_en_ts)[ii]
                    p_rehandling, p_retrieval = self.rehandling_probability(specific_output, batch_label_mask)
                    loss_retrieval = criterion((specific_output * batch_label_mask)[:, [20, 16, 12, 8, 4]], (batch_label * batch_label_mask)[:, [20, 16, 12, 8, 4]])
                    loss_rehandling = criterion((p_rehandling * batch_label_mask[:, [20, 16, 12, 8, 4]]), (rehandle_label * batch_label_mask[:, [20, 16, 12, 8, 4]]))
                    loss_p = criterion(p_retrieval[:, 1:], p_rehandling[:, 1:])

                    err = self.trade_off[0] * loss_retrieval + self.trade_off[1] * loss_rehandling + self.trade_off[2] * loss_p
                    self.zero_grad()
                    err.backward()
                    optimizer.step()

                    sys.stdout.write('\r epoch: %d, [iter: %d / all %d]' % (epoch, i + 1, num_iter))
                    sys.stdout.write('cluster No: %d, error: %f         ' % (ii, err.data.cpu().numpy()))
                    sys.stdout.flush()
                    pass
        print('       ')

    def get_output_stage2(self, input_property, input_timestamp):
        output_list = []
        with torch.no_grad():
            embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
            embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
            x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
            for i in range(len(self.feature)):
                if i % 4 == 0:
                    x, _ = self.feature[i](x.float())
                else:
                    x = self.feature[i](x)

        x = x.mean(dim=1)
        for i in range(len(self.specific_predictor_list)):
            output_list.append(self.specific_predictor_list[i](x))
        return output_list

    def test_model_stage2(self, test_loader, clusters_list):
        self.eval()
        area_mae = []
        area_mse = []
        area_acc = []
        mae_list = []
        mse_list = []
        flag_list = []
        flag_all_list = []
        for area_num in range(len(clusters_list)):
            mae_list.append(0)
            mse_list.append(0)
            flag_list.append(0)
            flag_all_list.append(0)

        with torch.no_grad():
            for data in test_loader:
                en_in = data[0].reshape(data[0].shape[0], -1, data[0].shape[-1]).to(self.device)
                de_in = data[1].reshape(data[1].shape[0], -1, data[1].shape[-1]).to(self.device)
                en_ts = data[2].reshape(data[2].shape[0], -1, data[2].shape[-1]).to(self.device)
                de_ts = data[3].reshape(data[3].shape[0], -1, data[3].shape[-1]).to(self.device)
                location = data[4].to(self.device)
                label = data[5].to(self.device)
                label_mask = data[6].to(self.device)
                specific_output_list = self.get_output_stage2(en_in, en_ts)

                for i, value in enumerate([20, 16, 12, 8, 4]):
                    if label_mask[0, value] != 0.0:
                        for ii in range(len(clusters_list)):
                            p_rehandling, _ = self.rehandling_probability(specific_output_list[ii], label_mask)
                            if (int(location[0, 0, 0].item()) - 2) in clusters_list[ii]:
                                mse_list[ii] += (p_rehandling[0, i].item() + label[0, value].item() - 1) ** 2
                                mae_list[ii] += abs(p_rehandling[0, i].item() + label[0, value].item() - 1)

                                flag_all_list[ii] += 1
                                if p_rehandling[0, i] > 0.5:
                                    p_rehandling[0, i] = 1
                                else:
                                    p_rehandling[0, i] = 0
        
                                if 1 - p_rehandling[0, i] - label[0, value] == 0:
                                    flag_list[ii] += 1
        self.train()
        for i in range(len(clusters_list)):
            print('Accuracy of cluster No: %d is : %f' % (i, np.array(flag_list[i]) / np.array(flag_all_list[i])))
        #return area_mae, area_mse, area_acc, flag_list, flag_all_list
    
    def update_lr(self, epoch, lr):
        '''
        lr_adjust = {
            5: 5e-5, 10: 1e-5, 15: 5e-6,
            30: 1e-6, 60: 5e-7, 70: 1e-7
        }
        if epoch in lr_adjust.keys():
            lr_new = lr_adjust[epoch]
        else:
            lr_new = lr
        '''
        if epoch >= 40 and epoch < 100:
            if epoch % 30 == 0:
                return lr*0.1
            else:
                return lr
        else:
            return lr

    def rehandling_probability(self, output, label_mask):
        p_rehandling, p_retrieval = torch.zeros([output.shape[0], 5]).to(self.device), torch.zeros([output.shape[0], 5]).to(self.device)
        pred = output.reshape(-1, 5, 5)
        mask = label_mask.reshape(-1, 5, 5)

        for i in range(pred.shape[0]):
            p_rehandling[i, 1] = pred[i, 4, 0] * pred[i, 4, 1] * mask[i, 3, 1]
            p_rehandling[i, 2] = pred[i, 4, 0] * pred[i, 4, 2] + pred[i, 3, 1] * pred[i, 3, 2] * mask[i, 2, 2]
            p_rehandling[i, 3] = pred[i, 4, 0] * pred[i, 4, 3] + pred[i, 3, 1] * pred[i, 3, 3] + pred[i, 2, 2] * pred[i, 2, 3] * mask[i, 1, 3]
            p_rehandling[i, 4] = pred[i, 4, 0] * pred[i, 4, 4] + pred[i, 3, 1] * pred[i, 3, 4] + pred[i, 2, 2] * pred[i, 2, 4] + pred[i, 1, 3] * pred[i, 1, 4]*mask[i, 0, 4]
        for i in range(pred.shape[0]):
            p_retrieval[i, 1] = (1 - pred[i, 3, 1]) * (1 - (1 - pred[i, 4, 0])) * mask[i, 3, 1]
            p_retrieval[i, 2] = (1 - pred[i, 2, 2]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1])) * mask[i, 2, 2]
            p_retrieval[i, 3] = (1 - pred[i, 1, 3]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1]) * (1-pred[i, 2, 2])) * mask[i, 1, 3]
            p_retrieval[i, 4] = (1 - pred[i, 0, 4]) * (1 - (1 - pred[i, 4, 0]) * (1 - pred[i, 3, 1]) * (1-pred[i, 2, 2]) * (1-pred[i, 1, 3])) * mask[i, 0, 4]
        return p_rehandling, p_retrieval







