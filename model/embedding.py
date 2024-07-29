import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Embedding_Model(nn.Module):
    def __init__(self, n_input=62, n_embedding=128, n_hidden=256, vocab_list=[], lr=1e-3, path=None):
        super(Embedding_Model, self).__init__()
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.valid_loss_list = []
        self.train_loss_list = []
        self.vocab_list = vocab_list
        self.lr = lr
        self.path = path

        self.embedding = nn.Linear(n_input, n_embedding, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(n_embedding)
        self.dropout1 = nn.Dropout(p=0.1)

        self.linear1 = nn.Linear(n_embedding, n_input)
        self.batchnorm2 = nn.BatchNorm1d(n_input)
        self.dropout2 = nn.Dropout(p=0.1)
        '''
        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(n_hidden, n_input)
        self.batchnorm3 = nn.BatchNorm1d(n_embedding)
        self.dropout3 = nn.Dropout(p=0.1)
        '''
        self.activation_function2 = [nn.Softmax(dim=1) for i in range(len(self.vocab_list))]


    def get_out(self, inputs):
        out_list = []
        index = 0
        x = inputs.reshape(-1, self.n_input)
        x = self.embedding(x)
        x = torch.sum(x.reshape(-1, inputs.shape[1], self.n_embedding), axis=1).squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        for i in range(len(self.vocab_list)):
            x_slice = x[:, index:index+len(self.vocab_list[i])]
            index += len(self.vocab_list[i])
            out_list.append(self.activation_function2[i](x_slice))
        return x

    def train_epoch(self, train_loader, valid_loader, optimizer, criterion, epoch):
        self.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            d_feat = data[0].to('cuda:0')
            d_label = data[1].to('cuda:0')

            output = self.get_out(d_feat)

            loss = criterion(output, d_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_valid = self.pred(valid_loader)

                self.valid_loss_list.append(loss_valid.item())
                self.train_loss_list.append(loss.item())

    def train_model(self, train_epoch, train_loader, valid_loader):
        optimizer = optim.Adam(self.parameters(), self.lr)
        criterion = nn.MSELoss()
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
            print('runtime = %f ####epoch = %d #####train_loss = %.5f' % (run_end - run_start, epoch, loss))
            print('stop_round = %d#### min_loss = %.5f' % (stop_round, min_loss))

    def pred(self, data_loader):
        total_loss = torch.zeros(1).to('cuda:0')
        criterion = nn.MSELoss()
        for data in data_loader:
            feat = data[0].to('cuda:0')
            label = data[1].to('cuda:0')

            output = self.get_out(feat)
            loss = criterion(output, label)
            total_loss += loss

        return total_loss / len(data_loader)

    def update_lr(self, epoch, lr):
        lr_adjust = {
            5: 1e-3, 15: 5e-4, 30: 1e-4,
            35: 5e-5, 40: 1e-5, 50: 5e-6
        }
        if epoch in lr_adjust.keys():
            lr_new = lr_adjust[epoch]
        else:
            lr_new = lr
        return lr_new