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

class RNNEncoder(nn.Module):
    def __init__(self, n_input=[77, 4], n_embedding=128, dropout_rate=0.0, len_seq=5, last_dim=64, device='cuda:0'):
        super(RNNEncoder, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding

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
        self.feature.add_module('f_rnn3', nn.GRU(input_size=64, num_layers=1, hidden_size=last_dim, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln3', nn.LayerNorm(last_dim))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_drop3', nn.Dropout(dropout_rate))

    def forward(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.feature)):
            if i % 4 == 0:
                x, _ = self.feature[i](x.float())
            else:
                x = self.feature[i](x)
        return x.view(len(input_property), -1)
    
class RNNEncoder_huge(nn.Module):
    def __init__(self, n_input=[77, 4], n_embedding=128, dropout_rate=0.0, len_seq=5, last_dim=64, device='cuda:0'):
        super(RNNEncoder_huge, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding

        self.property_embedding = nn.Linear(n_input[0], n_embedding, bias=False)
        self.time_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.feature = nn.Sequential()
        self.feature.add_module('f_rnn1', nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=256, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln1', nn.LayerNorm(256))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn2', nn.GRU(input_size=256, num_layers=1, hidden_size=256, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln2', nn.LayerNorm(256))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop2', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn3', nn.GRU(input_size=256, num_layers=1, hidden_size=256, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln3', nn.LayerNorm(256))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_drop3', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn4', nn.GRU(input_size=256, num_layers=1, hidden_size=256, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln4', nn.LayerNorm(256))
        self.feature.add_module('f_relu4', nn.ReLU(True))
        self.feature.add_module('f_drop4', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn5', nn.GRU(input_size=256, num_layers=1, hidden_size=256, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln5', nn.LayerNorm(256))
        self.feature.add_module('f_relu5', nn.ReLU(True))
        self.feature.add_module('f_drop5', nn.Dropout(dropout_rate))
        self.feature.add_module('f_rnn6', nn.GRU(input_size=256, num_layers=1, hidden_size=last_dim, batch_first=True, bidirectional=False))
        self.feature.add_module('f_ln6', nn.LayerNorm(last_dim))
        self.feature.add_module('f_relu6', nn.ReLU(True))
        self.feature.add_module('f_drop6', nn.Dropout(dropout_rate))

    def forward(self, input_property, input_timestamp):
        embedding_property = self.property_embedding(input_property.reshape(-1, self.n_input[0]))
        embedding_timestamp = self.time_embedding(input_timestamp.reshape(-1, self.n_input[1]))
        x = embedding_property.reshape(-1, self.len_seq, self.n_embedding) + embedding_timestamp.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.feature)):
            if i % 4 == 0:
                x, _ = self.feature[i](x.float())
            else:
                x = self.feature[i](x)
        return x.view(len(input_property), -1)

class RNNClassifier(nn.Module):
    def __init__(self, n_input=64, len_seq=5, device='cuda:0'):
        super(RNNClassifier, self).__init__()
        self.n_input = n_input
        self.len_seq = len_seq
        self.device = device

        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(n_input * self.len_seq, 128))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_out', nn.Linear(128, 2 * self.len_seq))# x=batch*10

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.classifier(x)
        x = self.softmax(x.view(-1, 2))
        return x

class Discriminator(nn.Module):
    def __init__(self, n_input=64, len_seq=5, device='cuda:0'):
        super(Discriminator, self).__init__()
        self.n_input = n_input
        self.len_seq = len_seq
        self.device = device

        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_fc1', nn.Linear(n_input * self.len_seq, 128))
        self.discriminator.add_module('d_bn1', nn.BatchNorm1d(128))
        self.discriminator.add_module('d_relu1', nn.ReLU(True))
        self.discriminator.add_module('d_fc2', nn.Linear(128, 1))

        #self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.discriminator(x)
        #return self.softmax(x)
        return self.lrelu(x)

class Weight(nn.Module):
    def __init__(self, n_input=64, len_seq=5, device='cuda:0'):
        super(Weight, self).__init__()
        self.n_input = n_input
        self.len_seq = len_seq
        self.device = device

        self.weightor = nn.Sequential()
        #self.weightor.add_module('w_fc1', nn.Linear(n_input*self.len_seq*2, n_input*self.len_seq))
        #self.weightor.add_module('w_bn1', nn.BatchNorm1d(n_input*self.len_seq))
        #self.weightor.add_module('w_relu1', nn.ReLU(True))
        self.weightor.add_module('w_fc2', nn.Linear(n_input * self.len_seq, 128))
        self.weightor.add_module('w_bn2', nn.BatchNorm1d(128))
        self.weightor.add_module('w_relu2', nn.ReLU(True))
        self.weightor.add_module('w_out', nn.Linear(128, 2 * self.len_seq))# x=batch*10

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.weightor(x)
        x = self.softmax(x.view(-1, 2))
        return x

class Weight_2(nn.Module):
    def __init__(self, n_input=64, len_seq=5, device='cuda:0'):
        super(Weight_2, self).__init__()
        self.n_input = n_input
        self.len_seq = len_seq
        self.device = device

        self.weightor = nn.Sequential()
        self.weightor.add_module('w_fc1', nn.Linear(4, 128))
        self.weightor.add_module('w_fc2', nn.Linear(128, 2))# x=batch*10
        self.weightor.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, x):
        x = self.weightor(x)
        return x

