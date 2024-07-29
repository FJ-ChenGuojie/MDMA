import torch
import torch.nn as nn
from params import args
class En_DeModel(nn.Module):
    def __init__(self, n_input=[77, 4], layer=[3, 2], n_embedding=128, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(En_DeModel, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding

        self.en_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.en_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.de_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.de_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer[0]):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=n_embedding, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_embedding))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential()
        for i in range(0, layer[1]):
            self.decoder.add_module('de_gru'+str(i), nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=n_embedding, batch_first=True, bidirectional=False))
            self.decoder.add_module('de_ln'+str(i), nn.LayerNorm(n_embedding))
            self.decoder.add_module('de_rl'+str(i), nn.ReLU(True))
            self.decoder.add_module('de_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(n_embedding, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask=None):
        embedding_en_p = self.en_p_embedding(en_p.reshape(-1, self.n_input[0]))
        embedding_en_ts = self.en_ts_embedding(en_ts.reshape(-1, self.n_input[1]))
        en_x = embedding_en_p.reshape(-1, self.len_seq, self.n_embedding) + embedding_en_ts.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, hx = self.encoder[i](en_x.float())
            else:
                en_x = self.encoder[i](en_x)

        embedding_de_p = self.de_p_embedding(de_p.reshape(-1, self.n_input[0]))
        embedding_de_ts = self.de_ts_embedding(de_ts.reshape(-1, self.n_input[1]))
        de_x = embedding_de_p.reshape(-1, 1, self.n_embedding) + embedding_de_ts.reshape(-1, 1, self.n_embedding)

        for i in range(len(self.decoder)):
            if i % 4 == 0:
                de_x, _ = self.decoder[i](de_x.float(), hx)
            else:
                de_x = self.decoder[i](de_x)

        out_put = self.softmax(self.linear(de_x.reshape(-1, self.n_embedding)))
        return out_put

class En_DeModel_2(nn.Module):
    def __init__(self, n_input=[77, 4], layer=[3, 2], n_embedding=128, n_hidden=1024, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(En_DeModel_2, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden

        self.en_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.en_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.en_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)
        self.de_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.de_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.de_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer[0]):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_hidden))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential()
        for i in range(0, layer[1]):
            self.decoder.add_module('de_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.decoder.add_module('de_ln'+str(i), nn.LayerNorm(n_hidden))
            self.decoder.add_module('de_rl'+str(i), nn.ReLU(True))
            self.decoder.add_module('de_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(self.n_hidden, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask):
        embedding_en_p = self.en_p_embedding(en_p.reshape(-1, self.n_input[0]))
        embedding_en_ts = self.en_ts_embedding(en_ts.reshape(-1, self.n_input[1]))
        en_x = self.en_linear(embedding_en_p+embedding_en_ts)
        en_x = en_x.reshape(-1, self.len_seq, self.n_hidden)
        hidden_list = []
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, hx = self.encoder[i](en_x.float())
                hidden_list.append(en_x)
            else:
                en_x = self.encoder[i](en_x)

        embedding_de_p = self.de_p_embedding(de_p.reshape(-1, self.n_input[0]))
        embedding_de_ts = self.de_ts_embedding(de_ts.reshape(-1, self.n_input[1]))
        de_x = self.de_linear(embedding_de_p + embedding_de_ts)
        de_x = de_x.reshape(-1, self.len_seq, self.n_hidden)

        for i in range(len(self.decoder)):
            if i % 4 == 0:
                de_x, _ = self.decoder[i](de_x.float(), hx=hx)
            else:
                de_x = self.decoder[i](de_x)

        out_put = self.softmax(self.linear(de_x[:, -1, :]))
        return out_put, hidden_list

class En_DeModel_3(nn.Module):
    def __init__(self, n_input=[77, 4], layer=[3, 2], n_embedding=128, n_hidden=1024, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(En_DeModel_3, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden

        self.en_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.en_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.en_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)
        self.de_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.de_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.de_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer[0]):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_hidden))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential()
        for i in range(0, layer[1]):
            self.decoder.add_module('de_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.decoder.add_module('de_ln'+str(i), nn.LayerNorm(n_hidden))
            self.decoder.add_module('de_rl'+str(i), nn.ReLU(True))
            self.decoder.add_module('de_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(self.n_hidden, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask):
        embedding_en_p = self.en_p_embedding(en_p.reshape(-1, self.n_input[0]))
        embedding_en_ts = self.en_ts_embedding(en_ts.reshape(-1, self.n_input[1]))
        en_x = self.en_linear(embedding_en_p+embedding_en_ts)
        en_x = en_x.reshape(-1, self.len_seq, self.n_hidden)
        hidden_list = []
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, en_hx = self.encoder[i](en_x.float())

            else:
                en_x = self.encoder[i](en_x)

        embedding_de_p = self.de_p_embedding(de_p.reshape(-1, self.n_input[0]))
        embedding_de_ts = self.de_ts_embedding(de_ts.reshape(-1, self.n_input[1]))
        de_x = self.de_linear(embedding_de_p + embedding_de_ts)
        de_x = de_x.reshape(-1, self.len_seq, self.n_hidden)

        for i in range(len(self.decoder)):
            if i % 4 == 0:
                #de_x, _ = self.decoder[i](de_x.float(), hx=en_hx)
                de_x, _ = self.decoder[i](de_x.float())
                hidden_list.append(de_x)
            else:
                de_x = self.decoder[i](de_x)

        #out_put = self.softmax(self.linear(de_x[:, -1, :]))
        out_put = torch.mean(de_x, dim=1)
        return out_put, hidden_list


class En_DeModel_3_wot_time(nn.Module):
    def __init__(self, n_input=[77, 4], layer=[3, 2], n_embedding=128, n_hidden=1024, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(En_DeModel_3_wot_time, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden

        self.en_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.en_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)
        self.de_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.de_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer[0]):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_hidden))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential()
        for i in range(0, layer[1]):
            self.decoder.add_module('de_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.decoder.add_module('de_ln'+str(i), nn.LayerNorm(n_hidden))
            self.decoder.add_module('de_rl'+str(i), nn.ReLU(True))
            self.decoder.add_module('de_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(self.n_hidden, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask):
        embedding_en_p = self.en_p_embedding(en_p.reshape(-1, self.n_input[0]))
        en_x = self.en_linear(embedding_en_p)
        en_x = en_x.reshape(-1, self.len_seq, self.n_hidden)
        hidden_list = []
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, en_hx = self.encoder[i](en_x.float())

            else:
                en_x = self.encoder[i](en_x)

        embedding_de_p = self.de_p_embedding(de_p.reshape(-1, self.n_input[0]))
        de_x = self.de_linear(embedding_de_p)
        de_x = de_x.reshape(-1, self.len_seq, self.n_hidden)

        for i in range(len(self.decoder)):
            if i % 4 == 0:
                #de_x, _ = self.decoder[i](de_x.float(), hx=en_hx)
                de_x, _ = self.decoder[i](de_x.float())
                hidden_list.append(de_x)
            else:
                de_x = self.decoder[i](de_x)

        #out_put = self.softmax(self.linear(de_x[:, -1, :]))
        out_put = torch.mean(de_x, dim=1)
        return out_put, hidden_list

class En_DeModel_3_wot_property(nn.Module):
    def __init__(self, n_input=[77, 4], layer=[3, 2], n_embedding=128, n_hidden=1024, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(En_DeModel_3_wot_property, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden


        self.en_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.en_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)

        self.de_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)
        self.de_linear = nn.Linear(self.n_embedding, self.n_hidden, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer[0]):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_hidden))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential()
        for i in range(0, layer[1]):
            self.decoder.add_module('de_gru'+str(i), nn.GRU(input_size=n_hidden, num_layers=1, hidden_size=n_hidden, batch_first=True, bidirectional=False))
            self.decoder.add_module('de_ln'+str(i), nn.LayerNorm(n_hidden))
            self.decoder.add_module('de_rl'+str(i), nn.ReLU(True))
            self.decoder.add_module('de_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(self.n_hidden, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask):

        embedding_en_ts = self.en_ts_embedding(en_ts.reshape(-1, self.n_input[1]))
        en_x = self.en_linear(embedding_en_ts)
        en_x = en_x.reshape(-1, self.len_seq, self.n_hidden)
        hidden_list = []
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, en_hx = self.encoder[i](en_x.float())

            else:
                en_x = self.encoder[i](en_x)


        embedding_de_ts = self.de_ts_embedding(de_ts.reshape(-1, self.n_input[1]))
        de_x = self.de_linear(embedding_de_ts)
        de_x = de_x.reshape(-1, self.len_seq, self.n_hidden)

        for i in range(len(self.decoder)):
            if i % 4 == 0:
                #de_x, _ = self.decoder[i](de_x.float(), hx=en_hx)
                de_x, _ = self.decoder[i](de_x.float())
                hidden_list.append(de_x)
            else:
                de_x = self.decoder[i](de_x)

        #out_put = self.softmax(self.linear(de_x[:, -1, :]))
        out_put = torch.mean(de_x, dim=1)
        return out_put, hidden_list

class RNNModel(nn.Module):
    def __init__(self, n_input=[77, 4], layer=3, n_embedding=128, dropout_rate=0.1, len_seq=5, device='cuda:0'):
        super(RNNModel, self).__init__()
        self.n_input = n_input
        self.device = device
        self.len_seq = len_seq
        self.n_embedding = n_embedding

        self.en_p_embedding = nn.Linear(n_input[0], n_embedding, bias=True)
        self.en_ts_embedding = nn.Linear(n_input[1], n_embedding, bias=True)

        self.encoder = nn.Sequential()
        for i in range(0, layer):
            self.encoder.add_module('en_gru'+str(i), nn.GRU(input_size=n_embedding, num_layers=1, hidden_size=n_embedding, batch_first=True, bidirectional=False))
            self.encoder.add_module('en_ln'+str(i), nn.LayerNorm(n_embedding))
            self.encoder.add_module('en_rl'+str(i), nn.ReLU(True))
            self.encoder.add_module('en_drop'+str(i), nn.Dropout(dropout_rate))

        self.linear = nn.Linear(n_embedding, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, en_p, en_ts, de_p, de_ts, mask):
        embedding_en_p = self.en_p_embedding(de_p.reshape(-1, self.n_input[0]))
        embedding_en_ts = self.en_ts_embedding(de_ts.reshape(-1, self.n_input[1]))
        en_x = embedding_en_p.reshape(-1, self.len_seq, self.n_embedding) + embedding_en_ts.reshape(-1, self.len_seq, self.n_embedding)
        for i in range(len(self.encoder)):
            if i % 4 == 0:
                en_x, hx = self.encoder[i](en_x.float())
            else:
                en_x = self.encoder[i](en_x)

        out_put = self.softmax(self.linear(en_x[:, -1, :]))
        return out_put

class Classfier_inte(nn.Module):
    def __init__(self, n_hidden=1024, num_cluster=4):
        super(Classfier_inte, self).__init__()

        self.linear = nn.Sequential()
        self.linear.add_module('ff1', nn.Linear(n_hidden*(num_cluster-1), 2048))
        self.linear.add_module('gelu', nn.ReLU(True))
        self.linear.add_module('ff2', nn.Linear(2048, n_hidden))
        self.linear.add_module('dp1', nn.Dropout(0.2))
        self.linear.add_module('ln1', nn.Linear(n_hidden, 2))
        # self.linear.add_module('softmax', nn.Softmax(dim=1))
        self.linear.add_module('leakrelu', nn.LeakyReLU())

    def forward(self, feature):
        out_put = self.linear(feature)
        return out_put

class Classfier(nn.Module):
    def __init__(self, n_hidden=1024):
        super(Classfier, self).__init__()

        self.linear = nn.Sequential()
        self.linear.add_module('ln', nn.Linear(n_hidden, 2))
        self.linear.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, feature):
        out_put = self.linear(feature)
        return out_put
