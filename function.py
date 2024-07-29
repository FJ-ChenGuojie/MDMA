import os
import time

import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import sys
sys.setrecursionlimit(5000)
import torch.optim as optim
from params import args
import torch.nn as nn
from model import En_DeModel
from loss.loss import H_Distance
from base.loss_transfer import TransferLoss
import math

def basic_train(f_model, c_model, train_loader, valid_loader, train_epoch):
    f_model_path = args.root + 'f_basic_best_model.pkl'
    c_model_path = args.root + 'c_basic_best_model.pkl'
    optimizer = optim.Adam(
        list(f_model.parameters())+list(c_model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    best_acc = 0
    best_err = np.Inf
    iter_data = iter(train_loader)
    num_iter = len(train_loader)
    iter_valid_data = iter(valid_loader)
    for epoch in range(train_epoch):
        f_model.train()
        c_model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(train_loader) == 0:
                iter_data = iter(train_loader)

            feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
            output = c_model(feature)
            err = criterion(output, batch_label)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f   ' % (epoch, i + 1, num_iter, err.data.cpu().numpy()))
            sys.stdout.flush()

        print('    ')
        num_container = 0
        acc = 0
        err_valid = 0
        f_model.eval()
        c_model.eval()
        start_time = time.time()
        for i in range(len(valid_loader)):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(valid_loader) == 0:
                iter_valid_data = iter(valid_loader)

            with torch.no_grad():
                feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
                output = c_model(feature)

                err_valid += criterion(output, batch_label)

                num_container += len(output)
                pred_cls = output.data.max(1)[1]
                acc += pred_cls.eq(batch_label.data).cpu().sum()

            sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, len(valid_loader)))
            sys.stdout.flush()

        end_time = time.time()
        time_speed = len(valid_loader)/(end_time - start_time)
        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(valid_loader), acc / num_container))
        # if err_valid / len(valid_loader) < best_err:
        #     best_err = err_valid / len(valid_loader)
        #     torch.save(f_model.state_dict(), f_model_path)
        #     torch.save(c_model.state_dict(), c_model_path)
        args.epoch_acc.append(acc / num_container)
        if acc / num_container > best_acc:
            best_acc = acc / num_container
            torch.save(f_model.state_dict(), f_model_path)
            torch.save(c_model.state_dict(), c_model_path)


        # torch.save(f_model.state_dict(), args.root + 'f_basic_model_' + str(epoch) + '.pkl')
        # torch.save(c_model.state_dict(), args.root + 'c_basic_model_' + str(epoch) + '.pkl')

        scheduler.step()

def basic_train_epoch(f_model, c_model, train_loader, valid_loader, train_epoch):
    f_model_path = args.root + 'f_basic_best_model.pkl'
    c_model_path = args.root + 'c_basic_best_model.pkl'
    optimizer = optim.Adam(
        list(f_model.parameters())+list(c_model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    best_acc = 0
    best_err = np.Inf
    iter_data = iter(train_loader)
    num_iter = len(train_loader)
    iter_valid_data = iter(valid_loader)
    for epoch in range(train_epoch):
        f_model.train()
        c_model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            ###############################valid###########################
            if i % 200 == 0:
                num_container = 0
                acc = 0
                err_valid = 0
                f_model.eval()
                c_model.eval()
                for i_valid in range(len(valid_loader)):
                    batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
                    batch_en_in = batch_en_in.to(args.device)
                    batch_en_ts = batch_en_ts.to(args.device)
                    batch_de_in = batch_de_in.to(args.device)
                    batch_de_ts = batch_de_ts.to(args.device)
                    batch_label = batch_label.long().to(args.device)

                    if i_valid % len(valid_loader) == 0:
                        iter_valid_data = iter(valid_loader)

                    with torch.no_grad():
                        feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
                        output = c_model(feature)

                        err_valid += criterion_mse(output[:, 1], batch_label.float())

                        num_container += len(output)
                        pred_cls = output.data.max(1)[1]
                        acc += pred_cls.eq(batch_label.data).cpu().sum()


                args.epoch_acc.append(acc / num_container)
                args.epoch_err.append((err_valid / len(valid_loader)).data.cpu().sum())
                if err_valid / len(valid_loader) < best_err:
                    best_err = err_valid / len(valid_loader)
                    torch.save(f_model.state_dict(), f_model_path)
                    torch.save(c_model.state_dict(), c_model_path)
                f_model.train()
                c_model.train()
            ############################################################

            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(train_loader) == 0:
                iter_data = iter(train_loader)

            feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
            output = c_model(feature)
            err = criterion(output, batch_label)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f   ' % (epoch, i + 1, num_iter, err.data.cpu().numpy()))
            sys.stdout.flush()

        # num_container = 0
        # acc = 0
        # err_valid = 0
        # f_model.eval()
        # c_model.eval()
        # for i_valid in range(len(valid_loader)):
        #     batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
        #     batch_en_in = batch_en_in.to(args.device)
        #     batch_en_ts = batch_en_ts.to(args.device)
        #     batch_de_in = batch_de_in.to(args.device)
        #     batch_de_ts = batch_de_ts.to(args.device)
        #     batch_label = batch_label.long().to(args.device)
        #
        #     if i_valid % len(valid_loader) == 0:
        #         iter_valid_data = iter(valid_loader)
        #
        #     with torch.no_grad():
        #         feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
        #         output = c_model(feature)
        #
        #         err_valid += criterion(output, batch_label)
        #
        #         num_container += len(output)
        #         pred_cls = output.data.max(1)[1]
        #         acc += pred_cls.eq(batch_label.data).cpu().sum()
        # args.epoch_acc.append(acc / num_container)
        # f_model.train()
        # c_model.train()
        scheduler.step()

def basic_train_no_valid(model, train_loader, train_epoch):
    model_path = args.root + 'en_de_model.pkl'
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    best_err = np.Inf
    iter_data = iter(train_loader)
    num_iter = len(train_loader)
    for epoch in range(train_epoch):
        model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(train_loader) == 0:
                iter_data = iter(train_loader)

            output, _ = model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
            err = criterion(output, batch_label)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_class: %f   ' % (epoch, i + 1, num_iter, err.data.cpu().numpy()))
            sys.stdout.flush()

        print('    ')
        scheduler.step()

def basic_test(f_model, c_model, test_loader):
    iter_test_data = iter(test_loader)
    criterion = nn.CrossEntropyLoss()
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    num_container = 0
    acc = 0
    err_valid = 0
    err_mae = 0
    err_mse = 0
    f_model.eval()
    c_model.eval()
    for i in range(len(test_loader)):
        batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_test_data.next()
        batch_en_in = batch_en_in.to(args.device)
        batch_en_ts = batch_en_ts.to(args.device)
        batch_de_in = batch_de_in.to(args.device)
        batch_de_ts = batch_de_ts.to(args.device)
        batch_label = batch_label.long().to(args.device)

        if i % len(test_loader) == 0:
            iter_test_data = iter(test_loader)

        with torch.no_grad():
            feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
            output = c_model(feature)
            err_valid += criterion(output, batch_label)
            err_mae += abs(output[:, 1].reshape(1).item()-batch_label.item())
            err_mse += pow(output[:, 1].reshape(1).item()-batch_label.item(), 2)
            # err_mae += criterion_mae(output[0][1].reshape(1), batch_label)
            # err_mse += criterion_mse(output[0][1].reshape(1), batch_label)

            num_container += len(output)
            pred_cls = output.data.max(1)[1]
            acc += pred_cls.eq(batch_label.data).cpu().sum()

        sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, len(test_loader)))
        sys.stdout.flush()
    print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(test_loader), acc / num_container))
    return err_mae / len(test_loader), pow(err_mse / len(test_loader), 0.5), acc / num_container

def stage_1_train_fast(cluster_data_list, valid_data_list, train_epoch):
    for target_ind in range(1, len(cluster_data_list)+1):
        for source_ind in [ii for ii in range(1, len(cluster_data_list)+1) if ii != target_ind]:
            cluster_s_data = cluster_data_list[source_ind-1]
            cluster_t_data = cluster_data_list[target_ind-1]
            valid_s_data = valid_data_list[source_ind-1]
            valid_t_data = valid_data_list[target_ind-1]

            t_f_model_path = args.root + str(len(cluster_data_list)) + '_' + str(source_ind) + '_' + str(target_ind) + '_f_model.pkl'
            t_c_model_path = args.root + str(len(cluster_data_list)) + '_' + str(source_ind) + '_' + str(target_ind) + '_c_model.pkl'
            # s_f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
            #                                   layer=[3, 3],
            #                                   n_embedding=128,
            #                                   n_hidden=1024,
            #                                   dropout_rate=0.1,
            #                                   len_seq=5,
            #                                   device=args.device).to(args.device)
            # s_c_model = En_DeModel.Classfier(n_hidden=1024).to(args.device)

            t_f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                                layer=[4, 3],
                                                n_embedding=128,
                                                n_hidden=256,
                                                dropout_rate=0.1,
                                                len_seq=5,
                                                device=args.device).to(args.device)
            t_c_model = En_DeModel.Classfier(n_hidden=256).to(args.device)

            f_model_basic_path = args.root + 'f_basic_best_model.pkl'
            c_model_basic_path = args.root + 'c_basic_best_model.pkl'
            t_f_model.load_state_dict(torch.load(f_model_basic_path))
            t_c_model.load_state_dict(torch.load(c_model_basic_path))

            # optimizer_s = optim.Adam(
            #     list(s_f_model.parameters())+list(s_c_model.parameters()),
            #     lr=args.lr*0.1,
            #     betas=(0.5, 0.9))
            optimizer_t = optim.Adam(
                list(t_f_model.parameters())+list(t_c_model.parameters()),
                lr=args.lr*0.01,
                betas=(0.5, 0.9))
            # scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=20, gamma=0.8)
            scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, step_size=5, gamma=0.8)
            criterion = nn.CrossEntropyLoss()
            iter_cluster_s_data = iter(cluster_s_data)
            iter_cluster_t_data = iter(cluster_t_data)
            iter_valid_data = iter(valid_t_data)

            num_iter = len(cluster_t_data) if len(cluster_t_data) < len(cluster_s_data) else len(cluster_s_data)
            num_valid_iter = len(valid_t_data)

            weight_list = []
            dist_old_list = []
            dist_new_list = []
            for index in range(0, args.layer[0]):
                weight_list.append((1.0 / args.seq_len * torch.ones(args.seq_len)).to(args.device))
                dist_old_list.append(torch.zeros(args.seq_len).to(args.device))
                dist_new_list.append(torch.zeros(args.seq_len).to(args.device))

            best_err = np.Inf
            for epoch in range(1, train_epoch + 1):
                # s_f_model.train()
                # s_c_model.train()
                t_f_model.train()
                t_c_model.train()
                torch.cuda.empty_cache()

                dist_new_list.clear()
                for index in range(0, args.layer[0]):
                    dist_new_list.append(torch.zeros(args.seq_len).to(args.device))

                for i in range(0, num_iter):
                    cluster_s_en_in, cluster_s_en_ts, cluster_s_en_loc, cluster_s_de_in, cluster_s_de_ts, cluster_s_de_loc, cluster_s_label, cluster_s_label_mask = iter_cluster_s_data.next()
                    cluster_t_en_in, cluster_t_en_ts, cluster_t_en_loc, cluster_t_de_in, cluster_t_de_ts, cluster_t_de_loc, cluster_t_label, cluster_t_label_mask = iter_cluster_t_data.next()

                    cluster_s_en_in = cluster_s_en_in.to(args.device)
                    cluster_s_en_ts = cluster_s_en_ts.to(args.device)
                    cluster_s_de_in = cluster_s_de_in.to(args.device)
                    cluster_s_de_ts = cluster_s_de_ts.to(args.device)
                    cluster_s_label = cluster_s_label.long().to(args.device)

                    cluster_t_en_in = cluster_t_en_in.to(args.device)
                    cluster_t_en_ts = cluster_t_en_ts.to(args.device)
                    cluster_t_de_in = cluster_t_de_in.to(args.device)
                    cluster_t_de_ts = cluster_t_de_ts.to(args.device)
                    cluster_t_label = cluster_t_label.long().to(args.device)

                    if i % len(cluster_s_data) == 0:
                        iter_cluster_s_data = iter(cluster_s_data)
                    if i % len(cluster_t_data) == 0:
                        iter_cluster_t_data = iter(cluster_t_data)

                    if len(cluster_s_en_in) != len(cluster_t_en_in):
                        continue

                    s_feature, hidden_s_list = t_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                    s_output = t_c_model(s_feature)
                    # s_feature, hidden_s_list = s_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                    # s_output = s_c_model(s_feature)
                    class_s_err = criterion(s_output, cluster_s_label)
                    t_feature, hidden_t_list = t_f_model(cluster_t_en_in, cluster_t_en_ts, cluster_t_de_in, cluster_t_de_ts, cluster_t_label_mask)
                    t_output = t_c_model(t_feature)
                    class_t_err = criterion(t_output, cluster_t_label)

                    dis_t_err = 0

                    criterion_transder = TransferLoss(loss_type='mmd', input_dim=1024)
                    for layer_ind in range(2, args.layer[0]):
                        for seq_ind in range(0, args.seq_len):
                            dis = criterion_transder.compute(hidden_s_list[layer_ind][:, seq_ind, :], hidden_t_list[layer_ind][:, seq_ind, :])
                            dis_t_err = dis_t_err + weight_list[layer_ind][seq_ind] * dis
                            dist_new_list[layer_ind][seq_ind] += dis
                            pass
                    t_err = class_s_err + class_t_err + 0.0 * dis_t_err * 0.2
                    optimizer_t.zero_grad()
                    t_err.backward()
                    optimizer_t.step()

                    # s_feature, hidden_s_list = s_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                    # s_output = s_c_model(s_feature)
                    # class_s_err = criterion(s_output, cluster_s_label)
                    # optimizer_s.zero_grad()
                    # class_s_err.backward()
                    # optimizer_s.step()

                    sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_t_class: %f , err_t_dis: %f , err_s_class: %f' % (epoch, i, num_iter, \
                                                                                                                               class_t_err.data.cpu().numpy(), dis_t_err.data.cpu().numpy(), \
                                                                                                                               class_s_err.data.cpu().numpy()))
                    sys.stdout.flush()

                print('    ')
                err_valid = 0
                t_f_model.eval()
                t_c_model.eval()
                for i in range(num_valid_iter):
                    batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
                    batch_en_in = batch_en_in.to(args.device)
                    batch_en_ts = batch_en_ts.to(args.device)
                    batch_de_in = batch_de_in.to(args.device)
                    batch_de_ts = batch_de_ts.to(args.device)
                    batch_label = batch_label.long().to(args.device)

                    if i % len(valid_t_data) == 0:
                        iter_valid_data = iter(valid_t_data)

                    with torch.no_grad():
                        feature, _ = t_f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, mask=batch_label_mask)
                        output = t_c_model(feature)
                        err_valid += criterion(output, batch_label)

                if err_valid < best_err:
                    best_err = err_valid
                    torch.save(t_f_model.state_dict(), t_f_model_path)
                    torch.save(t_c_model.state_dict(), t_c_model_path)

                # scheduler_s.step()
                scheduler_t.step()
                for layer_ind in range(2, args.layer[0]):
                    epsilon = 1e-5
                    dist_old = dist_old_list[layer_ind].detach()
                    dist_new = dist_new_list[layer_ind].detach()
                    ind = dist_new > dist_old + epsilon
                    weight_list[layer_ind] = weight_list[layer_ind][ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
                    weight_norm = torch.norm(weight_list[layer_ind], dim=0, p=1)
                    weight_list[layer_ind] = weight_list[layer_ind] / weight_norm.t().reshape(1).repeat(len(weight_list[layer_ind]))
                    pass

def stage_1_train_select(cluster_data_list, valid_data, cluster_target,  train_epoch):
    target_ind = cluster_target+1
    for source_ind in [ii for ii in range(1, len(cluster_data_list) + 1) if ii != target_ind]:
        cluster_s_data = cluster_data_list[source_ind - 1]
        cluster_t_data = cluster_data_list[target_ind - 1]
        best_acc = 0

        t_f_model_path = args.root + str(len(cluster_data_list)) + '_' + str(source_ind) + '_' + str(
            target_ind) + '_f_model.pkl'
        t_c_model_path = args.root + str(len(cluster_data_list)) + '_' + str(source_ind) + '_' + str(
            target_ind) + '_c_model.pkl'

        # s_f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
        #                                   layer=[3, 3],
        #                                   n_embedding=128,
        #                                   n_hidden=1024,
        #                                   dropout_rate=0.1,
        #                                   len_seq=5,
        #                                   device=args.device).to(args.device)
        # s_c_model = En_DeModel.Classfier(n_hidden=1024).to(args.device)

        t_f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                            layer=[3, 3],
                                            n_embedding=128,
                                            n_hidden=1024,
                                            dropout_rate=0.1,
                                            len_seq=5,
                                            device=args.device).to(args.device)
        t_c_model = En_DeModel.Classfier(n_hidden=1024).to(args.device)

        f_model_basic_path = args.root + 'f_basic_best_model.pkl'
        c_model_basic_path = args.root + 'c_basic_best_model.pkl'
        t_f_model.load_state_dict(torch.load(f_model_basic_path))
        t_c_model.load_state_dict(torch.load(c_model_basic_path))

        # optimizer_s = optim.Adam(
        #     list(s_f_model.parameters())+list(s_c_model.parameters()),
        #     lr=args.lr*0.1,
        #     betas=(0.5, 0.9))
        optimizer_t = optim.Adam(
            list(t_f_model.parameters()) + list(t_c_model.parameters()),
            lr=args.lr * 0.01,
            betas=(0.5, 0.9))
        # scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=20, gamma=0.8)
        scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, step_size=5, gamma=0.8)
        criterion = nn.CrossEntropyLoss()
        iter_cluster_s_data = iter(cluster_s_data)
        iter_cluster_t_data = iter(cluster_t_data)
        iter_valid_data = iter(valid_data)
        num_valid = len(valid_data)

        num_iter = len(cluster_t_data) if len(cluster_t_data) < len(cluster_s_data) else len(cluster_s_data)

        weight_list = []
        dist_old_list = []
        dist_new_list = []
        for index in range(0, args.layer[0]):
            weight_list.append((1.0 / args.seq_len * torch.ones(args.seq_len)).to(args.device))
            dist_old_list.append(torch.zeros(args.seq_len).to(args.device))
            dist_new_list.append(torch.zeros(args.seq_len).to(args.device))

        best_err = np.Inf
        for epoch in range(1, train_epoch + 1):
            # s_f_model.train()
            # s_c_model.train()
            t_f_model.train()
            t_c_model.train()
            torch.cuda.empty_cache()

            dist_new_list.clear()
            for index in range(0, args.layer[0]):
                dist_new_list.append(torch.zeros(args.seq_len).to(args.device))

            for i in range(0, num_iter):
                cluster_s_en_in, cluster_s_en_ts, cluster_s_en_loc, cluster_s_de_in, cluster_s_de_ts, cluster_s_de_loc, cluster_s_label, cluster_s_label_mask = iter_cluster_s_data.next()
                cluster_t_en_in, cluster_t_en_ts, cluster_t_en_loc, cluster_t_de_in, cluster_t_de_ts, cluster_t_de_loc, cluster_t_label, cluster_t_label_mask = iter_cluster_t_data.next()

                cluster_s_en_in = cluster_s_en_in.to(args.device)
                cluster_s_en_ts = cluster_s_en_ts.to(args.device)
                cluster_s_de_in = cluster_s_de_in.to(args.device)
                cluster_s_de_ts = cluster_s_de_ts.to(args.device)
                cluster_s_label = cluster_s_label.long().to(args.device)

                cluster_t_en_in = cluster_t_en_in.to(args.device)
                cluster_t_en_ts = cluster_t_en_ts.to(args.device)
                cluster_t_de_in = cluster_t_de_in.to(args.device)
                cluster_t_de_ts = cluster_t_de_ts.to(args.device)
                cluster_t_label = cluster_t_label.long().to(args.device)

                if i % len(cluster_s_data) == 0:
                    iter_cluster_s_data = iter(cluster_s_data)
                if i % len(cluster_t_data) == 0:
                    iter_cluster_t_data = iter(cluster_t_data)

                if len(cluster_s_en_in) != len(cluster_t_en_in):
                    continue

                s_feature, hidden_s_list = t_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts,
                                                     cluster_s_label_mask)
                s_output = t_c_model(s_feature)
                # s_feature, hidden_s_list = s_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                # s_output = s_c_model(s_feature)
                class_s_err = criterion(s_output, cluster_s_label)
                t_feature, hidden_t_list = t_f_model(cluster_t_en_in, cluster_t_en_ts, cluster_t_de_in, cluster_t_de_ts,
                                                     cluster_t_label_mask)
                t_output = t_c_model(t_feature)
                class_t_err = criterion(t_output, cluster_t_label)

                dis_t_err = 0

                criterion_transder = TransferLoss(loss_type='mmd', input_dim=1024)
                for layer_ind in range(2, args.layer[0]):
                    for seq_ind in range(0, args.seq_len):
                        dis = criterion_transder.compute(hidden_s_list[layer_ind][:, seq_ind, :],
                                                         hidden_t_list[layer_ind][:, seq_ind, :])
                        dis_t_err = dis_t_err + weight_list[layer_ind][seq_ind] * dis
                        dist_new_list[layer_ind][seq_ind] += dis
                        pass
                t_err = class_s_err + class_t_err + 0.1 * dis_t_err * 0.2
                optimizer_t.zero_grad()
                t_err.backward()
                optimizer_t.step()

                # s_feature, hidden_s_list = s_f_model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                # s_output = s_c_model(s_feature)
                # class_s_err = criterion(s_output, cluster_s_label)
                # optimizer_s.zero_grad()
                # class_s_err.backward()
                # optimizer_s.step()

                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_t_class: %f , err_t_dis: %f , err_s_class: %f' % (
                    epoch, i, num_iter, \
                    class_t_err.data.cpu().numpy(), dis_t_err.data.cpu().numpy(), \
                    class_s_err.data.cpu().numpy()))
                sys.stdout.flush()

            acc = 0
            err_valid = 0
            num_container = 0

            for valid_i in range(num_valid):
                en_in, en_ts, en_loc, de_in, de_ts, de_loc, label, label_mask = iter_valid_data.next()
                en_in = en_in.to(args.device)
                en_ts = en_ts.to(args.device)
                de_in = de_in.to(args.device)
                de_ts = de_ts.to(args.device)
                label = label.long().to(args.device)
                if valid_i % len(valid_data) == 0:
                    iter_valid_data = iter(valid_data)

                with torch.no_grad():
                    feature, _ = t_f_model(en_in, en_ts, de_in, de_ts, label_mask)
                    output = t_c_model(feature)

                    err_valid += criterion(output, label)

                    num_container += len(output)
                    pred_cls = output.data.max(1)[1]
                    acc += pred_cls.eq(label.data).cpu().sum()

                sys.stdout.write('\r test--->[iter: %d / all %d]   '% (valid_i + 1, len(valid_data)))
                sys.stdout.flush()

            print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(valid_data), acc / num_container))
            if acc / num_container > best_acc:
                best_acc = acc / num_container
                torch.save(t_f_model.state_dict(), t_f_model_path)
                torch.save(t_c_model.state_dict(), t_c_model_path)

            # scheduler_s.step()
            scheduler_t.step()
            for layer_ind in range(0, args.layer[0]):
                epsilon = 1e-5
                dist_old = dist_old_list[layer_ind].detach()
                dist_new = dist_new_list[layer_ind].detach()
                ind = dist_new > dist_old + epsilon
                weight_list[layer_ind] = weight_list[layer_ind][ind] * (
                            1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
                weight_norm = torch.norm(weight_list[layer_ind], dim=0, p=1)
                weight_list[layer_ind] = weight_list[layer_ind] / weight_norm.t().reshape(1).repeat(
                    len(weight_list[layer_ind]))
                pass

def stage_2_train_fast(target_data, valid_data, cluster_num, cluster_target, target_num, train_epoch):
    f_model_list = []
    f_model_path_list = []
    list_params_list = []
    for source_ind in [i for i in range(1, cluster_num+1) if i != cluster_target+1]:
        f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                        layer=[4, 3],
                                        n_embedding=128,
                                        n_hidden=256,
                                        dropout_rate=0.1,
                                        len_seq=5,
                                        device=args.device).to(args.device)

        f_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target+1) + '_f_model.pkl'
        f_model.load_state_dict(torch.load(f_model_path))
        f_model_list.append(f_model)

        # list_params_list += list(f_model_list[-1].parameters())
        # f_model_finetune_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target+1)+'_' +\
        #                       str(target_num+1) + '_finetune_f_model.pkl'
        # f_model_path_list.append(f_model_finetune_path)

    c_model = En_DeModel.Classfier_inte(n_hidden=256, num_cluster=args.num_cluster).to(args.device)
    c_model_finetune_path = args.root + str(cluster_num) + '_' + \
                            str(target_num + 1) + '_finetune_c_model.pkl'

    for m_index in range(len(f_model_list)):
        if m_index == 0:
            f_param = list(f_model_list[0].parameters())
        else:
            f_param = f_param+list(f_model_list[m_index].parameters())

    #optimizer = optim.SGD(list_params_list, lr=args.lr*0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer = optim.Adam([{'params': c_model.parameters(), 'lr': args.lr},
                            {'params': f_param, 'lr': args.lr*0.05}],
                           betas=(0.5, 0.9), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    iter_train_data = iter(target_data)
    iter_valid_data = iter(valid_data)

    num_iter = len(target_data)
    best_err = np.Inf
    best_acc = 0
    for epoch in range(1, train_epoch + 1):
        for model_ind in range(cluster_num - 1):
            f_model_list[model_ind].train()
        c_model.train()
        torch.cuda.empty_cache()
        for i in range(0, num_iter):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_train_data.next()

            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(target_data) == 0:
                iter_train_data = iter(target_data)

            feature_list = []
            for model_ind in range(cluster_num - 1):
                feature, _ = f_model_list[model_ind](batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                feature_list.append(feature)

            feature_inte = torch.cat(feature_list, dim=1)
            output = c_model(feature_inte)
            class_err = criterion(output, batch_label)
            #class_err = criterion(output_all[0][1].reshape(1), batch_label.float())

            optimizer.zero_grad()
            class_err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_t_class: %f ' % (epoch, i + 1, num_iter, class_err.data.cpu().numpy()))
            sys.stdout.flush()

        print('    ')
        num_container = 0
        acc = 0
        err_valid = 0
        for model_ind in range(cluster_num - 1):
            f_model_list[model_ind].eval()
        c_model.eval()
        for i in range(len(valid_data)):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(valid_data) == 0:
                iter_valid_data = iter(valid_data)

            with torch.no_grad():
                feature_list = []
                for model_ind in range(cluster_num - 1):
                    feature, _ = f_model_list[model_ind](batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                    feature_list.append(feature)

                feature_inte = torch.cat(feature_list, dim=1)
                output = c_model(feature_inte)

                err_valid += criterion(output, batch_label)

                num_container += len(output)
                pred_cls = output.data.max(1)[1]
                acc += pred_cls.eq(batch_label.data).cpu().sum()

            sys.stdout.write('\r stage_2_train--->test--->[iter: %d / all %d]           ' % (i + 1, len(valid_data)))
            sys.stdout.flush()
        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(valid_data), acc / num_container))

        # if err_valid / len(valid_data) < best_err:
        #     best_err = err_valid / len(valid_data)
        #     torch.save(c_model.state_dict(), c_model_finetune_path)

        if acc / num_container > best_acc:
            best_acc = acc / num_container
            torch.save(c_model.state_dict(), c_model_finetune_path)
            for path_index in range(len(f_model_list)):
                path = args.root + str(cluster_num) + '_' + \
                            str(target_num + 1) + '_finetune_f_model_' + str(path_index+1) +'.pkl'
                torch.save(f_model_list[path_index].state_dict(), path)

        scheduler.step()


def stage_2_test_fast(test_data, cluster_num, cluster_target, target_num):
    num_container = 0
    acc = 0
    err_valid = 0
    err_mse = 0
    err_mae = 0
    f_model_list = []
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    for i, source_ind in enumerate([index for index in range(1, cluster_num+1) if index != cluster_target+1]):
        f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                          layer=[4, 3],
                                          n_embedding=128,
                                          n_hidden=256,
                                          dropout_rate=0.1,
                                          len_seq=5,
                                          device=args.device).to(args.device)
        # f_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target+1)+'_' +\
        #                       str(target_num+1) + '_finetune_f_model.pkl'
        # f_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target + 1) + '_f_model.pkl'
        f_model_path = args.root + str(cluster_num) + '_' + str(target_num + 1) + '_finetune_f_model_' + str(i+1) +'.pkl'
        f_model.load_state_dict(torch.load(f_model_path))
        f_model_list.append(f_model)

    c_model_path = args.root + str(cluster_num) + '_' + str(target_num + 1) + '_finetune_c_model.pkl'
    c_model = En_DeModel.Classfier_inte(n_hidden=256, num_cluster=args.num_cluster).to(args.device)
    c_model.load_state_dict(torch.load(c_model_path))

    for model_ind in range(cluster_num - 1):
        f_model_list[model_ind].eval()
    c_model.eval()
    iter_test_data = iter(test_data)

    for i in range(len(test_data)):
        batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_test_data.next()
        batch_en_in = batch_en_in.to(args.device)
        batch_en_ts = batch_en_ts.to(args.device)
        batch_de_in = batch_de_in.to(args.device)
        batch_de_ts = batch_de_ts.to(args.device)
        batch_label = batch_label.long().to(args.device)
        output_all = 0

        if i % len(test_data) == 0:
            iter_test_data = iter(test_data)

        with torch.no_grad():
            feature_list = []
            for model_ind in range(cluster_num - 1):
                feature, _ = f_model_list[model_ind](batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                feature_list.append(feature)

            feature_inte = torch.cat(feature_list, dim=1)
            output = c_model(feature_inte)
            err_valid += criterion(output, batch_label)
            err_mse += criterion_mse(output[0][1].reshape(1), batch_label.float())
            err_mae += criterion_mae(output[0][1].reshape(1), batch_label.float())

            num_container += len(output)
            pred_cls = output.data.max(1)[1]
            acc += pred_cls.eq(batch_label.data).cpu().sum()

        sys.stdout.write('\r Area: %d, stage_2--->test--->[iter: %d / all %d]           ' % (target_num, i + 1, len(test_data)))
        sys.stdout.flush()
    print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(test_data), acc / num_container))
    return err_mae / len(test_data), pow(err_mse / len(test_data), 0.5), acc / num_container

def stage_1_test_respectively(test_data, cluster_num, cluster_target, target_num):
    num_container = 0
    acc = 0
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    err_valid = 0
    err_mse = 0
    err_mae = 0
    f_model_list = []
    c_model_list = []
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    for source_ind in [index for index in range(1, cluster_num+1) if index != cluster_target+1]:
        f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                          layer=[3, 3],
                                          n_embedding=128,
                                          n_hidden=1024,
                                          dropout_rate=0.1,
                                          len_seq=5,
                                          device=args.device).to(args.device)
        c_model = En_DeModel.Classfier(n_hidden=1024).to(args.device)
        # f_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target+1)+'_' +\
        #                       str(target_num+1) + '_finetune_f_model.pkl'
        f_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target + 1) + '_f_model.pkl'
        c_model_path = args.root + str(cluster_num) + '_' + str(source_ind) + '_' + str(cluster_target + 1) + '_c_model.pkl'
        f_model.load_state_dict(torch.load(f_model_path))
        c_model.load_state_dict(torch.load(c_model_path))
        f_model_list.append(f_model)
        c_model_list.append(c_model)


    c_agg_model_path = args.root + str(cluster_num) + '_' + str(target_num + 1) + '_finetune_c_model.pkl'
    c_agg_model = En_DeModel.Classfier_inte(n_hidden=1024, num_cluster=args.num_cluster).to(args.device)
    c_agg_model.load_state_dict(torch.load(c_agg_model_path))

    for model_ind in range(cluster_num - 1):
        f_model_list[model_ind].eval()
        c_model_list[model_ind].eval()
    c_agg_model.eval()
    iter_test_data = iter(test_data)

    for i in range(len(test_data)):
        batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_test_data.next()
        batch_en_in = batch_en_in.to(args.device)
        batch_en_ts = batch_en_ts.to(args.device)
        batch_de_in = batch_de_in.to(args.device)
        batch_de_ts = batch_de_ts.to(args.device)
        batch_label = batch_label.long().to(args.device)
        output_all = 0

        if i % len(test_data) == 0:
            iter_test_data = iter(test_data)

        with torch.no_grad():
            feature_list = []
            output_list = []
            for model_ind in range(cluster_num - 1):
                feature, _ = f_model_list[model_ind](batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                output = c_model_list[model_ind](feature)
                feature_list.append(feature)
                output_list.append(output)

            feature_inte = torch.cat(feature_list, dim=1)
            output = c_agg_model(feature_inte)
            err_valid += criterion(output, batch_label)
            err_mse += criterion_mse(output[0][1].reshape(1), batch_label.float())
            err_mae += criterion_mae(output[0][1].reshape(1), batch_label.float())

            num_container += len(output)
            pred_cls = output.data.max(1)[1]
            pred_cls_1 = output_list[0].data.max(1)[1]
            pred_cls_2 = output_list[1].data.max(1)[1]
            pred_cls_3 = output_list[2].data.max(1)[1]

            acc += pred_cls.eq(batch_label.data).cpu().numpy().sum()
            acc_1 += pred_cls_1.eq(batch_label.data).cpu().numpy().sum()
            acc_2 += pred_cls_2.eq(batch_label.data).cpu().numpy().sum()
            acc_3 += pred_cls_3.eq(batch_label.data).cpu().numpy().sum()

        sys.stdout.write('\r Area: %d, test--->[iter: %d / all %d]           ' % (target_num, i + 1, len(test_data)))
        sys.stdout.flush()
    print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(test_data), acc / num_container))
    return acc / num_container, acc_1 / num_container, acc_2 / num_container, acc_3 / num_container

def RRL_train(model, source_data_list, target_data, train_epoch):
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr * 0.1,
        betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    iter_s_data_list = []
    for i in range(0, len(source_data_list)):
        iter_s_data_list.append(iter(source_data_list[i]))
    iter_t_data = iter(target_data)
    num_iter = len(iter_t_data)
    for epoch in range(1, train_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            s_en_in_list = []
            s_en_ts_list = []
            s_en_loc_list = []
            s_de_in_list = []
            s_de_ts_list = []
            s_de_loc_list = []
            s_label_list = []
            s_label_mask_list = []
            s_domain_label_list = []
            for domain_ind in range(0, len(source_data_list)):
                s_en_in, s_en_ts, s_en_loc, s_de_in, s_de_ts, s_de_loc, s_label, s_label_mask = iter_s_data_list[domain_ind].next()                
                s_en_in_list.append(s_en_in.to(args.device))
                s_en_ts_list.append(s_en_ts.to(args.device))
                s_en_loc_list.append(s_en_loc.to(args.device))
                s_de_in_list.append(s_de_in.to(args.device))
                s_de_ts_list.append(s_de_ts.to(args.device))
                s_de_loc_list.append(s_de_loc.to(args.device))
                s_label_list.append(s_label.long().to(args.device))
                s_label_mask_list.append(s_label_mask.to(args.device))
                s_domain_label_list.append((domain_ind*torch.ones(len(s_label))).to(args.device))

            t_en_in, t_en_ts, t_en_loc, t_de_in, t_de_ts, t_de_loc, t_label, t_label_mask = iter_t_data.next()
            t_en_in = t_en_in.to(args.device)
            t_en_ts = t_en_ts.to(args.device)
            t_en_loc = t_en_loc.to(args.device)
            t_de_in = t_de_in.to(args.device)
            t_de_ts = t_de_ts.to(args.device)
            t_de_loc = t_de_loc.to(args.device)
            t_label = t_label.long().to(args.device)
            t_label_mask = t_label_mask.to(args.device)
            t_domain_label = (len(source_data_list)*torch.ones(len(t_label)).to(args.device))

            s_en_in_all = torch.cat(s_en_in_list, dim=0)
            s_en_ts_all = torch.cat(s_en_ts_list, dim=0)
            s_en_loc_all = torch.cat(s_en_loc_list, dim=0)
            s_de_in_all = torch.cat(s_de_in_list, dim=0)
            s_de_ts_all = torch.cat(s_de_ts_list, dim=0)
            s_de_loc_all = torch.cat(s_de_loc_list, dim=0)
            s_label_all = torch.cat(s_label_list, dim=0)
            s_label_mask_all = torch.cat(s_label_mask_list, dim=0)
            s_domain_label_all = torch.cat(s_domain_label_list, dim=0)

            for domain_ind in range(0, len(source_data_list)):
                if i % len(source_data_list[domain_ind]) == 0:
                    iter_s_data_list[domain_ind] = iter(source_data_list[domain_ind])

            if i % len(target_data) == 0:
                iter_t_data = iter(target_data)

            s_output, hidden_s_list = model(s_en_in_all, s_en_ts_all, s_de_in_all, s_de_ts_all, s_label_mask_all)
            t_output, hidden_t_list = model(t_en_in, t_en_ts, t_de_in, t_de_ts, t_label_mask)

            err_s = criterion(s_output, s_label_all)
            err_t = criterion(t_output, t_label)

            source_features = hidden_s_list[args.layer[0]-1][:, -1, :]
            target_features = hidden_t_list[args.layer[0]-1][:, -1, :]
            features = torch.cat([source_features, target_features])
            labels = torch.cat([s_label_all, t_label])
            domain_label = torch.cat([s_domain_label_all, t_domain_label])
            dis_err = H_Distance(features, labels, domain_label, device=args.device)

            err = err_s + err_t + dis_err
            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_t_class: %f , err_t_dis: %f , err_s_class: %f' % (epoch, i, num_iter, \
                                                                                                                       err_t.data.cpu().numpy(), dis_err.data.cpu().numpy(), \
                                                                                                                       err_s.data.cpu().numpy()))
            sys.stdout.flush()

        print('    ')
        scheduler.step()
        
def DIS_train(model, data_loader_list, train_epoch):
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr * 0.1,
        betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    iter_s_data_list = []
    num_iter = np.Inf
    for i in range(0, len(data_loader_list)):
        iter_s_data_list.append(iter(data_loader_list[i]))
        if len(iter(data_loader_list[i])) < num_iter:
            a = len(iter(data_loader_list[i]))
            num_iter = len(iter(data_loader_list[i]))
    
    for epoch in range(1, train_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            s_hx_list = []
            err_class_all = 0
            for domain_ind in range(0, len(data_loader_list)):
                s_en_in, s_en_ts, s_en_loc, s_de_in, s_de_ts, s_de_loc, s_label, s_label_mask = iter_s_data_list[domain_ind].next()
                s_en_in = s_en_in.to(args.device)
                s_en_ts = s_en_ts.to(args.device)
                s_en_loc = s_en_loc.to(args.device)
                s_de_in = s_de_in.to(args.device)
                s_de_ts = s_de_ts.to(args.device)
                s_de_loc = s_de_loc.to(args.device)
                s_label = s_label.long().to(args.device)
                s_label_mask = s_label_mask.to(args.device)

                output, hx = model(s_en_in, s_en_ts, s_de_in, s_de_ts, s_label_mask)
                err_class = criterion(output, s_label)
                err_class_all += err_class
                s_hx_list.append(hx)

                if i % len(data_loader_list[domain_ind]) == 0:
                    iter_s_data_list[domain_ind] = iter(data_loader_list[domain_ind])

            err_dis_all = 0
            criterion_transder = TransferLoss(loss_type='mmd', input_dim=1024)
            for i_ind in range(0, len(data_loader_list)-1):
                for j_ind in range(i_ind+1, len(data_loader_list)):
                    dis = criterion_transder.compute(s_hx_list[i_ind][-1][:, -1, :], s_hx_list[j_ind][-1][:, -1, :])
                    err_dis_all += dis

            err = err_class_all - 0.1*torch.log(err_dis_all)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], dis err_s_class: %f' % (epoch, i, num_iter, err_dis_all.data.cpu().numpy()))
            sys.stdout.flush()
        print('    ')
        scheduler.step()


def DIS_train(model, data_loader_list, train_epoch):
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr * 0.1,
        betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    iter_s_data_list = []
    num_iter = np.Inf
    for i in range(0, len(data_loader_list)):
        iter_s_data_list.append(iter(data_loader_list[i]))
        if len(iter(data_loader_list[i])) < num_iter:
            a = len(iter(data_loader_list[i]))
            num_iter = len(iter(data_loader_list[i]))

    for epoch in range(1, train_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        for i in range(num_iter):
            s_hx_list = []
            err_class_all = 0
            for domain_ind in range(0, len(data_loader_list)):
                s_en_in, s_en_ts, s_en_loc, s_de_in, s_de_ts, s_de_loc, s_label, s_label_mask = iter_s_data_list[domain_ind].next()
                s_en_in = s_en_in.to(args.device)
                s_en_ts = s_en_ts.to(args.device)
                s_en_loc = s_en_loc.to(args.device)
                s_de_in = s_de_in.to(args.device)
                s_de_ts = s_de_ts.to(args.device)
                s_de_loc = s_de_loc.to(args.device)
                s_label = s_label.long().to(args.device)
                s_label_mask = s_label_mask.to(args.device)

                output, hx = model(s_en_in, s_en_ts, s_de_in, s_de_ts, s_label_mask)
                err_class = criterion(output, s_label)
                err_class_all += err_class
                s_hx_list.append(hx)

                if i % len(data_loader_list[domain_ind]) == 0:
                    iter_s_data_list[domain_ind] = iter(data_loader_list[domain_ind])

            err_dis_all = 0
            criterion_transder = TransferLoss(loss_type='mmd', input_dim=1024)
            for i_ind in range(0, len(data_loader_list) - 1):
                for j_ind in range(i_ind + 1, len(data_loader_list)):
                    dis = criterion_transder.compute(s_hx_list[i_ind][-1][:, -1, :], s_hx_list[j_ind][-1][:, -1, :])
                    err_dis_all += dis

            err = err_class_all + torch.exp(-err_dis_all)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_class: %f     err_s_dis: %f' % (epoch, i, num_iter, err_class_all.data.cpu().numpy(), \
                                                                                                       err_dis_all.data.cpu().numpy()))
            sys.stdout.flush()
        print('    ')
        scheduler.step()

def get_feature(model, train_list_loader):
    model.eval()
    feature = []
    for i, data_loader in enumerate(train_list_loader):
        feature.clear()
        torch.cuda.empty_cache()
        for data_ind, data in enumerate(data_loader):
            if data_ind > len(data_loader)/2:
                break
            batch_en_in = data[0].to(args.device)
            batch_en_ts = data[1].to(args.device)

            batch_de_in = data[3].to(args.device)
            batch_de_ts = data[4].to(args.device)

            batch_label_mask = data[7].to(args.device)

            output, hx = model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
            #hx_tar = hx[-1][:, -1, :]
            feature.append(hx[-1].reshape(len(output), -1))
        path_x = args.root + 'datashift/'+str(i)+'_x.csv'
        path_y = args.root + 'datashift/'+str(i)+'_y.csv'
        x_feature = torch.cat(feature, dim=0)
        y_feature = torch.ones(len(x_feature))*i
        x_numpy = x_feature.data.cpu().numpy()
        y_numpy = y_feature.data.cpu().numpy()
        x_feature = pd.DataFrame(data=x_numpy)
        y_feature = pd.DataFrame(data=y_numpy)
        x_feature.to_csv(path_x, header=False, index=False)
        y_feature.to_csv(path_y, header=False, index=False)
        pass

def stage_1_train_fast_de(cluster_data_list, valid_data_list, train_epoch):
    for target_ind in range(1, len(cluster_data_list)+1):
        for source_ind in [ii for ii in range(1, len(cluster_data_list)+1) if ii != target_ind]:
            cluster_s_data = cluster_data_list[source_ind-1]
            cluster_t_data = cluster_data_list[target_ind-1]
            valid_s_data = valid_data_list[source_ind-1]
            valid_t_data = valid_data_list[target_ind-1]

            model_path = args.root + str(len(cluster_data_list)) + '_' + str(source_ind) + '_' + str(target_ind) + '_model.pkl'
            model = En_DeModel.En_DeModel_2(n_input=[args.n_dim, 4],
                                              layer=[3, 3],
                                              n_embedding=128,
                                              n_hidden=1024,
                                              dropout_rate=0.1,
                                              len_seq=5,
                                              device=args.device).to(args.device)

            model_basic_path = 'F:/partData/restored_5/en_de_model_basic.pkl'
            model.load_state_dict(torch.load(model_basic_path))
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr*0.1,
                betas=(0.5, 0.9))

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
            criterion = nn.CrossEntropyLoss()
            iter_cluster_s_data = iter(cluster_s_data)
            iter_cluster_t_data = iter(cluster_t_data)

            num_iter = len(cluster_t_data) if len(cluster_t_data) < len(cluster_s_data) else len(cluster_s_data)

            weight_list = []
            dist_old_list = []
            dist_new_list = []
            for index in range(0, args.layer[0]):
                weight_list.append((1.0 / args.seq_len * torch.ones(args.seq_len)).to(args.device))
                dist_old_list.append(torch.zeros(args.seq_len).to(args.device))
                dist_new_list.append(torch.zeros(args.seq_len).to(args.device))

            best_err = np.Inf
            for epoch in range(1, train_epoch + 1):
                model.train()
                torch.cuda.empty_cache()
                for i in range(0, num_iter):
                    cluster_s_en_in, cluster_s_en_ts, cluster_s_en_loc, cluster_s_de_in, cluster_s_de_ts, cluster_s_de_loc, cluster_s_label, cluster_s_label_mask = iter_cluster_s_data.next()
                    cluster_t_en_in, cluster_t_en_ts, cluster_t_en_loc, cluster_t_de_in, cluster_t_de_ts, cluster_t_de_loc, cluster_t_label, cluster_t_label_mask = iter_cluster_t_data.next()

                    cluster_s_en_in = cluster_s_en_in.to(args.device)
                    cluster_s_en_ts = cluster_s_en_ts.to(args.device)
                    cluster_s_de_in = cluster_s_de_in.to(args.device)
                    cluster_s_de_ts = cluster_s_de_ts.to(args.device)
                    cluster_s_label = cluster_s_label.long().to(args.device)

                    cluster_t_en_in = cluster_t_en_in.to(args.device)
                    cluster_t_en_ts = cluster_t_en_ts.to(args.device)
                    cluster_t_de_in = cluster_t_de_in.to(args.device)
                    cluster_t_de_ts = cluster_t_de_ts.to(args.device)
                    cluster_t_label = cluster_t_label.long().to(args.device)

                    if i % len(cluster_s_data) == 0:
                        iter_cluster_s_data = iter(cluster_s_data)
                    if i % len(cluster_t_data) == 0:
                        iter_cluster_t_data = iter(cluster_t_data)

                    s_output, hidden_s_list = model(cluster_s_en_in, cluster_s_en_ts, cluster_s_de_in, cluster_s_de_ts, cluster_s_label_mask)
                    t_output, hidden_t_list = model(cluster_t_en_in, cluster_t_en_ts, cluster_t_de_in, cluster_t_de_ts, cluster_t_label_mask)
                    class_t_err = criterion(t_output, cluster_t_label)
                    class_s_err = criterion(s_output, cluster_s_label)

                    dis_t_err = 0

                    criterion_transder = TransferLoss(loss_type='mmd', input_dim=1024)
                    for layer_ind in range(0, args.layer[0]):
                        for seq_ind in range(0, args.seq_len):
                            dis = criterion_transder.compute(hidden_s_list[layer_ind][:, seq_ind, :], hidden_t_list[layer_ind][:, seq_ind, :])
                            dis_t_err = dis_t_err + weight_list[layer_ind][seq_ind] * dis
                            dist_new_list[layer_ind][seq_ind] += dis
                            pass
                    err = class_t_err + 0.1 * dis_t_err
                    optimizer.zero_grad()
                    err.backward()
                    optimizer.step()

                    sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_t_class: %f , err_t_dis: %f , err_s_class: %f' % (epoch, i, num_iter, \
                                                                                                                               class_t_err.data.cpu().numpy(), dis_t_err.data.cpu().numpy(), \
                                                                                                                               class_s_err.data.cpu().numpy()))
                    sys.stdout.flush()

                print('    ')
                scheduler.step()

                for layer_ind in range(0, args.layer[0]):
                    epsilon = 1e-5
                    dist_old = dist_old_list[layer_ind].detach()
                    dist_new = dist_new_list[layer_ind].detach()
                    ind = dist_new > dist_old + epsilon
                    weight_list[layer_ind] = weight_list[layer_ind][ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
                    weight_norm = torch.norm(weight_list[layer_ind], dim=0, p=1)
                    weight_list[layer_ind] = weight_list[layer_ind] / weight_norm.t().reshape(1).repeat(len(weight_list[layer_ind]))
                    pass
            torch.save(model.state_dict(), model_path)

def finetune_cluster_source(train_data, valid_data, num_cluster, cluster_source_ind, cluster_target_ind, domain_target_ind, train_epoch):
    f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                      layer=[3, 3],
                                      n_embedding=128,
                                      n_hidden=1024,
                                      dropout_rate=0.1,
                                      len_seq=5,
                                      device=args.device).to(args.device)

    f_model_path = args.root + str(num_cluster) + '_' + str(cluster_source_ind+1) + '_' + str(cluster_target_ind + 1) + '_f_model.pkl'
    f_model.load_state_dict(torch.load(f_model_path))

    c_model = En_DeModel.Classfier(n_hidden=1024).to(args.device)

    optimizer = optim.Adam([{'params': c_model.parameters(), 'lr': args.lr}],
                           betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    iter_train_data = iter(train_data)
    iter_valid_data = iter(valid_data)

    num_iter = len(train_data)
    best_err = np.Inf
    best_acc = 0
    best_mae = 0
    best_rmse = 0
    for epoch in range(1, train_epoch + 1):
        f_model.train()
        c_model.train()
        torch.cuda.empty_cache()
        for i in range(0, num_iter):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_train_data.next()

            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(train_data) == 0:
                iter_train_data = iter(train_data)

            feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
            output = c_model(feature)
            class_err = criterion(output, batch_label)
            # class_err = criterion(output_all[0][1].reshape(1), batch_label.float())

            optimizer.zero_grad()
            class_err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_t_class: %f ' % (epoch, i + 1, num_iter, class_err.data.cpu().numpy()))
            sys.stdout.flush()

        print('    ')
        num_container = 0
        mae = 0
        mse = 0
        acc = 0
        err_valid = 0

        f_model.eval()
        c_model.eval()
        for i in range(len(valid_data)):
            batch_en_in, batch_en_ts, batch_en_loc, batch_de_in, batch_de_ts, batch_de_loc, batch_label, batch_label_mask = iter_valid_data.next()
            batch_en_in = batch_en_in.to(args.device)
            batch_en_ts = batch_en_ts.to(args.device)
            batch_de_in = batch_de_in.to(args.device)
            batch_de_ts = batch_de_ts.to(args.device)
            batch_label = batch_label.long().to(args.device)

            if i % len(valid_data) == 0:
                iter_valid_data = iter(valid_data)

            with torch.no_grad():
                feature, _ = f_model(batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                output = c_model(feature)
                err_valid += criterion(output, batch_label)

                num_container += len(output)
                pred_cls = output.data.max(1)[1]
                acc += pred_cls.eq(batch_label.data).cpu().sum()
                mae += abs(output[0][1].reshape(1).item()-batch_label.float().item())
                mse += pow(output[0][1].reshape(1).item()-batch_label.float().item(), 2)

            sys.stdout.write('\r test--->[iter: %d / all %d]           ' % (i + 1, len(valid_data)))
            sys.stdout.flush()
        print("Avg Loss = %.4f, Avg Accuracy = %.4f \n" % (err_valid / len(valid_data), acc / num_container))

        if err_valid / len(valid_data) < best_err:
            best_err = err_valid / len(valid_data)
            best_acc = acc/len(valid_data)
            best_mae = mae/len(valid_data)
            best_rmse = pow(mse/len(valid_data), 0.5)

        scheduler.step()

    print('num_cluster: %d, cluster_sourece_ind: %d, cluster_target_ind: %d, domain_target_ind: %d ---- mae: %.3f, rmse: %.3f, acc: %.2f'\
          %(num_cluster, cluster_source_ind+1, cluster_target_ind+1, domain_target_ind, best_mae, best_rmse, best_acc*100))
