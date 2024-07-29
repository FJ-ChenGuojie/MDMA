import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import data.dataprocess as dataprocess
from model import En_DeModel
from clustering import k_medoids
from params import args
import datetime
import numpy as np
import function
import pandas as pd
import time

def count_parameters(model):
    p_list = []
    #########
    for p in model.parameters():
        if p.requires_grad:
            p_list.append(p.numel())
    #########
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
path = 'F:/partData/'+now_time+args.model_dir
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.num_cluster = 4

feat_list, time_list, location_list, label_list, label_mask_list = dataprocess.bay_data_2(args.data_dir)
en_feat_list, en_ts_list, en_location_list, de_feat_list, de_ts_list, de_location_list, label_convert_list, mask_convert_list = dataprocess.data_covert_4(feat_list,
                                                                                                                                                        time_list,
                                                                                                                                                        location_list,
                                                                                                                                                        label_list,
                                                                                                                                                        label_mask_list)

individual_list = []
for index in range(60):
    individual_list.append([index])
train_indivisual_loader_list, valid_indivisual_loader_list, test_indivisual_loader_list, _, _, _ = dataprocess.stack_loader_k_medoid_fold(en_feat_list ,
                                                                                                                                                  en_ts_list ,
                                                                                                                                                  en_location_list ,
                                                                                                                                                  de_feat_list ,
                                                                                                                                                  de_ts_list ,
                                                                                                                                                  de_location_list ,
                                                                                                                                                  label_convert_list ,
                                                                                                                                                  mask_convert_list ,
                                                                                                                                                  individual_list,
                                                                                                                                                  batch_size=[8, 1, 1])

run_time = []
for num_cluster in [4]:
    start_time = time.time()
    args.num_cluster = num_cluster

    cluster_lists_sequence = k_medoids.area_clustering_sequence(feat_list=feat_list, loss_type='mmd', K=args.num_cluster)
    train_loader_list, valid_loader_list, test_loader_list, train_loader_all, valid_loader_all, test_loader_all = dataprocess.stack_loader_k_medoid_fold(en_feat_list ,
                                                                                                                                                      en_ts_list ,
                                                                                                                                                      en_location_list ,
                                                                                                                                                      de_feat_list ,
                                                                                                                                                      de_ts_list ,
                                                                                                                                                      de_location_list ,
                                                                                                                                                      label_convert_list ,
                                                                                                                                                      mask_convert_list ,
                                                                                                                                                      cluster_lists_sequence,
                                                                                                                                                      batch_size=[256, 128, 1])

    args.root = 'F:/partData/restored_MDMA_main/'
    f_model = En_DeModel.En_DeModel_3(n_input=[34, 4],
                                            layer=[4, 3],
                                            n_embedding=128,
                                            n_hidden=256,
                                            dropout_rate=0.1,
                                            len_seq=5,
                                            device=args.device).to(args.device)

    c_model = En_DeModel.Classfier(n_hidden=256).to(args.device)

    f_model_param = count_parameters(f_model)
    c_model_param = count_parameters(c_model)

    function.basic_train(f_model=f_model,
                         c_model=c_model,
                         train_loader=train_loader_all,
                         valid_loader=valid_loader_all,
                         train_epoch=50)

    rmse_list = []
    acc_list = []
    mae_list = []

    function.stage_1_train_fast(cluster_data_list=train_loader_list,
                                valid_data_list=valid_loader_list,
                                train_epoch=20)

    for target_area in range(0, 60):
        for index, clusters in enumerate(cluster_lists_sequence):
            if target_area in clusters:
                cluster_target = index
                break

        function.stage_2_train_fast(target_data=train_indivisual_loader_list[target_area],
                                   valid_data=test_indivisual_loader_list[target_area],
                                   cluster_num=args.num_cluster,
                                   cluster_target=cluster_target,
                                   target_num=target_area,
                                   train_epoch=25)

    end_time = time.time()
    run_time.append((end_time - start_time) / 3600.0)

    for target_area in range(60):
        for index, clusters in enumerate(cluster_lists_sequence):
            if target_area in clusters:
                cluster_target = index
                break
        result = function.stage_2_test_fast(test_data=test_indivisual_loader_list[target_area],
                                            cluster_num=args.num_cluster,
                                            cluster_target=cluster_target,
                                            target_num=target_area)
        mae_list.append(result[0].item())
        rmse_list.append(result[1].item())
        acc_list.append(result[2].item())

    test_mse = pd.DataFrame(data=rmse_list)
    test_mse.to_csv(args.root+'/'+str(args.num_cluster)+"/en_de_model_mse_"+str(args.num_cluster)+".csv")
    test_mae = pd.DataFrame(data=mae_list)
    test_mae.to_csv(args.root+'/'+str(args.num_cluster)+"/en_de_model_mae_"+str(args.num_cluster)+".csv")
    test_acc = pd.DataFrame(data=acc_list)
    test_acc.to_csv(args.root+'/'+str(args.num_cluster)+"/en_de_model_acc_"+str(args.num_cluster)+".csv")
    pass

print('Null')
