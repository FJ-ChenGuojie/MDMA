import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import dataprocess
import function
from params import args
import datetime
import numpy as np
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
path = 'F:/partData/'+now_time+args.model_dir
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.num_cluster = 4
args.data_dir = 'F:/partData/container storage sequence'
feat, time, location, label, label_mask = dataprocess.data(args.data_dir)
feat_list = []
time_list = []
location_list = []
label_list = []
label_mask_list = []
feat_list.append(feat)
time_list.append(time)
location_list.append(location)
label_list.append(label)
label_mask_list.append(label_mask)
en_feat, en_ts, en_location, de_feat, de_ts, de_location, label_convert, mask_convert = dataprocess.data_covert(feat_list,
                                                                                                                                                        time_list,
                                                                                                                                                        location_list,
                                                                                                                                                        label_list,
                                                                                                                                                        label_mask_list)

loader = dataprocess.data_convert_loader(en_feat, en_ts, en_location,
                                           de_feat, de_ts, de_location, label_convert, mask_convert)
args.root = 'F:/partData/restored_MDMA_master/'
f_model_list = []
for i in range(1, 2):
    f_model_list.clear()
    f1_dir = args.root + 'f_finetune_0_.pkl'
    f2_dir = args.root + 'f_finetune_1_.pkl'
    f3_dir = args.root + 'f_finetune_2_.pkl'
    c_initer_dir = args.root + 'c_finetune_inter.pkl'
    f_model_list.append(f1_dir)
    f_model_list.append(f2_dir)
    f_model_list.append(f3_dir)
    function.test(f_model_list, c_initer_dir, loader)
    pass