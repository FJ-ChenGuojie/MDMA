import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
sys.setrecursionlimit(5000)
from params import args
from model import En_DeModel

def test(f_dir_list, c_initer_dir, test_data):
    num_container = 0
    acc = 0
    f_model_list = []
    for i in range(len(f_dir_list)):
        f_model = En_DeModel.En_DeModel_3(n_input=[args.n_dim, 4],
                                          layer=[4, 3],
                                          n_embedding=128,
                                          n_hidden=1024,
                                          dropout_rate=0.1,
                                          len_seq=5,
                                          device=args.device).to(args.device)
        f_model.load_state_dict(torch.load(f_dir_list[i]))
        f_model_list.append(f_model)

    c_model = En_DeModel.Classfier_inte(n_hidden=1024, num_cluster=args.num_cluster).to(args.device)
    c_model.load_state_dict(torch.load(c_initer_dir))

    for model_ind in range(len(f_model_list) - 1):
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

        if i % len(test_data) == 0:
            iter_test_data = iter(test_data)

        with torch.no_grad():
            feature_list = []
            feature_list.clear()
            for model_ind in range(len(f_model_list)):
                feature, _ = f_model_list[model_ind](batch_en_in, batch_en_ts, batch_de_in, batch_de_ts, batch_label_mask)
                feature_list.append(feature)

            feature_inte = torch.cat(feature_list, dim=1)
            output = c_model(feature_inte)

            num_container += len(output)
            pred_cls = output.data.max(1)[1]
            acc += pred_cls.eq(batch_label.data).cpu().sum()

        sys.stdout.write('\r stage_2--->test--->[iter: %d / all %d]           ' % (i + 1, len(test_data)))
        sys.stdout.flush()

    print("Accuracy = %.4f \n" % (acc / num_container))