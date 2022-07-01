import argparse

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    """create seed samples and model weights"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='semantic3d', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--reg_strength', default=0.012, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--lr_decay', default=0.92, type=float)

    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'



    dataset_name = FLAGS.dataset
    reg_strength = FLAGS.reg_strength
    round_num = 1
    epoch = FLAGS.epoch
    lr_decay = FLAGS.lr_decay

    sampler_args = []
    sampler_args.append("baseline")
    sampler_args.append(str(epoch))
    sampler_args.append(str(lr_decay))

    if dataset_name == "semantic3d":
        test_area_idx = 0
        input_ = "input_0.060"
        cfg = ConfigSemantic3D
        cfg.max_epoch = epoch
        cfg.lr_decays = {i: lr_decay for i in range(0, 500)}


    round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) + "_" + str(reg_strength) + '.txt'), 'a')

    with open(os.path.join("data", dataset_name, str(reg_strength), "superpoint/total.pkl"), "rb") as f:
        total_obj = pickle.load(f)
    total_sp_num = total_obj["sp_num"]

    print("total_sp_num", total_sp_num)
    Sampler = SeedSampler("data/" +dataset_name + "/" + input_, "data/" + dataset_name + "/" + str(reg_strength), total_sp_num, sampler_args)

    w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0, "sub_p_num": 0}
    sp_batch_size = total_sp_num
    Sampler.sampling(None, sp_batch_size, last_round=round_num - 1, w=w)
    labeling_region_num = w["sp_num"] + w["sub_num"]
    labeling_point_num = w["p_num"] + w["sub_p_num"]
    log_out("round= " + str(round_num) + " |                    labeling_region_num=" + str(
            labeling_region_num) + ", labeling_point_num=" +
                str(labeling_point_num) + ", mean_points=" + str(labeling_point_num / labeling_region_num),
                round_result_file)

    model = Network(cfg, dataset_name, sampler_args, test_area_idx, reg_strength=reg_strength)
    best_miou, best_OA = model.train2(round_num=round_num)

    log_out("round= " + str(round_num) + " | best_miou= " + str(best_miou) + ", best_OA= " + str(best_OA), round_result_file)

    model.close()
    round_result_file.close()
