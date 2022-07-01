import argparse
import math

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='S3DIS', choices=["S3DIS", "semantic3d"])
    parser.add_argument('--seed_percent', type=float, default=0.01, help='seed percent')
    parser.add_argument('--reg_strength', default=0.008, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='tolerance threshold')

    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'

    sampler_args = []
    sampler_args.append("max")
    sampler_args.append("dominant")

    dataset_name = FLAGS.dataset
    reg_strength = FLAGS.reg_strength
    threshold = FLAGS.threshold
    round_num = 2

    if dataset_name == "S3DIS":
        test_area_idx = 5
        input_ = "input_0.040"
        cfg = ConfigS3DIS

    round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) + "_" + str(reg_strength) + '.txt'), 'a')

    with open(os.path.join("data", dataset_name, str(reg_strength), "superpoint/total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
    total_sp_num = total_obj["sp_num"]

    print("total_sp_num", total_sp_num)
    Sampler = AllSampler("data/" + dataset_name + "/" + input_, "data/" + dataset_name+ "/" + str(reg_strength), total_sp_num, sampler_args)

    w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0,  "sub_p_num": 0}
    Sampler.sampling(None, total_sp_num, last_round=round_num - 1, w=w, threshold=threshold)
    labeling_region_num = w["sp_num"] + w["sub_num"]
    labeling_point_num = w["p_num"] + w["sub_p_num"]
    log_out("round= " + str(round_num) + " |                    labeling_region_num=" + str(
            labeling_region_num) + ", labeling_point_num=" +
                str(labeling_point_num) + ", mean_points=" + str(labeling_point_num / labeling_region_num),
                round_result_file)

    model = Network(cfg, dataset_name, sampler_args, test_area_idx, reg_strength=reg_strength)
    best_miou, best_OA = model.train(round_num=round_num)

    log_out("round= " + str(round_num) + " | best_miou= " + str(best_miou) + ", best_OA= " + str(best_OA), round_result_file)

    model.close()
    round_result_file.close()