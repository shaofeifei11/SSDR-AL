
import os.path
import argparse
from graphs import compute_graph_nn_2
# from provider import *
from helper_ply import read_ply
import glob
import pickle
import os
import numpy as np
import sys
sys.path.append("partition/cut-pursuit/build/src")
sys.path.append("cut-pursuit/build/src")
sys.path.append("ply_c")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c

def s3dis_superpoint(args, val_split):
    path = "data/S3DIS"
    all_files = glob.glob(os.path.join(path, 'original_ply', '*.ply'))

    output_dir = os.path.join(path, str(args.reg_strength), "superpoint")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tree_path = os.path.join(path, 'input_{:.3f}'.format(0.04))

    total_obj = {}
    total_obj["unlabeled"] = {}
    sp_num = 0
    file_num = 0
    point_num = 0

    for i, file_path in enumerate(all_files):
        print(file_path)
        cloud_name = file_path.split('/')[-1][:-4]
        if val_split not in cloud_name:
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            xyz = np.vstack((data['x'], data['y'], data['z'])).T
            # xyz = xyz.astype('f4')
            # rgb = rgb.astype('uint8')
            # ---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            # ---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype(
                'float32')
            del target_fea
            # --compute the partition------
            # --- build the spg h5 file --
            features = np.hstack((geof, rgb)).astype('float32')  # add rgb as a feature for partitioning
            features[:, 3] = 2. * features[:, 3]  # increase importance of verticality (heuristic)

            graph_nn["edge_weight"] = np.array(
                1. / (args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])),
                dtype='float32')
            print("minimal partition...")


            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                                        , graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype='object')
            sp = {}
            sp["components"] = components
            sp["in_component"] = in_component
            with open(os.path.join(output_dir, cloud_name+".superpoint"),"wb") as f:
                pickle.dump(sp, f)

            pseudo_gt = np.zeros([2, len(xyz)], dtype=np.float32)
            with open(os.path.join(output_dir, cloud_name+".gt"), "wb") as f:
                pickle.dump(pseudo_gt, f)

            sp_num = sp_num + len(components)
            file_num = file_num + 1
            point_num = point_num + len(xyz)

            total_obj["unlabeled"][cloud_name] = np.arange(len(components))

    total_obj["file_num"] = file_num
    total_obj["sp_num"] = sp_num
    total_obj["point_num"] = point_num

    with open(os.path.join(output_dir, "total.pkl"), "wb") as f:
        pickle.dump(total_obj, f)

    print("file_num", file_num, "sp_num", sp_num, "point_num", point_num)


def test_superpoint_distribution(args):
    all_files = glob.glob(os.path.join('data/S3DIS', str(args.reg_strength), 'superpoint', '*.superpoint'))
    sp_count = 0
    point_count = 0

    dis = np.zeros([10000])

    for i, file_path in enumerate(all_files):
        with open(file_path, "rb") as f:
            superpoint = pickle.load(f)
        components = superpoint["components"]
        sp_count = sp_count + len(components)
        for sp in components:
            sp_size = len(sp)
            point_count = point_count + sp_size
            tt = int(sp_size / 10)
            dis[tt] = dis[tt] + 1

    mean_size = point_count / sp_count
    print("######### test_superpoint_less_than_5")
    for i in range(len(dis)):
        if dis[i] > 0:
            print(str(i*10)+"-"+str((i+1)*10)+": " + str(dis[i]))
    print("point_count=" + str(point_count), "sp_count=" + str(sp_count), "mean_size=" + str(mean_size))
    print("#####################################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='123434234')
    parser.add_argument('--dataset', default='s3dis', help='s3dis/sema3d/your_dataset')
    parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
    parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
    parser.add_argument('--lambda_edge_weight', default=1., type=float,
                        help='parameter determine the edge weight for minimal part.')
    parser.add_argument('--reg_strength', default=0.008, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    args = parser.parse_args()
    s3dis_superpoint(args, "Area_5")
    test_superpoint_distribution(args)