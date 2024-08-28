import pickle
import time
from os.path import join
import torch, chamfer3D.dist_chamfer_3D
import numpy as np
from sklearn.neighbors import KDTree

from helper_ply import read_ply
from kcenterGreedy import *

fps_gpu = 0

def create_cd_cuda(superpoint_list, superpoint_centroid_list, chamLoss):
    sp_num = len(superpoint_list)
    cd_dist = np.zeros([sp_num, sp_num])
    align_superpoint_list = []
    for i in range(sp_num):

        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)

    for i in range(len(align_superpoint_list)-1):
        for j in range(i+1, len(align_superpoint_list)):
            point1 = torch.Tensor(np.asarray(align_superpoint_list[i:i+1])).cuda(device=fps_gpu)
            point2 = torch.Tensor(np.asarray(align_superpoint_list[j:j+1])).cuda(device=fps_gpu)
            dist1, dist2, _, _ = chamLoss(point1, point2)
            ss = torch.add(torch.mean(torch.sqrt(dist1)), torch.mean(torch.sqrt(dist2))).cuda(device=fps_gpu)
            cd_dist[i, j] = ss.detach().cpu()
            cd_dist[j, i] = cd_dist[i, j]
    return cd_dist

def fps_adj_all(labeled_select_ref, unlabeled_candidate_ref, input_path, data_path):
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist().cuda(device=fps_gpu)
    begin_time = time.time()
    print("1")
    unlabeled_num = len(unlabeled_candidate_ref)
    print("2")
    labeled_num = len(labeled_select_ref)
    print("3")
    N = unlabeled_num + labeled_num
    total_cloud = {}  # {cloud_name: [{sp_idx, ref_idx}]}
    cloud_name_list = []
    for i in range(unlabeled_num):
        # print("3", i)
        cloud_name = unlabeled_candidate_ref[i]["cloud_name"]
        sp_idx = unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": i})
    print("4")
    for i in range(labeled_num):
        # print("4", i)
        cloud_name = labeled_select_ref[i]["cloud_name"]
        sp_idx = labeled_select_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": unlabeled_num + i})

    print("ed,cd below")
    A_ed = np.ones([N, N], dtype=np.float) * 1e10
    A_cd = np.ones([N, N], dtype=np.float) * 1e10
    cloud_name_list_len = len(cloud_name_list)
    for i in range(cloud_name_list_len):
        cloud_name = cloud_name_list[i]
        with open(join(data_path, "superpoint",
                       cloud_name + ".superpoint"), "rb") as f:
            sp = pickle.load(f)
        components = sp["components"]
        data = read_ply(
            join(input_path, '{:s}.ply'.format(cloud_name)))
        xyz = np.vstack((data['x'], data['y'], data['z'])).T  # shape=[point_number, 3]

        source_ref_idx_list = []
        one_cloud_candicate_superpoints = []
        one_cloud_center_xyz = np.zeros([len(total_cloud[cloud_name]), 3])
        one_cloud_center_xyz_len = len(one_cloud_center_xyz)
        print("5")
        point_size = 0
        for j in range(one_cloud_center_xyz_len):
            # print("5", j)
            # print(cloud_name_list_len, i, "f1", one_cloud_center_xyz_len, j)
            source_sp_idx = total_cloud[cloud_name][j]["sp_idx"]
            source_ref_idx_list.append(total_cloud[cloud_name][j]["ref_idx"])

            x_y_z = xyz[components[source_sp_idx]]
            one_cloud_center_xyz[j] = (np.min(x_y_z, axis=0) + np.max(x_y_z, axis=0)) / 2.0
            one_cloud_candicate_superpoints.append(x_y_z)

            point_size += len(components[source_sp_idx])

        print("6")
        print("xyz_len", one_cloud_center_xyz_len, "point_size", point_size)
        one_clound_cd_dist = create_cd_cuda(superpoint_list=one_cloud_candicate_superpoints,
                                       superpoint_centroid_list=one_cloud_center_xyz, chamLoss=chamLoss)
        print("7")
        for j in range(one_cloud_center_xyz_len):
            # print("6", j)
            # print(cloud_name_list_len, i, "f2", one_cloud_center_xyz_len, j)
            ssdr = one_cloud_center_xyz - one_cloud_center_xyz[j]
            dist = np.sqrt(np.sum(np.multiply(ssdr, ssdr), axis=1))
            A_ed[source_ref_idx_list[j], source_ref_idx_list] = dist
            A_cd[source_ref_idx_list[j], source_ref_idx_list] = one_clound_cd_dist[j]

    print("tensor", 3)
    adj = np.exp(-np.add(A_ed, A_cd))
    print("tensor", 4)
    adj += -1.0 * np.eye(adj.shape[0])  # S-I
    # print("tensor", 5)
    adj_diag = np.sum(adj, axis=1)  # rowise sum

    d_inv = np.power(adj_diag, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    # print("tensor", 6)
    adj = np.matmul(adj, d_mat_inv)
    print("tensor", 7)
    adj = adj + np.eye(adj.shape[0])
    print("tensor", 8)
    return adj, time.time() - begin_time

def farthest_features_sample(feature_list, sample_number):
    """
    Input:
        superpoint_list: pointcloud data, [sp_num, each_sp_p_num, 3]
        superpoint_centroid_list: pointcloud centroid xyz [sp_num, 3]
        sample_number: number of samples
    Return:
        centroids: sampled superpoint index, [sample_number]
    """
    list_num = len(feature_list)
    feature_list = np.array(feature_list)


    centroids = np.zeros([sample_number], dtype=np.int32)
    centroids[0] = np.random.randint(0, list_num)

    distance = np.ones([list_num]) * 1e10

    for i in range(sample_number - 1):

        current_superpoint_center = feature_list[centroids[i]]
        dist = np.sum((feature_list - current_superpoint_center) ** 2, axis=-1)


        mask = dist < distance
        distance[mask] = dist[mask]

        centroids[i + 1] = np.argmax(distance)
    return centroids


def GCN_FPS_sampling(labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref, input_path, data_path, sampling_batch):
    begin_time = time.time()
    adj, _ = fps_adj_all(labeled_select_ref=labeled_select_ref, unlabeled_candidate_ref=unlabeled_candidate_ref, input_path=input_path, data_path=data_path)
    print("-1", time.time() - begin_time)
    featuresV = np.concatenate([unlabeled_candidate_features, labeled_select_features])
    print("-2", time.time() - begin_time)
    combinational_features = np.add(np.matmul(adj, featuresV), featuresV)
    print("-3", time.time() - begin_time)

    unlabeled_num = len(unlabeled_candidate_features)
    selected_ids = farthest_features_sample(combinational_features[:unlabeled_num], sampling_batch)
    print("-4", time.time() - begin_time)

    file_list = {}
    for i in selected_ids:
        print("-4", i, time.time() - begin_time)
        cloud_name, sp_idx = unlabeled_candidate_ref[i]["cloud_name"], unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in file_list:
            file_list[cloud_name] = []
        file_list[cloud_name].append(sp_idx)
    return file_list

