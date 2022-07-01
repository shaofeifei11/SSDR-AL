import pickle
import time
from os.path import join

import numpy as np
from sklearn.neighbors import KDTree

from helper_ply import read_ply
from kcenterGreedy import *


def chamfer_distance(cloud_list, tree_list, centroid_idx):
    """numpy"""
    centroid_cloud = cloud_list[centroid_idx]
    centroid_tree = tree_list[centroid_idx]
    distances = np.zeros([len(cloud_list)])
    for i in range(len(cloud_list)):
        if i != centroid_idx:
            distances1, _ = centroid_tree.query(cloud_list[i])
            distances2, _ = tree_list[i].query(centroid_cloud)
            av_dist1 = np.mean(distances1)
            av_dist2 = np.mean(distances2)
            distances[i] = av_dist1 + av_dist2
    return distances

def create_cd(superpoint_list, superpoint_centroid_list):
    """numpy"""
    sp_num = len(superpoint_list)
    cd_dist = np.zeros([sp_num, sp_num])
    align_superpoint_list = []
    tree_list = []
    for i in range(sp_num):
        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)
        tree_list.append(KDTree(align_superpoint))
    for i in range(sp_num):
        cd_dist[i] = chamfer_distance(align_superpoint_list, tree_list, i)
    return cd_dist

def fps_adj_all(labeled_select_ref, unlabeled_candidate_ref, input_path, data_path):
    begin_time = time.time()

    unlabeled_num = len(unlabeled_candidate_ref)
    labeled_num = len(labeled_select_ref)
    N = unlabeled_num + labeled_num
    total_cloud = {}  # {cloud_name: [{sp_idx, ref_idx}]}
    cloud_name_list = []
    for i in range(unlabeled_num):
        cloud_name = unlabeled_candidate_ref[i]["cloud_name"]
        sp_idx = unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": i})
    for i in range(labeled_num):
        cloud_name = labeled_select_ref[i]["cloud_name"]
        sp_idx = labeled_select_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": unlabeled_num + i})

    # print("ed,cd below")
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
        for j in range(one_cloud_center_xyz_len):
            # print(cloud_name_list_len, i, "f1", one_cloud_center_xyz_len, j)
            source_sp_idx = total_cloud[cloud_name][j]["sp_idx"]
            source_ref_idx_list.append(total_cloud[cloud_name][j]["ref_idx"])

            x_y_z = xyz[components[source_sp_idx]]
            one_cloud_center_xyz[j, 0] = (np.min(x_y_z[:, 0]) + np.max(x_y_z[:, 0])) / 2.0
            one_cloud_center_xyz[j, 1] = (np.min(x_y_z[:, 1]) + np.max(x_y_z[:, 1])) / 2.0
            one_cloud_center_xyz[j, 2] = (np.min(x_y_z[:, 2]) + np.max(x_y_z[:, 2])) / 2.0
            one_cloud_candicate_superpoints.append(x_y_z)

        one_clound_cd_dist = create_cd(superpoint_list=one_cloud_candicate_superpoints,
                                       superpoint_centroid_list=one_cloud_center_xyz)
        for j in range(one_cloud_center_xyz_len):
            # print(cloud_name_list_len, i, "f2", one_cloud_center_xyz_len, j)
            ssdr = one_cloud_center_xyz - one_cloud_center_xyz[j]
            dist = np.sqrt(np.sum(np.multiply(ssdr, ssdr), axis=1))
            A_ed[source_ref_idx_list[j], source_ref_idx_list] = dist
            A_cd[source_ref_idx_list[j], source_ref_idx_list] = one_clound_cd_dist[j]

    # print("tensor", 3)
    adj = np.exp(-np.add(A_ed, A_cd))
    # print("tensor", 4)
    adj += -1.0 * np.eye(adj.shape[0])  # S-I
    # print("tensor", 5)
    adj_diag = np.sum(adj, axis=1)  # rowise sum

    d_inv = np.power(adj_diag, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)

    # print("tensor", 6)
    adj = np.matmul(adj, d_mat_inv)
    # print("tensor", 7)
    adj = adj + np.eye(adj.shape[0])  # D^(-1)(S-I) + I
    # print("tensor", 8)
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


def GCN_FPS_sampling(labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref, input_path, data_path, sampling_batch, gcn_number, gcn_top):
    adj, _ = fps_adj_all(labeled_select_ref=labeled_select_ref, unlabeled_candidate_ref=unlabeled_candidate_ref, input_path=input_path, data_path=data_path)

    if gcn_top > 0:

        gcn_top = int(gcn_top)
        mask = np.zeros(adj.shape)
        source_idx = np.repeat(np.expand_dims(np.arange(adj.shape[0]), axis=1), repeats=gcn_top, axis=1)
        arg_idx = np.argsort(adj, axis=1)[:, -gcn_top:]
        mask[source_idx, arg_idx] = 1.0
        adj = np.multiply(adj, mask)

    featuresV = np.concatenate([unlabeled_candidate_features, labeled_select_features])
    featureV_list = [featuresV]
    for i in range(int(gcn_number)):
        featuresV = np.matmul(adj, featuresV)
        featureV_list.append(featuresV)
    combinational_features = np.sum(featureV_list, axis=0)

    unlabeled_num = len(unlabeled_candidate_features)
    selected_ids = farthest_features_sample(combinational_features[:unlabeled_num], sampling_batch)

    file_list = {}
    for i in selected_ids:
        cloud_name, sp_idx = unlabeled_candidate_ref[i]["cloud_name"], unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in file_list:
            file_list[cloud_name] = []
        file_list[cloud_name].append(sp_idx)
    return file_list

