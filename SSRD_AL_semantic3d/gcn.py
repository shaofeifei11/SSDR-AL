import time

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import torch.optim as optim
from kcenterGreedy import *
from os.path import join
import pickle
from helper_ply import read_ply
from sklearn.neighbors import KDTree

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, gcn_gpu=1):
        super(GraphConvolution, self).__init__()
        self.gcn_gpu = gcn_gpu
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).cuda(device=self.gcn_gpu))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).cuda(device=self.gcn_gpu))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
         A * (V * W) + b:   [N, N] * ([N, m] * [m, hidden]) + [hidden]  => [N, hidden]
        Args:
            input:  V
            adj:    A

        Returns:

        """
        support = torch.mm(input, self.weight).cuda(device=self.gcn_gpu)
        output = torch.spmm(adj, support).cuda(device=self.gcn_gpu)
        if self.bias is not None:
            return (output + self.bias).cuda(device=self.gcn_gpu)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    A * ((A * (V * W1) + b1) * W2) + b2
    """

    def __init__(self, nfeat, nhid, nclass, dropout, gcn_gpu):
        super(GCN, self).__init__()
        self.gcn_gpu = gcn_gpu
        self.gc1 = GraphConvolution(nfeat, nhid).cuda(device=gcn_gpu)
        self.gc2 = GraphConvolution(nhid, nhid).cuda(device=gcn_gpu)
        self.gc3 = GraphConvolution(nhid, nclass).cuda(device=gcn_gpu)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1).cuda(device=gcn_gpu)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)).cuda(device=self.gcn_gpu)
        feat = F.dropout(x, self.dropout, training=self.training).cuda(device=self.gcn_gpu)
        x = self.gc3(feat, adj)

        return torch.sigmoid(x), feat, torch.cat((feat,x),1).cuda(device=self.gcn_gpu)

def BCEAdjLoss(scores, lbl, nlbl, l_adj, gcn_gpu):
    lnl = torch.log(scores[lbl]).cuda(device=gcn_gpu)
    lnu = torch.log(1 - scores[nlbl]).cuda(device=gcn_gpu)
    labeled_score = torch.mean(lnl).cuda(device=gcn_gpu)
    unlabeled_score = torch.mean(lnu).cuda(device=gcn_gpu)
    bce_adj_loss = (-labeled_score - l_adj*unlabeled_score).cuda(device=gcn_gpu)
    return bce_adj_loss

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
        # 中心对齐
        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)
        tree_list.append(KDTree(align_superpoint))
    for i in range(sp_num):
        cd_dist[i] = chamfer_distance(align_superpoint_list, tree_list, i)
    return cd_dist

def create_adj(featuresV, labeled_select_ref, unlabeled_candidate_ref, input_path, data_path, gcn_gpu):
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
        print("3", i)
        cloud_name = unlabeled_candidate_ref[i]["cloud_name"]
        sp_idx = unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": i})
    print("4")
    for i in range(labeled_num):
        print("4", i)
        cloud_name = labeled_select_ref[i]["cloud_name"]
        sp_idx = labeled_select_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": unlabeled_num+i})

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
        xyz = np.vstack((data['x'], data['y'], data['z'])).T

        source_ref_idx_list = []
        one_cloud_candicate_superpoints = []
        one_cloud_center_xyz = np.zeros([len(total_cloud[cloud_name]), 3])
        one_cloud_center_xyz_len = len(one_cloud_center_xyz)
        for j in range(one_cloud_center_xyz_len):
            print(cloud_name_list_len, i, "f1", one_cloud_center_xyz_len, j)
            source_sp_idx = total_cloud[cloud_name][j]["sp_idx"]
            source_ref_idx_list.append(total_cloud[cloud_name][j]["ref_idx"])

            x_y_z = xyz[components[source_sp_idx]]
            one_cloud_center_xyz[j, 0] = (np.min(x_y_z[:, 0]) + np.max(x_y_z[:, 0])) / 2.0
            one_cloud_center_xyz[j, 1] = (np.min(x_y_z[:, 1]) + np.max(x_y_z[:, 1])) / 2.0
            one_cloud_center_xyz[j, 2] = (np.min(x_y_z[:, 2]) + np.max(x_y_z[:, 2])) / 2.0
            one_cloud_candicate_superpoints.append(x_y_z)

        print("5")

        one_clound_cd_dist = create_cd(superpoint_list=one_cloud_candicate_superpoints,
                                       superpoint_centroid_list=one_cloud_center_xyz)
        for j in range(one_cloud_center_xyz_len):
            print(cloud_name_list_len, i, "f2", one_cloud_center_xyz_len, j)
            ssdr = one_cloud_center_xyz - one_cloud_center_xyz[j]
            dist = np.sqrt(np.sum(np.multiply(ssdr, ssdr), axis=1))
            A_ed[source_ref_idx_list[j], source_ref_idx_list] = dist
            A_cd[source_ref_idx_list[j], source_ref_idx_list] = one_clound_cd_dist[j]
        print("6")

    print("tensor", "1")
    featuresV = torch.nn.functional.normalize(torch.Tensor(featuresV).cuda(device=gcn_gpu))
    print("tensor", "2")
    A_latent = torch.mm(featuresV, featuresV.t()).cuda(device=gcn_gpu)
    print("tensor", "3")
    adj = torch.multiply(A_latent, torch.exp(-torch.add(torch.Tensor(A_ed).cuda(device=gcn_gpu), torch.Tensor(A_cd).cuda(device=gcn_gpu))))
    print("tensor", "4")
    adj += -1.0 * torch.eye(adj.shape[0]).cuda(device=gcn_gpu)  # S-I
    print("tensor", "5")
    adj_diag = torch.sum(adj, dim=0).cuda(device=gcn_gpu)  # rowise sum
    print("tensor", "6")
    adj = torch.mm(adj, torch.diag(1 / adj_diag)).cuda(device=gcn_gpu)
    print("tensor", "7")
    adj = adj + torch.eye(adj.shape[0]).cuda(device=gcn_gpu)  #
    print("tensor", "8")
    return featuresV, adj, time.time() - begin_time

def GCN_sampling(labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref, input_path, data_path, sampling_batch, gcn_gpu, coreGCN=True):
    """
        labeled(1),
        unlabeled(0)
    """
    # torch.cuda.set_device()

    featuresV = np.concatenate([unlabeled_candidate_features, labeled_select_features])
    nfeat = featuresV.shape[1]
    unlabeld_num = len(unlabeled_candidate_features)
    labeled_num = len(labeled_select_features)
    print("begin compute create_adj.")
    featuresV, adj_tensor, cost_time = create_adj(featuresV, labeled_select_ref, unlabeled_candidate_ref, input_path, data_path, gcn_gpu=gcn_gpu)
    print("\n############\n compute gcn adj successfully. CostTime = "+str(cost_time)+" s \n###############\n")

    gcn_module = GCN(nfeat=nfeat,
                     nhid=128,
                     nclass=1,
                     dropout=0.3,
                     gcn_gpu=gcn_gpu).cuda(device=gcn_gpu)

    optimizer = optim.Adam(gcn_module.parameters(), lr=1e-3, weight_decay=5e-4)

    lbl = np.arange(unlabeld_num, unlabeld_num+labeled_num, 1)
    nlbl = np.arange(0, unlabeld_num, 1)

    ############
    for bb in range(2000):
        print("training gcn ", bb)
        optimizer.zero_grad()
        outputs, _, _ = gcn_module(featuresV, adj_tensor)
        lamda = 1.2
        loss = BCEAdjLoss(outputs, lbl, nlbl, lamda, gcn_gpu=gcn_gpu)
        loss.backward()
        optimizer.step()

    print("\n############\n compute gcn train successfully \n###############\n")

    gcn_module.eval()
    with torch.no_grad():
        scores, _, feat = gcn_module(featuresV, adj_tensor)
        print("\n############\n compute gcn eval successfully \n###############\n")

    feat = feat.detach().cpu().numpy()

    feat = feat.astype(np.float64)

    where_are_nan = np.isnan(feat)
    where_are_inf = np.isinf(feat)

    feat[where_are_nan] = 1.0 * 1e-10
    feat[where_are_inf] = 1.0 * 1e10

    already_selected = np.arange(unlabeld_num, unlabeld_num + labeled_num)
    sampling22222222222 = kCenterGreedy(feat)
    sampling_index = sampling22222222222.select_batch_(already_selected, sampling_batch)


    file_list = {}
    for i in sampling_index:
        cloud_name, sp_idx = unlabeled_candidate_ref[i]["cloud_name"], unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in file_list:
            file_list[cloud_name] = []
        file_list[cloud_name].append(sp_idx)
    return file_list
