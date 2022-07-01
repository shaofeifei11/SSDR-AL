import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import glob
from sklearn.neighbors import KDTree
from helper_ply import read_ply
from helper_tool import ConfigS3DIS
from os.path import join
import pickle
from helper_tool import DataProcessing as DP
from torch.utils.data import DataLoader
from base_op import *

class S3DIS_Dataset(Dataset):
    def __init__(self, test_area_idx, sampler_args, round_num, mode, reg_strength):
        """
        :param test_area_idx:
        :param sampler_args:
        :param round_num:
        :param mode:  [training, validation, test, sampling]
        """
        self.sampler_args = sampler_args
        self.round_num = round_num
        self.mode = mode
        self.reg_strength = reg_strength

        self.name = 'S3DIS'
        self.path = './data/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # [0~12]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.input_cloud_names = []
        self.test_labels = []

        # # only used in test
        self.val_proj = []
        self.val_labels = []
        self.sub_grid_size = ConfigS3DIS.sub_grid_size

        self.all_files = glob.glob(os.path.join(self.path, 'input_{:.3f}'.format(self.sub_grid_size), '*.ply'))
        self.val_split = 'Area_' + str(test_area_idx)
        self.tree_path = join(self.path, 'input_{:.3f}'.format(self.sub_grid_size))

        self.init_data()

    def init_data(self):
        for i, file_path in enumerate(self.all_files):
            cloud_name = file_path.split('/')[-1][:-4]
            if (self.mode == "training" or self.mode == "sampling") and self.val_split not in cloud_name:
                self.input_cloud_names += [cloud_name]
            elif (self.mode == "validation" or self.mode == "test") and self.val_split in cloud_name:
                self.input_cloud_names += [cloud_name]
                if self.mode == "test":
                    sub_ply_file = join(self.tree_path, '{:s}.ply'.format(cloud_name))
                    data = read_ply(sub_ply_file)
                    sub_label = data['class']  # shape=[point_number]
                    self.test_labels += [sub_label]

        for i, file_path in enumerate(self.all_files):
            cloud_name = file_path.split('/')[-1][:-4]
            # Validation projection and labels
            if self.mode == "test" and self.val_split in cloud_name:
                proj_file = join(self.tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

    def load_data(self, cloud_name):
        # Name of the input files
        kd_tree_file = join(self.tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
        sub_ply_file = join(self.tree_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        sub_color = np.vstack((data['red'], data['green'], data['blue'])).T  # shape=[point_number, 3]
        sub_label = data['class']  # shape=[point_number]
        # Read pkl with search tree
        with open(kd_tree_file, 'rb') as f:
            search_tree = pickle.load(f)
        if (self.mode == "training" or self.mode == "sampling") and self.val_split not in cloud_name:
            # pseudo gt
            pseudo_gt_path = join(self.path, str(self.reg_strength), "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(self.round_num), cloud_name + ".gt")
            with open(pseudo_gt_path, "rb") as f:
                pseudo_gt = pickle.load(f)
                pseudo_gt = np.asarray(pseudo_gt)
            input_activation = pseudo_gt[0]
            input_pseudo = pseudo_gt[1]
        elif (self.mode == "validation" or self.mode == "test") and self.val_split in cloud_name:
            input_activation = np.ones([len(sub_label)])
            input_pseudo = sub_label

        input_tree = search_tree
        input_color = sub_color
        input_label = sub_label

        return input_activation, input_pseudo, input_tree, input_color, input_label

    def spatially_regular_gen(self, input_activation, input_pseudo, input_tree, input_color, input_label, cloud_idx):
        # Get all points within the cloud from tree structure
        points = np.array(input_tree.data, copy=False)
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.random.randint(low=0, high=points.shape[0])

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=ConfigS3DIS.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if self.mode == "sampling" or len(points) < ConfigS3DIS.num_points:
            # Query all points within the cloud
            queried_idx = input_tree.query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = input_tree.query(pick_point, k=ConfigS3DIS.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_color = input_color[queried_idx]
        queried_pc_label = input_label[queried_idx]
        queried_pc_activation = input_activation[queried_idx]
        queried_pc_pseudo = input_pseudo[queried_idx]

        # up_sampled with replacement
        if len(points) < ConfigS3DIS.num_points:
            queried_pc_xyz, queried_pc_color, queried_idx, queried_pc_label, queried_pc_activation, queried_pc_pseudo = \
                DP.data_aug(queried_pc_xyz, queried_pc_color, queried_pc_label, queried_pc_activation,
                            queried_pc_pseudo, queried_idx, ConfigS3DIS.num_points)

        return np.asarray([queried_pc_xyz]), np.asarray([queried_pc_color]), np.asarray([queried_pc_label]), \
               np.asarray([queried_pc_activation]), np.asarray([queried_pc_pseudo]), np.asarray([queried_idx]), \
               np.array([cloud_idx], dtype=np.int32)

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx):

        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigS3DIS.num_layers):
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, ConfigS3DIS.k_n)
            sub_points = batch_xyz[:, :np.shape(batch_xyz)[1] // ConfigS3DIS.sub_sampling_ratio[i],
                         :]
            pool_i = neighbour_idx[:, :np.shape(batch_xyz)[1] // ConfigS3DIS.sub_sampling_ratio[i],
                     :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)

            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            batch_xyz = sub_points


        input_list = input_points + input_neighbors + input_pools + input_up_samples  # shape[0] = 5+5+5+5
        input_list += [batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx]  # shape[0] = 20+6

        return input_list

    def __len__(self):
        return len(self.input_cloud_names)

    def __getitem__(self, idx):
        cloud_name = self.input_cloud_names[idx]
        input_activation, input_pseudo, input_tree, input_color, input_label = self.load_data(cloud_name)
        batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx = self.spatially_regular_gen(input_activation, input_pseudo, input_tree, input_color, input_label, idx)
        input_list = self.tf_map(batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx)
        return input_list