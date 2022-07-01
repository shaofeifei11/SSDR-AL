import os
import torch
from os.path import join, exists
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import glob
from sklearn.neighbors import KDTree
from helper_ply import read_ply
from helper_tool import ConfigSemantic3D
from os.path import join
import pickle
from helper_tool import DataProcessing as DP
from torch.utils.data import DataLoader
from base_op import *

train_cloud_name_list = ['bildstein_station1_xyz_intensity_rgb',
                             'bildstein_station5_xyz_intensity_rgb',
                             'domfountain_station1_xyz_intensity_rgb',
                             'domfountain_station2_xyz_intensity_rgb',
                             'domfountain_station3_xyz_intensity_rgb',
                             'neugasse_station1_xyz_intensity_rgb',
                             'sg27_station1_intensity_rgb',
                             'sg27_station4_intensity_rgb',
                             'sg27_station5_intensity_rgb',
                             'sg27_station9_intensity_rgb',
                             'sg28_station4_intensity_rgb',
                             'untermaederbrunnen_station1_xyz_intensity_rgb',
                             'untermaederbrunnen_station3_xyz_intensity_rgb']

val_cloud_name_list = ['bildstein_station3_xyz_intensity_rgb',
                       'sg27_station2_intensity_rgb']

test_cloud_name_list = ['MarketplaceFeldkirch_Station4_rgb_intensity-reduced',
                        'sg27_station10_rgb_intensity-reduced',
                        'sg28_Station2_rgb_intensity-reduced',
                        'StGallenCathedral_station6_rgb_intensity-reduced']

class Semantic3D_Dataset_Sampling(Dataset):
    def __init__(self, sampler_args, round_num, mode, reg_strength):
        self.sampler_args = sampler_args
        self.round_num = round_num
        self.mode = mode
        self.reg_strength = reg_strength

        self.name = 'semantic3d'
        self.path = './data/semantic3d/'


        self.label_to_names = {0: 'man-made terrain',
                               1: 'natural terrain',
                               2: 'high vegetation',
                               3: 'low vegetation',
                               4: 'buildings',
                               5: 'hard scape',
                               6: 'scanning artefacts',
                               7: 'cars'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(ConfigSemantic3D.sub_grid_size))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.class_weight = []

        self.input_cloud_names = []
        self.sub_grid_size = ConfigSemantic3D.sub_grid_size
        self.tree_path = join(self.path, 'input_{:.3f}'.format(self.sub_grid_size))
        self.init_data()

    def init_data(self):
        if self.mode == "training" or self.mode == "sampling" :
            self.input_cloud_names = train_cloud_name_list

    def load_data(self, cloud_name):
        # Name of the input files
        kd_tree_file = join(self.tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
        sub_ply_file = join(self.tree_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        sub_color = np.vstack((data['red'], data['green'], data['blue'])).T  # shape=[point_number, 3]

        if self.mode == 'test':
            sub_label = None
        else:
            sub_label = data['class']

        # Read pkl with search tree
        with open(kd_tree_file, 'rb') as f:
            search_tree = pickle.load(f)
        if (self.mode == "training" or self.mode == "sampling") and cloud_name in train_cloud_name_list:
            # pseudo gt
            pseudo_gt_path = join(self.path, str(self.reg_strength), "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(self.round_num), cloud_name + ".gt")
            with open(pseudo_gt_path, "rb") as f:
                pseudo_gt = pickle.load(f)
                pseudo_gt = np.asarray(pseudo_gt)
            input_activation = pseudo_gt[0]
            input_pseudo = pseudo_gt[1]

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

        noise = np.random.normal(scale=ConfigSemantic3D.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if self.mode == "sampling":
            # Query all points within the cloud
            queried_idx = input_tree.query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = input_tree.query(pick_point, k=ConfigSemantic3D.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_color = input_color[queried_idx]
        if self.mode == 'test':
            queried_pc_label = np.zeros(queried_pc_xyz.shape[0])
        else:
            queried_pc_label = input_label[queried_idx]
            queried_pc_label = np.array([self.label_to_idx[l] for l in queried_pc_label])
        queried_pc_activation = input_activation[queried_idx]
        queried_pc_pseudo = input_pseudo[queried_idx]



        return np.asarray([queried_pc_xyz]), np.asarray([queried_pc_color]), np.asarray([queried_pc_label]), \
               np.asarray([queried_pc_activation]), np.asarray([queried_pc_pseudo]), np.asarray([queried_idx]), \
               np.array([cloud_idx], dtype=np.int32)

    def tf_augment_input(self, inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = np.random.uniform(low=0, high=2 * np.pi, size=(1,))
        # Rotation matrices
        c, s = np.cos(theta), np.sin(theta)
        cs0 = np.zeros_like(c)
        cs1 = np.ones_like(c)
        R = np.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = np.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = np.reshape(np.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = ConfigSemantic3D.augment_scale_min
        max_s = ConfigSemantic3D.augment_scale_max
        if ConfigSemantic3D.augment_scale_anisotropic:
            s = np.random.uniform(size=(1, 3), low=min_s, high=max_s)
        else:
            s = np.random.uniform(size=(1, 1), low=min_s, high=max_s)

        symmetries = []
        for i in range(3):
            if ConfigSemantic3D.augment_symmetries[i]:
                symmetries.append(np.round(np.random.uniform(size=(1, 1))) * 2 - 1)
            else:
                symmetries.append(np.ones([1, 1], dtype=np.float32))
        s *= np.concatenate(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = np.tile(s, [np.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = np.random.normal(size=np.shape(transformed_xyz), scale=ConfigSemantic3D.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = np.concatenate([transformed_xyz, rgb], axis=-1)
        return stacked_features

    def split3(self, batch_xyz, source_idx, part_list, max_size=800000):
        """
        batch_xyz [point, 3]
        """
        x_min = float(np.min(batch_xyz[:, 0]))
        x_max = float(np.max(batch_xyz[:, 0]))
        x_len = x_max - x_min
        y_min = float(np.min(batch_xyz[:, 1]))
        y_max = float(np.max(batch_xyz[:, 1]))
        y_len = y_max - y_min
        z_min = float(np.min(batch_xyz[:, 2]))
        z_max = float(np.max(batch_xyz[:, 2]))
        z_len = z_max - z_min
        all_index = set(np.arange(len(batch_xyz)).tolist())

        x_select_list = []
        x1 = set(np.where(batch_xyz[:, 0] < x_min + 0.5 * x_len)[0].tolist())
        x_select_list.append(x1)
        x_select_list.append(all_index - x1)

        y_select_list = []
        y1 = set(np.where(batch_xyz[:, 1] < y_min + 0.5 * y_len)[0].tolist())
        y_select_list.append(y1)
        y_select_list.append(all_index - y1)

        z_select_list = []
        z1 = set(np.where(batch_xyz[:, 2] < z_max + 0.5 * z_len)[0].tolist())
        z_select_list.append(z1)
        z_select_list.append(all_index - z1)

        for x in x_select_list:
            for y in y_select_list:
                for z in z_select_list:
                    current_idx = list(x & y & z)
                    source_idx_part = source_idx[current_idx]
                    if len(current_idx) > max_size:
                        self.split3(batch_xyz[current_idx], source_idx_part, part_list, max_size=800000)
                    else:
                        part_list.append(source_idx_part)

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx):

        batch_features = np.asarray([self.tf_augment_input([batch_xyz[0], batch_features[0]])])
        input_part_list = []
        part_list = []
        self.split3(batch_xyz[0], np.arange(len(batch_xyz[0])), part_list, max_size=800000)

        combination_part_list = []
        for part in part_list:
            if len(part) > 2000:
                combination_part_list.append(part)
            else:
                if len(combination_part_list) > 0:
                    combination_part_list[-1] = np.concatenate([combination_part_list[-1], part],axis=0)
                else:
                    combination_part_list.append(part)



        for part in combination_part_list:
            part = list(part)
            if len(part) > 0:
                batch_xyz_sub = batch_xyz[:,part,:]
                batch_features_sub = batch_features[:, part, :]
                batch_labels_sub= batch_labels[:, part]
                batch_activation_sub = batch_activation[:, part]
                batch_pseudo_sub= batch_pseudo[:, part]
                batch_pc_idx_sub = batch_pc_idx[:, part]
                batch_cloud_idx_sub = batch_cloud_idx

                input_points = []
                input_neighbors = []
                input_pools = []
                input_up_samples = []

                for i in range(ConfigSemantic3D.num_layers):
                    neighbour_idx = DP.knn_search(batch_xyz_sub, batch_xyz_sub, ConfigSemantic3D.k_n)
                    sub_points = batch_xyz_sub[:, :np.shape(batch_xyz_sub)[1] // ConfigSemantic3D.sub_sampling_ratio[i],
                                 :]
                    pool_i = neighbour_idx[:, :np.shape(batch_xyz_sub)[1] // ConfigSemantic3D.sub_sampling_ratio[i],
                             :]
                    up_i = DP.knn_search(sub_points, batch_xyz_sub, 1)

                    input_points.append(batch_xyz_sub)
                    input_neighbors.append(neighbour_idx)
                    input_pools.append(pool_i)
                    input_up_samples.append(up_i)

                    batch_xyz_sub = sub_points


                input_list = input_points + input_neighbors + input_pools + input_up_samples  # shape[0] = 5+5+5+5

                input_list += [batch_features_sub, batch_labels_sub, batch_activation_sub, batch_pseudo_sub, batch_pc_idx_sub, batch_cloud_idx_sub]  # shape[0] = 20+6

                input_part_list.append(input_list)
        return input_part_list

    def __len__(self):
        return len(self.input_cloud_names)

    def __getitem__(self, idx):
        cloud_name = self.input_cloud_names[idx]
        input_activation, input_pseudo, input_tree, input_color, input_label = self.load_data(cloud_name)
        batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx = self.spatially_regular_gen(input_activation, input_pseudo, input_tree, input_color, input_label, idx)
        input_list = self.tf_map(batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx)
        return input_list
