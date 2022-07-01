import numpy as np
import glob
import pickle
import time
from os.path import join

import numpy as np

from helper_ply import read_ply
from helper_tool import ConfigS3DIS
from helper_tool import DataProcessing as DP


class S3DIS_Dataset_Test:
    def __init__(self, test_area_idx):
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
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'input_{:.3f}'.format(ConfigS3DIS.sub_grid_size), '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_names = []

        self.load_sub_sampled_clouds(ConfigS3DIS.sub_grid_size)
        self.init_possibility()

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                # Name of the input files
                kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
                sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T   # shape=[point_number, 3]
                sub_labels = data['class']  # shape=[point_number]

                # Read pkl with search tree
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)


                self.input_trees += [search_tree]
                self.input_colors += [sub_colors]
                self.input_labels += [sub_labels]
                self.input_names += [cloud_name]

                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

                size = sub_colors.shape[0] * 4 * 7
                print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

    def init_possibility(self):
        self.current_batch = 0
        self.possibility = []
        self.min_possibility = []
        # Random initialize
        for i, tree in enumerate(self.input_colors):
            self.possibility += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

    def reset_current_batch(self):
        self.current_batch = 0

    def get_batch(self):
        self.current_batch = self.current_batch + 1
        if self.current_batch > ConfigS3DIS.val_steps:
            return []
        else:
            batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx = [], [], [], [], []
            # Generator loop
            for i in range(ConfigS3DIS.val_batch_size):
                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(np.asarray(self.min_possibility)))
                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[cloud_idx])
                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[cloud_idx].data, copy=False)
                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                noise = np.random.normal(scale=ConfigS3DIS.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < ConfigS3DIS.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[cloud_idx].query(pick_point, k=ConfigS3DIS.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[cloud_idx][queried_idx]
                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)  # shape=[queried_number]
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[cloud_idx][queried_idx] += delta
                self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))
                # up_sampled with replacement
                if len(points) < ConfigS3DIS.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_pc_activation, queried_pc_pseudo = \
                            DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels,
                                        np.zeros([len(queried_pc_labels)]), np.zeros([len(queried_pc_labels)]),
                                        queried_idx, ConfigS3DIS.num_points)

                batch_xyz.append(queried_pc_xyz.astype(np.float32))
                batch_features.append(queried_pc_colors.astype(np.float32))
                batch_labels.append(queried_pc_labels)
                batch_pc_idx.append(queried_idx.astype(np.int32))
                batch_cloud_idx.append(cloud_idx)


            return self.tf_map(np.asarray(batch_xyz), np.asarray(batch_features), np.asarray(batch_labels),
                               np.asarray(batch_pc_idx, dtype=np.int32), np.asarray(batch_cloud_idx, dtype=np.int32))

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):

        # print(batch_xyz.shape, batch_features.shape)
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigS3DIS.num_layers):
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, ConfigS3DIS.k_n)
            sub_points = batch_xyz[:, :np.shape(batch_xyz)[1] // ConfigS3DIS.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :np.shape(batch_xyz)[1] // ConfigS3DIS.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)

            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples  # shape[0] = 5+5+5+5
        input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]  # shape[0] = 20+4

        return input_list