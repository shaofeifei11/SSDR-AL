import numpy as np

from semantic3d_dataset_sampling import *

train_val_cloud_name_list = ['bildstein_station1_xyz_intensity_rgb',
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


class Semantic3D_Dataset_Train:

    def __init__(self, reg_strength, sampler_args, round_num):
        self.reg_strength = reg_strength
        self.sampler_args = sampler_args
        self.round_num = round_num

        self.name = 'semantic3d'
        self.path = './data/semantic3d'


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

        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(ConfigSemantic3D.sub_grid_size))

        # Initial training-validation-testing files
        self.train_files = []
        self.val_files = []

        cloud_names = [file_name[:-4] for file_name in os.listdir(self.sub_pc_folder) if file_name[-4:] == '.ply']
        for pc_name in cloud_names:
            if pc_name in val_cloud_name_list:
                self.val_files.append(join(self.full_pc_folder, pc_name + '.ply'))
            elif pc_name in train_val_cloud_name_list:
                self.train_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        self.val_files = np.sort(self.val_files)
        self.train_files = np.sort(self.train_files)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []

        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_names = []
        self.input_activations = []
        self.input_pseudos = []

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(ConfigSemantic3D.sub_grid_size)
        self.init_possibility()

    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))

        for file_path in self.train_files:
            cloud_name = file_path.split('/')[-1][:-4]

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']  # shape=[point_number]

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            self.input_names += [cloud_name]

            pseudo_gt_path = join(self.path, str(self.reg_strength), "sampling",
                                  get_sampler_args_str(self.sampler_args), "round_" + str(self.round_num),
                                  cloud_name + ".gt")
            with open(pseudo_gt_path, "rb") as f:
                pseudo_gt = pickle.load(f)
                pseudo_gt = np.asarray(pseudo_gt)
            self.input_activations += [pseudo_gt[0]]
            self.input_pseudos += [pseudo_gt[1]]

    # Generate the input data flow
    def init_possibility(self):
        self.current_batch = 0
        # Reset possibility
        self.possibility = []
        self.min_possibility = []
        self.class_weight = []

        # Random initialize
        for i, tree in enumerate(self.input_trees):
            self.possibility += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

        _, num_class_total = np.unique(np.hstack(self.input_labels), return_counts=True)
        self.class_weight += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

    def get_batch(self):
        self.current_batch = self.current_batch + 1
        if self.current_batch > ConfigSemantic3D.train_steps:
            self.current_batch = 0
            return []
        else:
            batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx = [], [], [], [], [], [], []
            # Generator loop
            for i in range(ConfigSemantic3D.batch_size):
                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=ConfigSemantic3D.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[cloud_idx].query(pick_point, k=ConfigSemantic3D.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[cloud_idx][query_idx]

                queried_pc_labels = self.input_labels[cloud_idx][query_idx]
                queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])

                # print(self.input_activations[0])
                # print(self.input_activations[0][0])
                queried_pc_activation = self.input_activations[cloud_idx][query_idx]
                queried_pc_pseudo = self.input_pseudos[cloud_idx][query_idx]

                queried_pt_weight = np.array([self.class_weight[0][n] for n in queried_pc_labels])

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[cloud_idx][query_idx] += delta
                self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

                batch_xyz.append(queried_pc_xyz.astype(np.float32))
                batch_features.append(queried_pc_colors.astype(np.float32))
                batch_labels.append(queried_pc_labels)
                batch_activation.append(queried_pc_activation)
                batch_pseudo.append(queried_pc_pseudo)
                batch_pc_idx.append(query_idx.astype(np.int32))
                batch_cloud_idx.append(cloud_idx)

            return self.tf_map(np.asarray(batch_xyz), np.asarray(batch_features), np.asarray(batch_labels), np.asarray(batch_activation),
                               np.asarray(batch_pseudo), np.asarray(batch_pc_idx, dtype=np.int32), np.asarray(batch_cloud_idx, dtype=np.int32))

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx):
        batch_features = np.asarray([self.tf_augment_input([batch_xyz[j], batch_features[j]]) for j in range(ConfigSemantic3D.batch_size)])
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(ConfigSemantic3D.num_layers):
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, ConfigSemantic3D.k_n)
            sub_points = batch_xyz[:, :np.shape(batch_xyz)[1] // ConfigSemantic3D.sub_sampling_ratio[i],:]
            pool_i = neighbour_idx[:, :np.shape(batch_xyz)[1] // ConfigSemantic3D.sub_sampling_ratio[i],:]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)

            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_labels, batch_activation, batch_pseudo, batch_pc_idx, batch_cloud_idx]

        return input_list

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