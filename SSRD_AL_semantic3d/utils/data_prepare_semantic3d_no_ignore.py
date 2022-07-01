from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

grid_size = 0.06
dataset_path = './data/semantic3d/original_data'
original_pc_folder = join(dataset_path, 'original_ply')
sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
cloud_name_list = ['bildstein_station1_xyz_intensity_rgb',
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
                   'untermaederbrunnen_station3_xyz_intensity_rgb',
                   "bildstein_station3_xyz_intensity_rgb",
                   "sg27_station2_intensity_rgb"]
if True:
    for cloud_name in cloud_name_list:
        pc = DP.load_pc_semantic3d(join(dataset_path, cloud_name + '.txt'))
        labels = DP.load_label_semantic3d(join(dataset_path, cloud_name + '.labels'))
        full_ply_path = join(original_pc_folder, cloud_name + '.ply')

        #  Subsample to save space
        points, colors, labels = DP.grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                                  pc[:, 4:7].astype(np.uint8), labels, 0.01)
        print(cloud_name, len(points), len(labels))
        labels = np.squeeze(labels)
        print(len(labels))
        colors = np.array(colors)
        points = np.array(points)


        valid_idx = np.where(labels != 0)
        points = points[valid_idx]
        colors = colors[valid_idx]
        labels = labels[valid_idx]
        labels = labels - 1


        write_ply(full_ply_path, (points, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        # save sub_cloud and KDTree file
        sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(points, colors, labels, grid_size)
        sub_colors = sub_colors / 255.0
        sub_labels = np.squeeze(sub_labels)
        sub_ply_file = join(sub_pc_folder, cloud_name + '.ply')
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(sub_pc_folder, cloud_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(points, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, cloud_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)
