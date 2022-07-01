import pickle, yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = 0.06
dataset_path = './data/SemanticKITTI/dataset/sequences'
output_path = './data/SemanticKITTI/' + 'input_{:.3f}'.format(grid_size)
seq_list = np.sort(os.listdir(dataset_path))

for seq_id in seq_list:
    print('sequence' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    os.makedirs(output_path) if not exists(output_path) else None

    if int(seq_id) < 11:
        label_path = join(seq_path, 'labels')
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))
            labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
            search_tree = KDTree(sub_points)
            KDTree_save = join(output_path, str(seq_id)+"-"+str(scan_id[:-4]) + '.pkl')
            write_ply(join(output_path, str(seq_id)+"-"+str(scan_id[:-4]) + '.ply'), [sub_points, sub_labels], ['x', 'y', 'z', 'class'])
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)

            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            proj_save = join(output_path, str(seq_id)+"-"+str(scan_id[:-4]) + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)

