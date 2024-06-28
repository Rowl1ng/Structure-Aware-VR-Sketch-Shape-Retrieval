import os
from src.utils import utils
import os
import logging
import numpy as np
from glob import glob
def get_index():
    image_dir = '/scratch/dataset/3D_sketch_2021/compare_old_new_user_id'
    obj_files = glob(os.path.join(image_dir, '10_*.png'))
    indexes = [int(os.path.basename(item).split('_')[1]) for item in obj_files]
    filtered_list = [x for x in range(202) if x not in indexes]    
    return filtered_list
def compute_acc_at_k(pair_sort, index_list):
    # pair_sort = np.argsort(dist)
    count_1 = 0
    count_5 = 0
    count_10 = 0
    query_num = len(index_list)
    for idx1 in index_list:
        if idx1 in pair_sort[idx1, 0:1]:
            count_1 = count_1 + 1
        if idx1 in pair_sort[idx1, 0:5]:
            count_5 = count_5 + 1
        if idx1 in pair_sort[idx1, 0:10]:
            count_10 = count_10 + 1
    acc_1 = count_1 / float(query_num)
    acc_5 = count_5 / float(query_num)
    acc_10 = count_10 / float(query_num)
    return [acc_1, acc_5, acc_10]

def compute_val(args):
    indexes = get_index()

    save_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference'.format(args.deformer)

    log = utils.logger_config(os.path.join(save_dir, 'run.log'), __name__)
    dist_path = os.path.join(save_dir, 'test_rank_5794_sketch_21_test_best.npy'.format())
    dis_mat = np.load(dist_path)
    pair_sort = np.argsort(dis_mat)
    acc_at_k = compute_acc_at_k(pair_sort, indexes)
    for acc, k in zip(acc_at_k, [1, 5, 10]):
        log.info(f' * Acc@{k}: {acc * 100 :.2f}')

    for metric in ['CD', 'CD_d']:
        fg_path = os.path.join(save_dir, '{}_sketch_21_test_best.npy'.format(metric))
        fg = np.load(fg_path)
        new_fg = fg[indexes]
        for k in [1, 5, 10]:
            log.info(f' * Mean {metric}@{k}: {new_fg[:, :k].mean() * 100 :.2f}')




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--deformer', type=str, default='retrieval_fitting_gap_regression_triplet_CD_d_CD_d@5_symmetric', 
                        help='deformer')

    args = parser.parse_args()

    compute_val(args)
