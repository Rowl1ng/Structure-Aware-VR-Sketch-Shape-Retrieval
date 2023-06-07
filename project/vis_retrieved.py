from unicodedata import category
import numpy as np
import cv2
import os
VIEW_DIR = '/vol/vssp/datasets/multiview/3VS/visualization/images/mitsuba_view'
# 
new_sketch_3d_dir = '/scratch/dataset/3D_sketch_2021/mitsuba'
exp_dict ={
    'retrieval_fitting_gap_triplet_2': '3DV21',
    'retrieval_fitting_gap_regression_triplet_margin_0.3_1.2': 'weighted margin loss'
}
cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
numbers = ['A', 'B', 'C', 'D', 'E']

test_data = 'sketch_21_test' #'aligned_sketch'

def make_comparison_fig(query_name_list, gallery_name_list, pair_sorts, save_dir, sketch_3d_dir, shape_dir, metric, EXTENSION='.jpg', with_score=False):
    K = 10

    for index in range(len(query_name_list)):
        query_id = query_name_list[index]
        #GT
        shape_dx=20
        gt = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]

        # 3D baseline
        fig_3Ds = []
        for pair_sort in pair_sorts:
            top_5 = pair_sort[index, :K]

            scores = metric[index, top_5]
            ##shape
            shape_imgs = []

            for x, score in zip(top_5, scores):
                shape_img = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(gallery_name_list[x])))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] 
                if with_score:
                    shape_img = cv2.putText(shape_img, f'{score * 100 :.2f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                shape_imgs.append(shape_img)

            # shape_imgs = [cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(gallery_name_list[i])))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] for i in top_5]
            rank = np.where(top_5==index)[0]

            if len(rank) > 0:
                start = 10
                width = shape_imgs[0].shape[0]
                new_img = cv2.rectangle(shape_imgs[rank[0]], (start, start), (width-start, width-start), (0,255,0), 2)
                shape_imgs[rank[0]] = new_img
            shape_top5 = cv2.hconcat(shape_imgs)

            ##3d sketch
            if test_data == 'sketch_21_test':
                output = cv2.imread(os.path.join(new_sketch_3d_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
            else:
                output = cv2.imread(os.path.join(sketch_3d_dir, query_id + EXTENSION))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
            # offset = (output_2d.shape[1] - output.shape[1])//2
            # output_3d = cv2.copyMakeBorder(output, 0,0,offset, output_2d.shape[1] - output.shape[1] - offset, cv2.BORDER_CONSTANT, value=(255,255,255))
            fig_3d = cv2.hconcat([output, gt, shape_top5])
            fig_3Ds.append(fig_3d)
        fig_final = cv2.vconcat(fig_3Ds)
        if with_score:
            save_path = os.path.join(save_dir, '{}_{}_score.png'.format(index, query_id))
        else:
            save_path = os.path.join(save_dir, '{}_{}.png'.format(index, query_id))
        cv2.imwrite(save_path,fig_final) 
    # return fig_final
def make_comparison_fig_metrics(query_name_list, gallery_name_list, pair_sorts, save_dir, shape_dir, index_list=None):
    K = 6
    rows = len(pair_sorts)
    if index_list is None:
        index_list = range(len(query_name_list))
    for index, item in enumerate(index_list):

        query_id = query_name_list[item]
        #GT
        shape_dx=20
        gt = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]

        # 3D baseline
        fig_3Ds = []
        for i, pair_sort in enumerate(pair_sorts):
            top_5 = pair_sort[index, 1:K]

            ##shape
            shape_imgs = [cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(gallery_name_list[i])))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] for i in top_5]
            # rank = np.where(top_5==item)[0]
            # if len(rank) > 0:
            #     start = 10
            #     width = shape_imgs[0].shape[0]
            #     new_img = cv2.rectangle(shape_imgs[rank[0]], (start, start), (width-start, width-start), (0,255,0), 2)
            #     shape_imgs[rank[0]] = new_img
            text_image = np.ones((shape_imgs[0].shape[0],int(shape_imgs[0].shape[1]/2),3), dtype=np.uint8) * 255
            text_image = cv2.putText(text_image, numbers[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 0), 2, cv2.LINE_AA)
            shape_top5 = cv2.hconcat(shape_imgs)
            fig_3d = cv2.hconcat([text_image, shape_top5])

            fig_3Ds.append(fig_3d)
        fig_final = cv2.vconcat(fig_3Ds)
        offset = int((fig_final.shape[0] - gt.shape[0])/2)
        gt_pad = cv2.copyMakeBorder(gt, offset, fig_final.shape[0] - gt.shape[0] - offset, 0,0, cv2.BORDER_CONSTANT, value=(255,255,255))

        gt_fig_final = cv2.hconcat([gt_pad, fig_final])
        save_path = os.path.join(save_dir, '{}_{}.png'.format(index, query_id))
        cv2.imwrite(save_path,gt_fig_final) 

def get_retrieval_sorted_ids(filepath):
    data = np.load(filepath)
    pair_sort = np.argsort(data)    
    return pair_sort

def vis_retrieved():

    file_retrieval_tl_hs = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold/{}/inference/test_rank_{}_last.npy'
    # file_retrieval_fitting_gap = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference/CD_d.npy'

    # Query and target lists:
    file_gallery = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'
    file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    # exps = ['retrieval_fitting_gap_triplet_CD_d_CD_d@1_2', 
    #         'retrieval_fitting_gap_triplet_CD_d_CD_d@1_aug_sketch_aug_shape_2',
    #         'retrieval_fitting_gap_regression_triplet_CD_d_CD_d@1_d2_2',
    #         'retrieval_fitting_gap_regression_triplet_CD_d_CD_d@1_d2_aug_sketch_aug_shape',
    #         'retrieval_fitting_gap_regression_triplet_CD_d_CD_d@1_symmetric'
    #         ]
    exps = [
    'triplet_multickpt_aug_1', 
    'adaptive_triplet_multickpt_aug_1',
    'adaptive_triplet_multickpt_sym_aug_1'
    ]
    pair_sorts  =  [get_retrieval_sorted_ids(file_retrieval_tl_hs.format(exp, test_data)) for exp in exps]

    save_dir = '/scratch/visualization/baselines_comparison/new/multifold_3baselines'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sketch_3d_dir = new_sketch_3d_dir
    shape_dir = os.path.join(VIEW_DIR, 'shapenet', '03001627')
    split = 'test'
    metric_name = 'CD_d'
    metric_path = os.path.join(cache_dir, '{}_selected_metrics'.format(split), '{}.npy'.format(metric_name))
    metric = np.load(metric_path)
    make_comparison_fig(query_name_list, gallery_name_list, pair_sorts, save_dir, sketch_3d_dir, shape_dir, metric)

def vis_retrieved_multifold():
    models = ['triplet_multickpt_0.6', 
          'triplet_multickpt_aug',
          'adaptive_triplet_multickpt_0.3_1.2', 
         'adaptive_triplet_multickpt_aug',
          'adaptive_triplet_multickpt_sym',
          'adaptive_triplet_multickpt_sym_aug',
          'adaptive_triplet_multickpt_CD',
          'adaptive_triplet_multickpt_CD_aug',
         ]
    dist_path = 'inferencenew/test_rank_aligned_sketch_{}.npy'
    root_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold'
    
    # Query and target lists:
    file_gallery = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'
    file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    ckpt = 'val_bi_CD_d@1'
    fold_id = 1
    pair_sorts  =  [get_retrieval_sorted_ids(os.path.join(root_dir, '{}_{}'.format(model, fold_id), dist_path.format(ckpt))) for model in models]

    save_dir = '/scratch/visualization/baselines_comparison/multifold_{}'.format(ckpt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    make_comparison_fig(query_name_list, gallery_name_list, pair_sorts, save_dir)

def vis_retrieved_RL_TL():
    models = [
          'adaptive_triplet_multickpt_0.3_1.2',
          'regression_multickpt'
         ]
    dist_path = 'inference/test_rank_aligned_sketch_{}.npy'
    root_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold'
    
    # Query and target lists:
    file_gallery = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'
    file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    ckpt = 'last'
    fold_id = 1
    pair_sorts  =  [get_retrieval_sorted_ids(os.path.join(root_dir, '{}_{}'.format(model, fold_id), dist_path.format(ckpt))) for model in models]

    save_dir = '/scratch/visualization/baselines_comparison/RL_vs_TL'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sketch_3d_dir = os.path.join(VIEW_DIR, 'chair_1005_align_view')
    shape_dir = os.path.join(VIEW_DIR, 'shapenet', '03001627')
    split = 'test'
    metric_name = 'CD_d'
    metric_path = os.path.join(cache_dir, '{}_selected_metrics'.format(split), '{}.npy'.format(metric_name))
    metric = np.load(metric_path)
    make_comparison_fig(query_name_list, gallery_name_list, pair_sorts, save_dir, sketch_3d_dir, shape_dir, metric, with_score=True)
    # make_comparison_fig_metrics(query_name_list, gallery_name_list, pair_sorts, save_dir, shape_dir)
def vis_retrieved_category(category, ckpt='val_CD_d@5'):
    root_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold'
    models = ['triplet_multickpt_aug_{}_ss1.0', 
            'adaptive_triplet_multickpt_aug_{}_ss1.0', 
            'adaptive_triplet_multickpt_sym_aug_{}_ss1.0']
    dist_path = 'inference/test_rank_aligned_sketch_{}.npy'
    pair_sorts  =  [get_retrieval_sorted_ids(os.path.join(root_dir, model.format(category), dist_path.format(ckpt))) for model in models]
    data_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet'
    file_query = os.path.join(data_dir, 'lists', category, 'sketch_test.txt')
    file_gallery = os.path.join(data_dir, 'lists', category, 'sketch_test_shape.txt')

    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]
    save_dir = '/scratch/visualization/baselines_comparison/category/{}/ckpt_{}'.format(category, ckpt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    shape_dir = os.path.join(VIEW_DIR, 'shapenet', category)
    sketch_3d_dir =  os.path.join(VIEW_DIR ,'shapenet/synthetic_sketch/{}'.format(category))

    make_comparison_fig(query_name_list, gallery_name_list, pair_sorts, save_dir, sketch_3d_dir, shape_dir, EXTENSION='_network_20_aggredated_sketch_1.0_00.jpg')

def vis_nearest_neighbor():
    file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
    file_gallery = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'

    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    split = 'test'
    save_dir = os.path.join(cache_dir, '{}_selected_metrics'.format(split))
    metrics = {
    # 'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    # "F_0.02":[],
    # "F_0.01_d":[],
    # "bi_F_0.01_d":[],
    }

    for item in metrics.keys():
        save_path = os.path.join(save_dir, '{}.npy'.format(item))
        metrics[item] = np.load(save_path)
    
        if 'F' in item:
            metrics[item] = 1 - metrics[item]
    pair_sorts = [np.argsort(metrics[item]) for item in metrics.keys()]
    save_dir = '/scratch/visualization/selected_metrics_comparison_symmetric'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # index_list = [8, 17, 57, 68, 89, 91, 95, 114, 127, 166, 172, 182, 201]
    index_list = None
    shape_dir = os.path.join(VIEW_DIR, 'shapenet', '03001627')
    make_comparison_fig_metrics(query_name_list, gallery_name_list, pair_sorts, save_dir, shape_dir, index_list=index_list)

def generate_deformed(shape_dataset, deformer_name, save_dir, query_name_list, gallery_name_list, metrics, index_list=None):
    K = 6
    import torch
    from src.test import get_deformer
    deformer = get_deformer(deformer_name)
    pair_sorts = [np.argsort(metrics[item]) for item in metrics.keys()]


    if index_list is None:
        index_list = range(len(query_name_list))
    for query_id, item in enumerate(index_list):
        # target
        source_shape = torch.tensor(shape_dataset.__getitem__(query_id)['shape']).unsqueeze(0).cuda()
        point_list = np.array(shape_dataset.__getitem__(query_id)['shape'], dtype='float32')
        save_filename = '{}.npy'.format(query_name_list[query_id])
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.exists(save_path):
            np.save(save_path, point_list)
        # 3D baseline
        fig_3Ds = []
        for i, pair_sort in enumerate(pair_sorts):
            top_5 = pair_sort[query_id, 1:K]
            for gallery_id in top_5:
                target_shape = torch.tensor(shape_dataset.__getitem__(gallery_id)['shape']).unsqueeze(0).cuda()
                point_list = np.array(shape_dataset.__getitem__(gallery_id)['shape'], dtype='float32')
                save_filename = '{}.npy'.format(gallery_name_list[gallery_id])
                save_path = os.path.join(save_dir, save_filename)
                if not os.path.exists(save_path):
                    np.save(save_path, point_list)

                save_filename = '{}_to_{}.npy'.format(query_name_list[query_id], gallery_name_list[gallery_id])
                save_path = os.path.join(save_dir, save_filename)
                if not os.path.exists(save_path):
                    deformed = deformer(source_shape, target_shape)['deformed'].squeeze(0).detach().data # torch.Size([6, 1024, 3])   
                    np.save(save_path, deformed.cpu().numpy())
                
                save_filename = '{}_to_{}.npy'.format(gallery_name_list[gallery_id], query_name_list[query_id])
                save_path = os.path.join(save_dir, save_filename)
                if not os.path.exists(save_path):
                    deformed = deformer(target_shape, source_shape)['deformed'].squeeze(0).detach().data # torch.Size([6, 1024, 3])   
                    np.save(save_path, deformed.cpu().numpy())
    
    # Render views
    import sys
    sys.path.append("..") 
    from graphics.vis_pc_mitsuba import main
    from glob import glob
    pc_paths = glob(os.path.join(save_dir, '*.npy'))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    work_info = [['', pathToFile, save_dir] for pathToFile in pc_paths if not os.path.exists(pathToFile.replace('.npy', '_00.jpg'))]
    from multiprocessing import Pool

    with Pool(16) as p:
        p.map(main, work_info)

    

def make_comparison_fig_metrics_deformed(shape_dataset, shape_dir, deformed_dir, query_name_list, gallery_name_list, metrics, save_dir, index_list=None):
    def get_fitting_gap(source_name, target_index):
        import pytorch3d.loss
        import torch
        deformed_shape = np.load(os.path.join(deformed_dir, '{}_to_{}.npy'.format(source_name, gallery_name_list[x])))
        target_shape = shape_dataset.__getitem__(target_index)
        assert target_shape['file'] == gallery_name_list[x]
        cd_dist = pytorch3d.loss.chamfer_distance(torch.tensor(deformed_shape).unsqueeze(0),
         torch.tensor(target_shape['shape']).unsqueeze(0), batch_reduction=None)[0][0].data.cpu().numpy()
        return cd_dist
    K = 6
    pair_sorts = {item: np.argsort(metrics[item]) for item in metrics.keys()}
    # deformed_dir = '/scratch/visualization/test_deformed_results'
    rows = len(metrics.keys())
    if index_list is None:
        index_list = range(len(query_name_list))
    for index, item in enumerate(index_list):

        query_id = query_name_list[item]
        #GT
        shape_dx=20
        gt = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]

        # 3D baseline
        fig_3Ds = []
        for i, metric in enumerate(metrics.keys()):
            pair_sort = pair_sorts[metric]
            top_5 = pair_sort[index, 1:K]
            if 'F' in metric:
                scores = 1 - metrics[metric][index, top_5]
            else:
                scores = metrics[metric][index, top_5]
            ##shape
            shape_imgs = []
            for x, score in zip(top_5, scores):
                shape_img = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(gallery_name_list[x])))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] 
                shape_img = cv2.putText(shape_img, f'{score * 100 :.2f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                shape_imgs.append(shape_img)
            shape_top5 = cv2.hconcat(shape_imgs)
            if '_d' in metric:
                # deformed results: query <- gallery
                deformed_shape_imgs = []
                for x, score in zip(top_5, scores):
                    CD_d_score = metrics['CD_d'][item, x]
                    deformed_shape_img = cv2.imread(os.path.join(deformed_dir, '{}_to_{}_00.jpg'.format(gallery_name_list[x], query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] 
                    deformed_shape_img = cv2.putText(deformed_shape_img, f'{CD_d_score * 100 :.2f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    deformed_shape_imgs.append(deformed_shape_img)                
                deformed_top5 = cv2.hconcat(deformed_shape_imgs)
                shape_top5 = cv2.vconcat([shape_top5, deformed_top5])

                # deformed results: query -> gallery
                deformed_shape_imgs = []
                for x, score in zip(top_5, scores):
                    deformed_shape_img = cv2.imread(os.path.join(deformed_dir, '{}_to_{}_00.jpg'.format(query_id, gallery_name_list[x])))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] 

                    if 'bi' in metric:
                        CD_d_reverse_score = 2 * score - metrics['CD_d'][item, x]
                    else:
                        CD_d_reverse_score = get_fitting_gap(query_id, x)
                    deformed_shape_img = cv2.putText(deformed_shape_img, f'{CD_d_reverse_score * 100 :.2f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    deformed_shape_imgs.append(deformed_shape_img)                
                deformed_top5 = cv2.hconcat(deformed_shape_imgs)
                shape_top5 = cv2.vconcat([shape_top5, deformed_top5])
                
                text_image = np.ones((shape_imgs[0].shape[0]*3, int(shape_imgs[0].shape[1]/2), 3), dtype=np.uint8) * 255
                # else:
                #     text_image = np.ones((shape_imgs[0].shape[0]*2, int(shape_imgs[0].shape[1]/2), 3), dtype=np.uint8) * 255

            else:
                text_image = np.ones((shape_imgs[0].shape[0], int(shape_imgs[0].shape[1]/2), 3), dtype=np.uint8) * 255
            text_image = cv2.putText(text_image, numbers[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 0), 2, cv2.LINE_AA)

            fig_3d = cv2.hconcat([text_image, shape_top5])

            fig_3Ds.append(fig_3d)
        fig_final = cv2.vconcat(fig_3Ds)
        offset = int((fig_final.shape[0] - gt.shape[0])/2)
        gt_pad = cv2.copyMakeBorder(gt, offset, fig_final.shape[0] - gt.shape[0] - offset, 0,0, cv2.BORDER_CONSTANT, value=(255,255,255))

        gt_fig_final = cv2.hconcat([gt_pad, fig_final])
        save_path = os.path.join(save_dir, '{}_{}.png'.format(index, query_id))
        cv2.imwrite(save_path,gt_fig_final)            

def vis_nearest_neighbor_with_deformed():
    category = '03001627'

    file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
    file_gallery = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test_shape.txt'

    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    split = 'test'
    save_dir = os.path.join(cache_dir, '{}_selected_metrics'.format(split))
    metrics = {
    'CD':[],
    'CD_d':[],
    # 'bi_CD_d':[],
    # "F_0.02":[],
    "F_0.01_d":[],
    # "bi_F_0.01_d":[],
    }

    for item in metrics.keys():
        save_path = os.path.join(save_dir, '{}.npy'.format(item))
        metrics[item] = np.load(save_path)
    
        if 'F' in item:
            metrics[item] = 1 - metrics[item]

    deformer_name = 'deformer_cage_sh2sh_template'
    deformed_dir = '/scratch/visualization/test_deformed_results/{}'.format(category)
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'list/hs/test_shape.txt',
        'category': category
    }
    from src.datasets.SketchyVRLoader import SketchyVR_single
    shape_dataset = SketchyVR_single(datamodule, mode='test', type='shape')

    generate_deformed(shape_dataset, deformer_name, deformed_dir, query_name_list, gallery_name_list, metrics)
    save_dir = '/scratch/visualization/selected_metrics_comparison/with_deformed/{}/perceptual_study/with_scores'.format(category)
    # shape_dir = os.path.join(VIEW_DIR, 'shapenet', category)
    shape_dir = deformed_dir
    make_comparison_fig_metrics_deformed(shape_dataset, shape_dir, deformed_dir, query_name_list, gallery_name_list, metrics, save_dir)

def vis_nearest_neighbor_with_deformed_category(category):
    data_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet'
    mode = 'val'
    name_list = os.path.join(data_dir, 'lists', category, 'sketch_{}.txt'.format(mode))

    file_query = name_list
    file_gallery = name_list

    query_name_list = [line.rstrip() for line in open(file_query)]
    gallery_name_list = [line.rstrip() for line in open(file_gallery)]

    metrics = {
    # 'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    # "F_0.02":[],
    # "F_0.01_d":[],
    # "bi_F_0.01_d":[],
    }

    for item in metrics.keys():
        save_path = os.path.join(cache_dir, 'category', '{}_{}.npy'.format(category, item))
        metrics[item] = np.load(save_path)
    
        if 'F' in item:
            metrics[item] = 1 - metrics[item]

    deformed_dir = '/scratch/visualization/test_deformed_results/{}'.format(category)
    if not os.path.exists(deformed_dir):
        os.mkdir(deformed_dir)
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'list/hs/test_shape.txt',
        'category': category
    }
    from src.datasets.SketchyVRLoader import SketchyVR_original_category_single
    # shape_dataset = SketchyVR_single(datamodule, mode='test', type='shape')
    shape_dataset = SketchyVR_original_category_single(datamodule, mode=mode, type='shape', cache=True)

    deformer_name = 'deformer_cage_sh2sh_shapenet_template_{}'.format(category)

    generate_deformed(shape_dataset, deformer_name, deformed_dir, query_name_list, gallery_name_list, metrics)

    # Generate figures:
    save_dir = '/scratch/visualization/selected_metrics_comparison_symmetric/with_deformed/{}'.format(category)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    shape_dir = os.path.join(VIEW_DIR, 'shapenet', category)

    make_comparison_fig_metrics_deformed(shape_dir, deformed_dir, query_name_list, gallery_name_list, metrics, save_dir)

if __name__ == '__main__':
    vis_retrieved()
    # vis_retrieved_RL_TL()
    # vis_nearest_neighbor_with_deformed()
    # vis_retrieved_multifold()
    # for category in ['03636649', '02691156']:
        # for ckpt in ['last', 'val_CD_d@5']:
        # vis_nearest_neighbor_with_deformed_category(category)
            # vis_retrieved_category(category, ckpt)
