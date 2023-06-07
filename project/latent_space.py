from src.datasets.SketchyVRLoader import SketchyVR_single
from torch.utils.data import DataLoader
import numpy as np
from src.test import get_retrieval_model
import torch
import os

def compute_feature(split, model_name):
    batch_size = 10
    data_type = 'shape'
    monitor = 'last'
    target_datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/hs/val.txt',
        'val_shape_list': 'list/hs/val.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test_shape.txt',

    }
    target_shape_dataset = SketchyVR_single(target_datamodule, mode=split, type=data_type )

    shape_loader = DataLoader(dataset=target_shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=target_shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    model = get_retrieval_model(model_name)
    shape_features = []
    with torch.no_grad():
        for i, data in enumerate(shape_loader):
            shape = data['shape'].cuda()
            feat = model.encoder(shape.transpose(1, 2))
            shape_features.append(feat)
    shape_features = torch.cat(shape_features, 0).detach().data.cpu().numpy()
    
    save_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold'
    save_path = os.path.join(save_dir, model_name, 'inference', '{}_{}_{}_feat.npy'.format(split, data_type , monitor))
    np.save(save_path, shape_features)
    print('Save to: ' + save_path)



if __name__ == "__main__":
    models = [
        # 'regression_multickpt',
        #  'adaptive_triplet_multickpt_aug',
        #  'triplet_multickpt_aug',
        'adaptive_triplet_multickpt_sym_aug'
         ]
    fold_id = '1'
    for model in models:
        model_name = '_'.join([model, fold_id])
        compute_feature('test', model_name)