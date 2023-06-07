from glob import glob
import os
from numpy.lib.npyio import save
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.point_cloud_utils import resample_mesh, normalize, rotate_point_cloud

s = 1
def rotate_point_cloud(batch_data, dim='x', angle=-90): # torch.Size([1024, 3])
    rotation_angle = angle/360 * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if dim=='x':
        rotation_matrix = torch.tensor([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]]).float()
    elif dim=='y':
        rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]]).float()
    elif dim=='z':
        rotation_matrix = torch.tensor([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]]).float()
    else:
        NotImplementedError
        
    rotated_data = torch.mm(batch_data, rotation_matrix)
    return rotated_data # torch.Size([1024, 3])

def draw_subplot(pc, original_pc, fn, s=s):
    pc = rotate_point_cloud(pc, dim='x')
    pc = rotate_point_cloud(pc, dim='z', angle=90)
    original_pc = rotate_point_cloud(original_pc, dim='x')
    original_pc = rotate_point_cloud(original_pc, dim='z', angle=90)

    rows = 1
    cols = 2
    fig = plt.figure(figsize=plt.figaspect(rows / cols))
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    ax.scatter(original_pc[:, 0],original_pc[:, 1], original_pc[:, 2], c=original_pc[:, 2], s=s)
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 2, projection='3d')
    ax.scatter(pc[:, 0],pc[:, 1], pc[:, 2], c=pc[:, 2], s=s)
    plt.axis('off')

    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()
    print('save file:'+fn)


original_dir = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/aligned_sketch'
save_dir = '/scratch/dataset/3D_sketch_2021/sketch_view'
file_query = '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch/list/hs/test.txt'
query_name_list = [line.rstrip() for line in open(file_query)]
pc_dir = '/scratch/dataset/3D_sketch_2021/sketch_pc_align'

def vis(index_list):


    # model_files = glob(os.path.join(pc_dir, '*.npy'))
    for index in index_list:
        obj_name = query_name_list[index]
        pc_path = os.path.join(pc_dir, obj_name + '.npy')
        pc = torch.tensor(np.load(pc_path))
        # pc = rotate_point_cloud(pc, dim='y', angle=-90)

        # obj_name = os.path.basename(model_files[index]).split('.')[0]
        # obj_name = filename.split('_')[0]

        original_pc_path = os.path.join(original_dir, obj_name + '.npy')
        original_pc = torch.tensor(np.load(original_pc_path))

        fn = os.path.join(save_dir, '{}_{}.png'.format(index, obj_name))
        draw_subplot(pc, original_pc, fn)

def align():
    save_dir = '/scratch/dataset/3D_sketch_2021/sketch_pc_align'

    model_files = glob(os.path.join(pc_dir, '*.npy'))
    for index in range(len(model_files)):
        pc = torch.tensor(np.load(model_files[index]))
        pc = rotate_point_cloud(pc, dim='y', angle=-90)
        point_list = np.array(pc, dtype='float32')
        filename = os.path.basename(model_files[index]).split('.')[0]
        obj_name = filename.split('_')[0]

        save_path = os.path.join(save_dir, obj_name + '.npy')
        np.save(save_path, point_list)

def filter(index_list):
    filter_save_dir = '/scratch/dataset/3D_sketch_2021/filtered_view'
    pc_save_dir = '/scratch/dataset/3D_sketch_2021/sketch_pc_align'
    import open3d as o3d
    
    print("Radius oulier removal")
    for index in index_list:
        obj_name = query_name_list[index]
        pc_path = os.path.join(pc_dir, obj_name + '.npy')
        xyz = np.load(pc_path)
        xyz, center, scale = normalize(xyz) #[1024, 3]

        # pcd = o3d.io.read_point_cloud(pc_path, format='xyz')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        cl, ind = pcd.remove_radius_outlier(nb_points=400, radius=0.5)
        # new_pc = torch.tensor(np.asarray(cl.points)).d
        new_pc = xyz[ind]
        # fn = os.path.join(filter_save_dir, '{}_{}.png'.format(index, obj_name))
        # draw_subplot(torch.tensor(new_pc), torch.tensor(xyz), fn)
        save_path = os.path.join(pc_save_dir, obj_name + '.npy')
        np.save(save_path, new_pc)

def rotate(index_list):
    save_dir = '/scratch/dataset/3D_sketch_2021/sketch_pc_align'
    rotate_save_dir = '/scratch/dataset/3D_sketch_2021/selected_rotate'
    for index in index_list:
        obj_name = query_name_list[index]
        pc_path = os.path.join(pc_dir, obj_name + '.npy')
        pc = torch.tensor(np.load(pc_path))

        pc = rotate_point_cloud(pc, dim='z', angle=180)
        # original_pc_path = os.path.join(original_dir, obj_name + '.npy')
        # original_pc = torch.tensor(np.load(original_pc_path))
        # fn = os.path.join(rotate_save_dir, '{}_{}.png'.format(index, obj_name))
        # draw_subplot(torch.tensor(pc), torch.tensor(original_pc), fn)

        save_path = os.path.join(save_dir, obj_name + '.npy')
        point_list = np.array(pc, dtype='float32')
        np.save(save_path, point_list)

def vis_comparison():
    import cv2
    VIEW_DIR = '/vol/vssp/datasets/multiview/3VS/visualization/images/mitsuba_view'
    shape_dir = os.path.join(VIEW_DIR, 'shapenet')
    sketch_3d_dir = os.path.join(VIEW_DIR, 'chair_1005_align_view')
    new_sketch_3d_dir = '/scratch/dataset/3D_sketch_2021/mitsuba'
    save_dir = '/scratch/dataset/3D_sketch_2021/compare_old_new'
    obj_dir = '/scratch/dataset/3D_sketch_2021/obj'
    for index in range(len(query_name_list)):
        query_id = query_name_list[index]
        obj_file = glob(os.path.join(obj_dir, '{}_*.obj'.format(query_id)))[0]
        user_id = os.path.basename(obj_file).split('_')[2]
        shape_dx=20
        gt = cv2.imread(os.path.join(shape_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
        old_sketch = cv2.imread(os.path.join(sketch_3d_dir, '{}.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
        new_sketch = cv2.imread(os.path.join(new_sketch_3d_dir, '{}_00.jpg'.format(query_id)))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
        fig_3d = cv2.hconcat([gt, old_sketch, new_sketch])
        save_path = os.path.join(save_dir, '{}_{}_{}.png'.format(user_id, index, query_id))
        cv2.imwrite(save_path,fig_3d) 

def filter_outlier():
    image_dir = '/scratch/dataset/3D_sketch_2021/compare_old_new'
    obj_files = glob(os.path.join(image_dir, '10_*.png'))
    indexes = [os.path.basename(item).split('_')[1] for item in obj_files]
    filtered = 



if __name__ == "__main__":
    # index_list = [7, 10, 20, 35, 36, 45, 50, 93, 99, 100, 101, 112, 113, 115, 149, 161, 199, 185, 179, 174]
    # align()
    # filter(index_list)
    # rotate(index_list)
    # vis(index_list)
    vis_comparison()