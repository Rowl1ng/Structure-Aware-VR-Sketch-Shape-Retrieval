import os
import warnings
from glob import glob
from typing import Optional

import numpy as np
import pytorch3d.io
import torch

def vis(source_shapes, target_shapes, batch, outputs, save_dir_root):
    for b in range(source_shapes.shape[0]):
        source_shape = source_shapes[b]
        target_shape = target_shapes[b]
        data = split_batch(batch, b, singleton_keys=['cage_face'])
        output = split_batch(outputs, b, singleton_keys=['cage_face'])

        name = data['source_file'] + '_' + data['target_file']
        save_dir = os.path.join(save_dir_root, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save pointclouds
        save_pts_torch(os.path.join(save_dir, 'source_pointcloud.pts'), source_shape) 
        save_pts_torch(os.path.join(save_dir, 'target_pointcloud.pts'), target_shape) 
        if 'deformed' in output:
            save_pts_torch(os.path.join(save_dir, 'deformed_pointcloud.pts'), output['deformed']) # torch.Size([1024, 3])
        # save_pts(os.path.join(save_dir, 'target_pointcloud.pts'), target_shape)
        
        if 'target_mesh' in data:
            # save target mesh
            save_mesh(os.path.join(save_dir, 'target_mesh.obj'), data["target_mesh"], data["target_face"])

        if 'cage' in output:
            # save cages
            save_ply(os.path.join(save_dir, 'cage.ply'), output["cage"], output["cage_face"])
            save_ply(os.path.join(save_dir, 'deformed_cage.ply'), output["new_cage"], output["cage_face"])

        # save KP
        save_keypoints(os.path.join(save_dir, 'source_keypoints.txt'), output["source_keypoints"])
        save_keypoints(os.path.join(save_dir, 'target_keypoints.txt'), output["target_keypoints"])
        if 'deformed_keypoints' in output:
            save_keypoints(os.path.join(save_dir, 'deformed_keypoints.txt'), output["deformed_keypoints"])

def split_batch(data, b, singleton_keys=[]):
    return {k: v[b] if k not in singleton_keys else v[0] for k, v in data.items()}

def save_ply(f, verts, faces):
    pytorch3d.io.save_ply(f, verts.cpu(), faces=faces.cpu())

def save_pts_torch(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    save_pts(f, points.cpu().detach().numpy(), normals=normals)


def save_string_list(l, save_path):
    with open(save_path, 'w') as f:
        for item in l[:-1]:
            f.write("%s\n" % item)
        f.write("%s" % l[-1])

def read_keypoints(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    keypoints = []
    for line in lines:
        split = line.split(' ')
        keypoint = [float(x) for x in split[1:]]
        assert len(keypoint) == 3
        keypoints.append(keypoint)
    keypoints = np.array(keypoints, dtype=np.float32)
    return keypoints


def save_keypoints(file_path, keypoints):
    s = ''
    for i, keypoint in enumerate(keypoints):
        s += '%03d ' % i + ' '.join([str(float(x)) for x in keypoint]) + '\n'
    with open(file_path, 'w') as f:
        f.write(s)

 
def save_labelled_pointcloud(file_path, points, labels):
    colors = ['0 0 0', '0 1 0', '0 0 1', '1 0 0', '0.5 0.5 0.5', '0.5 0.5 0', '0.5 0 0']
    s = ''
    for point, label in zip(points, labels):
        s += ' '.join([str(float(x)) for x in point]) + ' ' + colors[label] + '\n'
    with open(file_path, 'w') as f:
        f.write(s)


def save_labeles(file_path, labels):
    with open(file_path, 'w') as f:
        f.write('\n'.join([str(x) for x in labels]))


def read_mesh(path, normal=False, return_mesh=False, load_textures=False):
    mesh = pytorch3d.io.load_objs_as_meshes([path], load_textures=load_textures)[0]
    vertices = mesh.verts_padded()[0]
    if normal:
        vertex_normals = mesh.verts_normals_padded()[0]
        vertices = torch.cat([vertices, vertex_normals], dim=-1)
    faces = mesh.faces_padded()[0]
    if return_mesh:
        return vertices, faces, mesh
    else:
        return vertices, faces
    

def _save_mesh(f, verts, faces, decimal_places: Optional[int] = None) -> None:
    """
    Faster version of https://pytorch3d.readthedocs.io/en/stable/_modules/pytorch3d/io/obj_io.html

    Adding .detach().numpy() to the input tensors makes it 10x faster
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    verts, faces = verts.cpu().detach().numpy(), faces.cpu().detach().numpy()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)
            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)


def save_mesh(f, verts, faces, decimal_places: Optional[int] = None):
    with open(f, 'w') as f: 
        _save_mesh(f, verts, faces, decimal_places)


def save_pts(filename, points, normals=None, labels=None):
    assert(points.ndim==2)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)
    if normals is not None:
        points = np.concatenate([points, normals], axis=1)
    if labels is not None:
        points = np.concatenate([points, labels], axis=1)
        np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1]+["\"%i\""])
    else:
        np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1])


def read_pcd(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = [x.split(' ') for x in lines]
    is_data = False
    points = []
    for line in lines:
        if line[0] == 'DATA':
            is_data = True
            continue
        if is_data:
            points += [line[:3]]
    return np.array(points, dtype=np.float32)


def find_files(source, file_ext=["txt",]):
    """
    From https://github.com/yifita/deep_cage
    """
    def is_type(file, file_ext):
        if isinstance(file_ext, str):
            file_ext = [file_ext]
        tmp = [os.path.splitext(file)[-1].lower()[1:] == ext for ext in file_ext]
        return any(tmp)

    # If file_ext is a list
    if source is None:
        return []
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source) or source[-1] == '*':
            if isinstance(file_ext, list):
                for fmt in file_ext:
                    source_fns += find_files(source, fmt)
            else:
                source_fns = sorted(glob("{}/**/*.{}".format(source, file_ext),recursive=True))
        elif os.path.isfile(source):
            source_fns = [source]
        assert (all([is_type(f, file_ext) for f in source_fns])), "Given files contain files with unsupported format"
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(find_files(s, file_ext=file_ext))
    return source_fns
