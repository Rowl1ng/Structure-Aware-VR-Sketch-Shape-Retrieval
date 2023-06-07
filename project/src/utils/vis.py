import numpy as np
import torch

import matplotlib.pyplot as plt
import random
s = 1

def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    random.seed(30)
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 1)
        colors[i, 1] = random.randint(0, 1)
        colors[i, 2] = random.randint(0, 1)
    return colors
COLOR_LIST = create_color_list(5000)

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

title_font = {'fontname':'Arial', 'size':'8', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space

def draw_txt(fig, h, w, index, txt):
    plt.axis('off')

    ax = fig.add_subplot(h, w, index)
    # fig.patch.set_visible(False)
    ax.axis('off')
    ax.text(0.,0.5,txt,**title_font)
    # ax.set_title(txt)

def draw_subplot(fig, h, w, index, pc, txt, kp, s=s):
    pc = rotate_point_cloud(pc, dim='x')
    pc = rotate_point_cloud(pc, dim='z', angle=90)
    ax = fig.add_subplot(h, w, index, projection='3d')
    ax.scatter(pc[:, 0],pc[:, 1], pc[:, 2], c=pc[:, 2], s=s)
    if kp is not None:
        ax.scatter(kp[:, 0],kp[:, 1], kp[:, 2], c=COLOR_LIST[:kp.shape[0], :], s=3)
    plt.axis('off')

    ax.set_title(txt,**title_font)