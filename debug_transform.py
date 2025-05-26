import numpy as np
from tqdm import tqdm
import math
import os
import torch
import time

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')


def transform_positions(positions, rot_mat, trans_mat):

    #positions shape : [frames x 3 x 1024]
    #affine matrix shape : [frames x  x 3]
    #translate shape: [frames x 1 x 3]

    rot_pos = torch.bmm(rot_mat, positions).transpose(1,2)

    trans_pos = rot_pos + trans_mat

    return trans_pos


def make_rot_mat_trans_mat(ref_pos, handle_ind, global_to_local=False):
    #positions shape: [frames x 1024 x 3]
    
    a = ref_pos[:, handle_ind[0], :] - ref_pos[:, handle_ind[1], :]
    a = a / torch.norm(a, dim=1, keepdim=True)
    #shape of a: [frames x 3]

    j = torch.tensor([0.0, 1.0, 0.0]).repeat(ref_pos.shape[0], 1).to(device)

    i = torch.cross(j, a, dim=1)

    k = torch.cross(i, j, dim=1)

    o = ref_pos[:, handle_ind[0], :]

    end_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(ref_pos.shape[0], 1, 1).to(device)
    
    aff_mat = torch.stack((i,j,k,o), 1).transpose(1,2)
    aff_mat = torch.cat((aff_mat, end_row), 1)
    #shape of aff_mat: [frames x 4 x 4]

    if global_to_local == True:
        glob2loc_mat = aff_mat

    aff_mat = torch.inverse(aff_mat)

    if global_to_local == False:
        
        return aff_mat[:, 0:3, 0:3], aff_mat[:, 0:3, 3].reshape(ref_pos.shape[0], 1, 3)

    else:

        return aff_mat[:, 0:3, 0:3], aff_mat[:, 0:3, 3].reshape(ref_pos.shape[0], 1, 3), glob2loc_mat[:, 0:3, 0:3], glob2loc_mat[:, 0:3, 3].reshape(ref_pos.shape[0], 1, 3)
    




def global2local_v2(handle_ind, ref, tar, mode, is_inverse=False):


    #print(tar.shape)
    #print(ref.shape)

    first = True

    for n in tqdm(range(tar.shape[0])):
        

        if n == 69:
            print("foo")
            
        p0 = ref[n][handle_ind[0]]
        p1 = ref[n][handle_ind[1]]

        #load the two vectors
        j = np.array([0.0,1.0,0.0])

        #get k hat
        a = p0 - p1
        k = a/np.linalg.norm(a)

        #get the cross product
        i = np.cross(j, k)
        if(math.isnan(i[0]) or math.isnan(i[1]) or math.isnan(i[2])):
            print("foo")

        #get origin
        o = p0

        #reshape
        i = i.reshape((-1,1))
        j = j.reshape((-1,1))
        k = k.reshape((-1,1))
        o = o.reshape((-1,1))

        #make the transformation matrix
        m = np.concatenate((i,j,k,o), axis=1)
        m = np.concatenate((m,[[0, 0, 0, 1]]), axis=0)
        
        if(is_inverse == False):    #meaning that the transformation would be from local to global
            m = np.linalg.inv(m)

        if(mode=="position"):    #meaning the target is a position based
            last_row_tar = np.ones((1, tar.shape[1]))
        elif(mode=="vector"):
            last_row_tar = np.zeros((1, tar.shape[1]))
        
        tar_frame = np.concatenate((tar[n].T, last_row_tar), axis=0)
        

        #solve the transformation
        tar_result = np.matmul(m, tar_frame)

        #remove row 4 and reshape
        tar_result = np.delete(tar_result, 3, 0).T.reshape((1, tar_result.shape[1], tar_result.shape[0] - 1))

        #append to total
        if(first):
            tar_total = tar_result
            first = False
        else:
            tar_total = np.concatenate((tar_total, tar_result), axis=0)

    return tar_total