import os
import numpy as np
import torch
from tqdm import tqdm
import math
import jstyleson
from torch.nn.functional import normalize
import enum
import random
from prep_run_step_config import get_run_step_config


import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

# class NodeType(enum.IntEnum):
#     NORMAL = 0
#     OBSTACLE = 1
#     # AIRFOIL = 2
#     # HANDLE = 3
#     # INFLOW = 4
#     # OUTFLOW = 5
#     # WALL_BOUNDARY = 6
#     SIZE = 2

handle_label = None


def set_handle_label(is_new_labels):
    global handle_label
    if is_new_labels == False:
        handle_label = 1
    else:
        handle_label = 0
        
def get_handle_label():
    assert(handle_label is not None)
    return handle_label


def triangles_to_edges(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        
        
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges= torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)


        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}

def triangles_to_edges_with_adjf(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        
        opposite_v = torch.cat((faces[:,2], faces[:,0], faces[:,1]), dim=0)
        
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges, inverse_indices = torch.unique(packed_edges, return_inverse=True, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        opposite_v_unique = torch.ones((unique_edges.shape[0], 2)) * -1
        opposite_v_unique = opposite_v_unique.to(torch.int64)

        for idx, inv_ind in enumerate(inverse_indices):
            ivi = inv_ind.item()
            if(opposite_v_unique[ivi,0] == -1):
                opposite_v_unique[ivi,0] = opposite_v[idx]
            else:
                opposite_v_unique[ivi,1] = opposite_v[idx]

        assert(opposite_v_unique.shape[0] == unique_edges.shape[0])

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0), torch.cat((opposite_v_unique, opposite_v_unique), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    
def obj_to_mesh_pos(filename):
    mesh_space = []

    with open(filename, 'r') as f:
        for line in f:
            reading = np.array(line.strip().split(" "))
            if (reading[0] == 'vt'):
                x = reading[1].astype(np.float32)
                y = reading[2].astype(np.float32)
                mesh_space.append([x,y])
    
    return np.array(mesh_space).astype(np.float32)

def obj_to_real_pos(filename):
    mesh_space = []

    with open(filename, 'r') as f:
        for line in f:
            reading = np.array(line.strip().split(" "))
            if (reading[0] == 'v'):
                x = reading[1].astype(np.float32)
                y = reading[2].astype(np.float32)
                z = reading[3].astype(np.float32)
                mesh_space.append([x,y,z])
    
    return np.array(mesh_space).astype(np.float32)

def make_global_feature(gv_path):
    gv_npy = np.load(gv_path)
    global_gv = (gv_npy[:,960] + gv_npy[:,930])/2
    return global_gv

def npy2obj_single(garment_f, garment_pos, output_path, uv_coordinates=None):
    
    obj_pos = garment_pos

    obj_name = output_path
    with open(obj_name, 'w') as file:
        for j in range(obj_pos.shape[0]):
            file.write('v %f %f %f\n' % (obj_pos[j][0], obj_pos[j][1], obj_pos[j][2]))
        if uv_coordinates is not None:
            for l in range(uv_coordinates.shape[0]):
                file.write('vt %f %f\n' %(uv_coordinates[l][0], uv_coordinates[l][1]))
        for k in range(len(garment_f)):
            file.write('f %s %s %s\n' % (garment_f[k][0]+1, garment_f[k][1]+1, garment_f[k][2]+1))

def npy2obj_special(garment_f, garment_pos, output_path, uv_coordinates, vn):
    
    obj_pos = garment_pos

    obj_name = output_path
    with open(obj_name, 'w') as file:
        for j in range(obj_pos.shape[0]):
            file.write('v %f %f %f\n' % (obj_pos[j][0], obj_pos[j][1], obj_pos[j][2]))
        
        for l in range(uv_coordinates.shape[0]):
            file.write('vt %f %f\n' %(uv_coordinates[l][0], uv_coordinates[l][1]))
        for m in vn:
            file.write(m)
        for n in garment_f:
            file.write(n)

def npy2obj_single(garment_f, garment_pos, output_path, uv_coordinates=None):

    with open(output_path, 'w') as file:
        for pos in garment_pos:
            file.write('v %f %f %f\n' % (pos[0], pos[1], pos[2]))
        if uv_coordinates is not None:
            for uv_pos in uv_coordinates:
                file.write('vt %f %f\n' %(uv_pos[0], uv_pos[1]))
        for face in garment_f:
            file.write('f %s %s %s\n' % (face[0]+1, face[1]+1, face[2]+1))

def npy2obj_v2(garment_f, garment_pos, output_path, uv_coordinates=None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n in tqdm(range(garment_pos.shape[0])):
        obj_pos = garment_pos[n]

        obj_name = output_path + '/' + 'obj_%06d.obj' %(n) 
        with open(obj_name, 'w') as file:
            for j in range(obj_pos.shape[0]):
                file.write('v %f %f %f\n' % (obj_pos[j][0], obj_pos[j][1], obj_pos[j][2]))
            if uv_coordinates is not None:
                for l in range(uv_coordinates.shape[0]):
                    file.write('vt %f %f\n' %(uv_coordinates[l][0], uv_coordinates[l][1]))
            for k in range(len(garment_f)):
                file.write('f %s %s %s\n' % (garment_f[k][0]+1, garment_f[k][1]+1, garment_f[k][2]+1))

def npy2obj(ref_obj_path, tar_npy, output_path):
    #extract face information
    garment_f = []
    with open(ref_obj_path, "rb") as f:
        for line in f:
            reading2 = np.array(line.decode("utf-8").strip().split(" "))
            if (reading2[0] == 'f'):
                garment_f.append([reading2[1],reading2[2],reading2[3]])

    #print(len(garment_f))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for n in tqdm(range(tar_npy.shape[0])):
        obj_pos = tar_npy[n]

        obj_name = output_path + '/' + 'obj_%06d.obj' %(n) 
        with open(obj_name, 'w') as file:
            for j in range(obj_pos.shape[0]):
                file.write('v %f %f %f\n' % (obj_pos[j][0], obj_pos[j][1], obj_pos[j][2]))
            for k in range(len(garment_f)):
                file.write('f %s %s %s\n' % (garment_f[k][0], garment_f[k][1], garment_f[k][2]))

def get_local_gp(folder_path, frame_steps, frame_cut, handle_ind, frame_time):
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))

    #convert steps to frame
    gp = steps2frame(frame_steps, gp_npy, "skip")

    gp = cut_npy(frame_cut, gp)

    gp_local = global2local_v2(handle_ind, gp, gp, "position")

    handle_v = get_handle_v(handle_ind, gp, frame_time)

    gp_local = torch.from_numpy(gp_local.astype(np.float32))
    handle_v = torch.from_numpy(handle_v.astype(np.float32))

    return gp_local, handle_v, gp

def get_global_gp(gp_npy, frame_steps, frame_cut, last_frame_cut):

    #convert steps to frame
    gp = steps2frame(frame_steps, gp_npy, "skip")

    gp = cut_npy(frame_cut, gp)

    gp = last_cut_npy(last_frame_cut, gp)

    gp = torch.from_numpy(gp.astype(np.float32))
    
    return gp

def get_global_forces(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time):
    
    fext_npy = np.load(os.path.join(folder_path, "fext.npy"))
    bf_npy = np.load(os.path.join(folder_path, "bf.npy"))
    sf_npy = np.load(os.path.join(folder_path, "sf.npy"))
    
    #convert steps to frame
    
    fext = steps2frame(frame_steps, fext_npy, "skip")
    sf = steps2frame(frame_steps, sf_npy, "skip")
    bf = steps2frame(frame_steps, bf_npy, "skip")

    assert(fext.shape==sf.shape==bf.shape)

    fext = cut_npy(frame_cut, fext)
    sf = cut_npy(frame_cut, sf)
    bf = cut_npy(frame_cut, bf)

    assert(fext.shape==sf.shape==bf.shape)

    fext = last_cut_npy(last_frame_cut, fext)
    sf = last_cut_npy(last_frame_cut, sf)
    bf = last_cut_npy(last_frame_cut, bf)

    assert(fext.shape==sf.shape==bf.shape)

    fext = torch.from_numpy(fext.astype(np.float32))
    sf = torch.from_numpy(sf.astype(np.float32))
    bf = torch.from_numpy(bf.astype(np.float32))
    
    return fext, sf, bf

def get_local_gp(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time):
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))
    
    #convert steps to frame
    gp = steps2frame(frame_steps, gp_npy, "skip")

    gp = cut_npy(frame_cut, gp)

    gp = last_cut_npy(last_frame_cut, gp)


    #global to local post processing
    gp_local = global2local_v2(handle_ind, gp, gp, "position")

    handle_v = get_handle_v(handle_ind, gp, frame_time)

    gp_local = torch.from_numpy(gp_local.astype(np.float32))
    handle_v = torch.from_numpy(handle_v.astype(np.float32))


    return gp_local, handle_v

def get_local_force(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time):
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))
    fext_npy = np.load(os.path.join(folder_path, "fext.npy"))
    bf_npy = np.load(os.path.join(folder_path, "bf.npy"))
    sf_npy = np.load(os.path.join(folder_path, "sf.npy"))
    
    #convert steps to frame
    fext = steps2frame(frame_steps, fext_npy, "skip")
    sf = steps2frame(frame_steps, sf_npy, "skip")
    bf = steps2frame(frame_steps, bf_npy, "skip")
    gp = steps2frame(frame_steps, gp_npy, "skip")

    assert(fext.shape==sf.shape==bf.shape==gp.shape)

    fext = cut_npy(frame_cut, fext)
    sf = cut_npy(frame_cut, sf)
    bf = cut_npy(frame_cut, bf)
    gp = cut_npy(frame_cut, gp)

    assert(fext.shape==sf.shape==bf.shape==gp.shape)

    fext = last_cut_npy(last_frame_cut, fext)
    sf = last_cut_npy(last_frame_cut, sf)
    bf = last_cut_npy(last_frame_cut, bf)
    gp = last_cut_npy(last_frame_cut, gp)

    assert(fext.shape==sf.shape==bf.shape==gp.shape)

    #global to local post processing
    fext_local = global2local_v2(handle_ind, gp, fext, "vector")
    sf_local = global2local_v2(handle_ind, gp, sf, "vector")
    bf_local = global2local_v2(handle_ind, gp, bf, "vector")


    fext_local = torch.from_numpy(fext_local.astype(np.float32))
    sf_local = torch.from_numpy(sf_local.astype(np.float32))
    bf_local = torch.from_numpy(bf_local.astype(np.float32))



    return fext_local, sf_local, bf_local


def get_local_gp_force(folder_path, frame_steps, frame_cut, handle_ind, frame_time):
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))
    fext_npy = np.load(os.path.join(folder_path, "fext.npy"))
    bf_npy = np.load(os.path.join(folder_path, "bf.npy"))
    sf_npy = np.load(os.path.join(folder_path, "sf.npy"))
    
    #convert steps to frame
    fext = steps2frame(frame_steps, fext_npy, "skip")
    sf = steps2frame(frame_steps, sf_npy, "skip")
    bf = steps2frame(frame_steps, bf_npy, "skip")
    gp = steps2frame(frame_steps, gp_npy, "skip")

    assert(fext.shape==sf.shape==bf.shape==gp.shape)

    fext = cut_npy(frame_cut, fext)
    sf = cut_npy(frame_cut, sf)
    bf = cut_npy(frame_cut, bf)
    gp = cut_npy(frame_cut, gp)

    assert(fext.shape==sf.shape==bf.shape==gp.shape)

    #global to local post processing
    fext_local = global2local_v2(handle_ind, gp, fext, "vector")
    sf_local = global2local_v2(handle_ind, gp, sf, "vector")
    bf_local = global2local_v2(handle_ind, gp, bf, "vector")
    gp_local = global2local_v2(handle_ind, gp, gp, "position")

    handle_v = get_handle_v(handle_ind, gp, frame_time)

    fext_local = torch.from_numpy(fext_local.astype(np.float32))
    sf_local = torch.from_numpy(sf_local.astype(np.float32))
    bf_local = torch.from_numpy(bf_local.astype(np.float32))
    gp_local = torch.from_numpy(gp_local.astype(np.float32))
    handle_v = torch.from_numpy(handle_v.astype(np.float32))


    return fext_local, sf_local, bf_local, gp_local, handle_v

def get_face_info(folder_path, num_frames):
    filename = os.path.join(folder_path, "cloth.obj")

    face_data = obj_to_face_data(filename)

    face_data = torch.from_numpy(face_data.astype(np.int32))

    #repeat face_data for num_frames
    return face_data.repeat([num_frames,1,1])

def get_mesh_space(filename, num_frames, mesh_pos_mode):

    if mesh_pos_mode == "2d":
        mesh_pos_data = obj_to_mesh_pos(filename)
    elif mesh_pos_mode == "3d":
        mesh_pos_data = obj_to_real_pos(filename)

    mesh_pos_data = torch.from_numpy(mesh_pos_data.astype(np.float32))

    #repeat face_data for num_frames
    return mesh_pos_data.repeat([num_frames,1,1])

def obj_to_face_data(filename):
    face_data = []

    with open(filename, 'r') as f:
        for line in f:
            reading = np.array(line.strip().split(" "))
            if (reading[0] == 'f'):
                f_indices = []
                for n in range(3):
                    face_ind = np.array(reading[n+1].strip().split("/"))

                    face_ind_int = face_ind[0].astype(int) - 1
                    
                    f_indices.append(face_ind_int)
                face_data.append(f_indices)
    
    return np.array(face_data).astype(np.int64)

def obj_to_face_data_v2(filename):
    face_data = []

    with open(filename, 'r') as f:
        for line in f:
            reading = np.array(line.strip().split(" "))
            if (reading[0] == 'f'):
                face_data.append(line)
    
    return face_data

def obj_to_vn(filename):
    face_data = []

    with open(filename, 'r') as f:
        for line in f:
            reading = np.array(line.strip().split(" "))
            if (reading[0] == 'vn'):
                face_data.append(line)
    
    return face_data
    
def get_node_type(handle_ind, num_frames, num_nodes):
    node_type = np.zeros((num_nodes, 1))
    for handle in handle_ind:
        node_type[handle] = 1

    node_type = torch.from_numpy(node_type.astype(np.int32))

    return node_type.repeat([num_frames, 1, 1])

def get_handle_v(handle_ind, gp, frame_time):
    all_v_global_gv, _ = make_new_gv_dv(gp, frame_time)
    all_v_local_gv = global2local_v2(handle_ind, gp, all_v_global_gv, "vector")

    handle_gv_global = (all_v_global_gv[:,handle_ind[0]] + all_v_global_gv[:,handle_ind[1]])/2
    handle_gv_local = (all_v_local_gv[:,handle_ind[0]] + all_v_local_gv[:,handle_ind[1]])/2

    

    return handle_gv_local

def make_new_gv_dv(g_pos, frame_time):

    new_dv = np.zeros((g_pos.shape))
    new_v = np.zeros(g_pos.shape)
    
    for frame in range(new_v.shape[0]):
        if(frame == 0):
            new_v[frame] = 0
        else:
            new_v[frame] = (g_pos[frame] - g_pos[frame - 1])/frame_time
    
    for frame in range(new_dv.shape[0] - 1):
        new_dv[frame] = new_v[frame + 1] - new_v[frame]
    
    return new_v, new_dv

def extract_handle_v(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time):
    gp_local, handle_v= get_local_gp(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time)
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))
    gp_global = get_global_gp(gp_npy, frame_steps, frame_cut, last_frame_cut)
    save_tensor_to_txt(gp_global[:,0,:], os.path.join(folder_path, "debug_gp.txt"))
    save_tensor_to_txt(handle_v, os.path.join(folder_path, "debug_handle_v.txt"))

def make_pkl_file_v2(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time, output_folder, with_forces, with_theta, mesh_pos_mode):
    #this is for the "global" data
    gp_npy = np.load(os.path.join(folder_path, "g_p.npy"))
    gp = get_global_gp(gp_npy, frame_steps, frame_cut, last_frame_cut)
    
    if with_forces:
        fext, sf, bf = get_global_forces(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time)

    #cut the first frame
    gp = gp[1:gp.shape[0]]
    if with_forces:
        fext = fext[1:fext.shape[0]]
        sf = sf[1:sf.shape[0]]
        bf = bf[1:bf.shape[0]]

    face_info = get_face_info(folder_path, gp.shape[0])

    #mesh_pos_info = get_mesh_space(os.path.join(folder_path, "cloth.obj"), gp.shape[0], mesh_pos_mode)

    node_type = get_node_type(handle_ind, gp.shape[0], gp.shape[1])

    ##--get theta--##
    
    decomposed_face = triangles_to_edges_with_adjf(face_info[0])
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']
    ##---get senders and receivers--##
    full_senders = senders.repeat([gp.shape[0], 1])
    full_receivers = receivers.repeat([gp.shape[0], 1])
    ##------------------------------##
    if with_theta:
        full_dihedral_angle = []
        for frame in tqdm(range(gp.shape[0])):
            relative_cloth_pos = (torch.index_select(input=gp[frame], dim=0, index=senders) -
                                torch.index_select(input=gp[frame], dim=0, index=receivers))
        
            dihedral_angle = compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, gp[frame])
            full_dihedral_angle.append(dihedral_angle)
        dihedral_angle_stack = torch.stack(full_dihedral_angle, dim=0)
    
    ##-------------##
    
    npy_dict = {}
    npy_dict['cloth_pos'] = gp
    if with_forces:
        npy_dict['fext'] = fext
        npy_dict['sf'] = sf
        npy_dict['bf'] = bf
    npy_dict['face'] = face_info
    #npy_dict['mesh_pos'] = mesh_pos_info
    npy_dict['node_type'] = node_type
    if with_theta:
        npy_dict['dihedral_angle'] = dihedral_angle_stack
    npy_dict['senders'] = full_senders
    npy_dict['receivers'] = full_receivers

    np.save(os.path.join(output_folder, "full.npy"), npy_dict)

def make_pkl_file(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time, output_folder, with_forces, with_theta, mesh_pos_mode):
    
    gp, handle_v = get_local_gp(folder_path, frame_steps, frame_cut, last_frame_cut,  handle_ind, frame_time)
    
    if with_forces: 
        fext, sf, bf = get_local_force(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time)

    #cut the first frame
    if with_forces:
        fext = fext[1:fext.shape[0]]
        sf = sf[1:sf.shape[0]]
        bf = bf[1:bf.shape[0]]

    gp = gp[1:gp.shape[0]]
    handle_v = handle_v[1:handle_v.shape[0]]

    face_info = get_face_info(folder_path, gp.shape[0])

    
    mesh_pos_info = get_mesh_space(os.path.join(folder_path, "cloth.obj"), gp.shape[0], mesh_pos_mode)
    

    node_type = get_node_type(handle_ind, gp.shape[0], gp.shape[1])

    ##--get theta--##

    decomposed_face = triangles_to_edges_with_adjf(face_info[0])
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']
    ##---get senders and receivers--##
    full_senders = senders.repeat([gp.shape[0], 1])
    full_receivers = receivers.repeat([gp.shape[0], 1])
    ##------------------------------##
    if with_theta:
        full_dihedral_angle = []
        for frame in tqdm(range(gp.shape[0])):
            relative_cloth_pos = (torch.index_select(input=gp[frame], dim=0, index=senders) -
                                torch.index_select(input=gp[frame], dim=0, index=receivers))
        
            dihedral_angle = compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, gp[frame])
            full_dihedral_angle.append(dihedral_angle)
        dihedral_angle_stack = torch.stack(full_dihedral_angle, dim=0)
        
    ##-------------##
    
    npy_dict = {}
    npy_dict['cloth_pos'] = gp
    if with_forces:
        npy_dict['fext'] = fext
        npy_dict['sf'] = sf
        npy_dict['bf'] = bf
    npy_dict['face'] = face_info
    npy_dict['mesh_pos'] = mesh_pos_info
    npy_dict['node_type'] = node_type
    npy_dict['handle_v'] = handle_v
    if with_theta:
        npy_dict['dihedral_angle'] = dihedral_angle_stack
    npy_dict['senders'] = full_senders
    npy_dict['receivers'] = full_receivers

    np.save(os.path.join(output_folder, "full.npy"), npy_dict)

def cut_npy(frame_cut, data):
    return data[frame_cut:data.shape[0]]

def last_cut_npy(frame_cut, data):
    return data[0:data.shape[0] - frame_cut]

def global2local_v2(handle_ind, ref, tar, mode, is_inverse=False):


    #print(tar.shape)
    #print(ref.shape)

    first = True

    for n in tqdm(range(tar.shape[0])):
        

        
        p0 = ref[n][handle_ind[0]]
        p1 = ref[n][handle_ind[1]]
        p2 = np.array([p0[0]+0.5, (p0[1]+p1[1])/2, (p0[2]+p1[2])/2])

        #load the two vectors
        a = p1 - p0
        b = p2 - p0

        #get the cross product
        j = np.cross(a, b)
        j = j/np.linalg.norm(j)
        if(math.isnan(j[0]) or math.isnan(j[1]) or math.isnan(j[2])):
            print("foo")

        #get i hat
        i = a/np.linalg.norm(a)

        #get k hat
        k = np.cross(i,j)

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

def steps2frame(frame_steps, steps_data, mode):
    tot_steps = steps_data.shape[0]
    assert(tot_steps%frame_steps == 0)
    tot_frames = int(tot_steps/frame_steps)
    tot_verts = steps_data.shape[1]

    result = np.zeros((tot_frames, tot_verts, steps_data.shape[2]))

    frame = -1
    for step in range(tot_steps):
        if(step % frame_steps == 0):
            frame = frame + 1

        if(mode == "average"):
            result[frame] = result[frame] + (steps_data[step]/frame_steps)
        elif(mode == "sum"):
            result[frame] = result[frame] + steps_data[step]
        elif(mode == "skip"):
            if(step % frame_steps == 0):
                result[frame] = steps_data[step]
    
    #print(result.shape)
    
    #np.save(npy_name + "_frame_" + mode + ".npy", result)
    return result

def make_train_test_pkl(folder_path, train_test_ratio, output_folder, mode):
    full_pkl = np.load(os.path.join(folder_path, "full.npy"),
                allow_pickle=True)
    full_pkl = full_pkl.tolist()
    tot_frames = full_pkl['cloth_pos'].shape[0]

    if mode == 'old':
        train_frames = int(tot_frames * train_test_ratio)
        test_frames = tot_frames - train_frames

        train_pkl = {}
        test_pkl = {}
        for key, val in full_pkl.items():
            assert(full_pkl[key].shape[0] == tot_frames)

            train_pkl[key] = val[0:train_frames]
            test_pkl[key] = val[train_frames:tot_frames]

            assert(train_pkl[key].shape[0] == train_frames)
            assert(test_pkl[key].shape[0] == test_frames)

        np.save(os.path.join(output_folder, "train.npy"), train_pkl)
        np.save(os.path.join(output_folder, "test.npy"), test_pkl)
    elif mode == 'dataset_v2':
        train_frames = tot_frames
        test_frames = int(tot_frames * (1-train_test_ratio))

        train_pkl = {}
        test_pkl = {}
        for key, val in full_pkl.items():
            assert(full_pkl[key].shape[0] == tot_frames)

            train_pkl[key] = val[0:train_frames]
            test_pkl[key] = val[240:240+test_frames]

            assert(train_pkl[key].shape[0] == train_frames)
            assert(test_pkl[key].shape[0] == test_frames)

        np.save(os.path.join(output_folder, "train.npy"), train_pkl)
        np.save(os.path.join(output_folder, "test.npy"), test_pkl)

def get_folder_name_from_json_file(json_file):
    json_params = read_json_file(json_file)
    return json_params['output_folder']

def read_json_file(json_file_name, mode):
    
    json_file = os.path.join("params", json_file_name + ".json")# FLAGS.params)
    params_parent = JsonParse(json_file)
    json_params = params_parent.params

    #overtake the name code with the filename
    json_params['name_code'] = json_file_name

    run_step_config = {}
    run_step_config = get_run_step_config(run_step_config, json_params, mode)
    
    return run_step_config

class JsonParse():

    def __init__(self, json_path):
        self.params = jstyleson.load(open(json_path))
    
    def parse(self, label, default=None):
        try:
            return self.params[label]
        except:
            if default == None:
                raise Exception("Cannot find \"%s\" in json file. You can define a default value instead." %(label))
            else:
                return default

    def parse_cat(self, category, label, default=None):
        try:
            return self.params[category][label]
        except:
            if default == None:
                raise Exception("Cannot find \"%s\\%s\" in json file. You can define a default value instead." %(category, label))
            else:
                return default

def compute_normal_face(face):
    v0 = face[0]
    v1 = face[1]
    v2 = face[2]
    return normalize(torch.cross(v1 - v0, v2-v0), dim=0)

def compute_dihedral_angle_fast(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos, mode):
    norm_rel_cloth_pos = normalize(relative_cloth_pos, dim=1)


    mask_1 = ~torch.eq(opposite_v[:,1], torch.tensor([-1]).to(device))
    #mask_2 = torch.eq((norm_rel_cloth_pos*norm_rel_cloth_pos).sum(1), torch.tensor([0.]))

    opposite_v_ind_1 = torch.where(mask_1, opposite_v[:,1], opposite_v[:,0])

    cross_p_face_0 = torch.cross(torch.index_select(cloth_pos, 0, receivers) -
                        torch.index_select(cloth_pos, 0, senders), 
                        torch.index_select(cloth_pos, 0, opposite_v[:,0]) - 
                        torch.index_select(cloth_pos, 0, senders))
    
    # cross_p_face_1 = torch.where(mask_1, torch.cross(torch.index_select(cloth_pos, 0, receivers) -
    #                     torch.index_select(cloth_pos, 0, senders), 
    #                     torch.index_select(cloth_pos, 0, opposite_v_ind_1) - 
    #                     torch.index_select(cloth_pos, 0, senders)), 
    #                     torch.zeros((senders.shape[0], cloth_pos.shape[1])))

    cross_p_face_1 = torch.cross(torch.index_select(cloth_pos, 0, receivers) -
                        torch.index_select(cloth_pos, 0, senders), 
                        torch.index_select(cloth_pos, 0, opposite_v_ind_1) - 
                        torch.index_select(cloth_pos, 0, senders))
    
    norm_0 = normalize(cross_p_face_0)
    norm_1 = normalize(cross_p_face_1)

    cosine = (norm_0 * norm_1).sum(1)
    sine = (norm_rel_cloth_pos * torch.cross(norm_0, norm_1)).sum(1)
    arctan = torch.atan2(sine, cosine)
    if mode == "unsigned":
        theta = math.pi - torch.abs(arctan)
    elif mode == "signed":
        theta = torch.where(arctan >= 0, math.pi - torch.abs(arctan), torch.abs(arctan) - math.pi)

    dihedral_angle = torch.where(mask_1, theta, torch.zeros(senders.shape).to(device))

    
    return dihedral_angle

def compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos):
    norm_rel_cloth_pos = normalize(relative_cloth_pos, dim=1)
    dihedral_angles = []
    #debug_idx = 0
    for sender, receiver, opposite_v_single, norm_rel_single in zip(senders, receivers, opposite_v, norm_rel_cloth_pos):
        #check if the edge have two adjacent v
        #debug_idx = debug_idx + 1
        if opposite_v_single[1] == -1:
            dihedral_angle = torch.tensor([0.]).to(device)
            dihedral_angles.append(dihedral_angle)
            continue
        if torch.dot(norm_rel_single, norm_rel_single) == 0:
            dihedral_angle = torch.tensor([0.]).to(device)
            dihedral_angles.append(dihedral_angle)
            continue
        
        face_0 = (cloth_pos[sender], cloth_pos[receiver], cloth_pos[opposite_v_single[0]])
        face_1 = (cloth_pos[sender], cloth_pos[receiver], cloth_pos[opposite_v_single[1]])
        norm_0 = compute_normal_face(face_0)
        norm_1 = compute_normal_face(face_1)
        if torch.dot(norm_0, norm_0)==0 or torch.dot(norm_1, norm_1)==0:
            dihedral_angle = torch.tensor([0.]).to(device)
            dihedral_angles.append(dihedral_angle)
            continue
        cosine = torch.dot(norm_0, norm_1)
        sine = torch.dot(norm_rel_single, torch.cross(norm_0, norm_1))
        dihedral_angle = torch.atan2(sine, cosine).reshape(1)
        dihedral_angles.append(dihedral_angle)

        #if sender == 360 and receiver == 111:
        #    print("foo")

        
        
        #print(sender, receiver, opposite_v_single, norm_rel_single)
    return torch.cat(dihedral_angles, dim=0).to(device)
def edit_dihedral_angle(dihedral_angle):
    #data = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
    #                            "full.npy"), allow_pickle=True).tolist()
    #dihedral_angle = data['dihedral_angle']
    first_half = dihedral_angle[:, 0:int(dihedral_angle.shape[1]/2)]
    
    new_da = torch.cat((first_half, first_half*-1), dim=1)
    #data['dihedral_angle'] = new_da
    #np.save(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
    #                    "full.npy"), data)
    return new_da
    
def debug_dihedral_angle_symmetry():
    for file in ["full.npy", "train.npy", "test.npy"]:
        dihedral_angle = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
                                    file), allow_pickle=True).tolist()['dihedral_angle']
        
        first_half = dihedral_angle[:, 0:int(dihedral_angle.shape[1]/2)]
        second_half = dihedral_angle[:, int(dihedral_angle.shape[1]/2):int(dihedral_angle.shape[1])]
        assert(torch.equal(first_half, second_half*-1)==True)

def debug_dihedral_angle_local_global():
    data = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
                    "full.npy"), allow_pickle=True).tolist()
    dihedral_angle_local = data['dihedral_angle']

    dihedral_angle_global = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
                    "debug_dihedral_global", "dihedral.npy"))
    dihedral_angle_global = torch.from_numpy(dihedral_angle_global.astype(np.float32))
    dihedral_angle_global = edit_dihedral_angle(dihedral_angle_global)

    '''
    frame_steps = 8
    frame_cut = 240
    global_gp = np.load(os.path.join("input", "raw_npy", "mesh_1024", "g_p.npy"))
    gp = steps2frame(frame_steps, global_gp, "skip")
    gp = cut_npy(frame_cut, gp)
    gp = torch.from_numpy(gp.astype(np.float32))

    face_info = data['face']
    fext = data['fext']
    gp = gp[1:gp.shape[0]]
    assert(fext.shape == gp.shape)
    ##--get theta--##

    decomposed_face = triangles_to_edges_with_adjf(face_info[0])
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']
    ##------------------------------##
    full_dihedral_angle = []
    for frame in tqdm(range(gp.shape[0])):
        relative_cloth_pos = (torch.index_select(input=gp[frame], dim=0, index=senders) -
                              torch.index_select(input=gp[frame], dim=0, index=receivers))
    
        dihedral_angle = compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, gp[frame])
        full_dihedral_angle.append(dihedral_angle)
    dihedral_angle_stack = torch.stack(full_dihedral_angle, dim=0)
    ##-------------##
    np.save(os.path.join("input", "processed_npy", "mesh_1024_with_theta", "debug_dihedral_global", "dihedral.npy"), 
                dihedral_angle_stack)
    '''
    for frame in range(dihedral_angle_global.shape[0]):
        for edge in range(dihedral_angle_local.shape[0]):
            #if(dihedral_angle_global[frame][edge] != dihedral_angle_local[frame][edge]):
            #    print("foo")
            if(torch.allclose(dihedral_angle_global[frame][edge], dihedral_angle_local[frame][edge]) == False):
                print("foo")
    assert(torch.equal(dihedral_angle_local, dihedral_angle_global))
    print("foo")

def get_len_zero_frame(num_frames, senders, receivers, obj_filename, mesh_pos_mode):
    if mesh_pos_mode == "3d":
        numpy_zero_frame = obj_to_real_pos(obj_filename)
    elif mesh_pos_mode == "2d":
        numpy_zero_frame = obj_to_mesh_pos(obj_filename)

    zero_frame_cloth_pos = torch.from_numpy(numpy_zero_frame)
    rel_cloth_pos = (torch.index_select(zero_frame_cloth_pos, 0, senders) -
                                torch.index_select(zero_frame_cloth_pos, 0, receivers))
    len_zero_frame = torch.norm(rel_cloth_pos, dim=-1, keepdim=True)

    return len_zero_frame.repeat([num_frames,1,1])

def debug_check_length():
    
    #get data of the mesh space
    data_mesh = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta",
                                    "full.npy"), allow_pickle=True).tolist()
    mesh_pos = data_mesh['mesh_pos'][0]
    senders = data_mesh['senders'][0]
    receivers = data_mesh['receivers'][0]
    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                torch.index_select(mesh_pos, 0, receivers))
    len_mesh_space = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)

    #get data of 0th frame
    obj_filename = os.path.join("input","raw_npy","mesh_1024","cloth.obj")
    zero_frame_cloth_pos = torch.from_numpy(obj_to_real_pos(obj_filename))
    rel_cloth_pos = (torch.index_select(zero_frame_cloth_pos, 0, senders) -
                                torch.index_select(zero_frame_cloth_pos, 0, receivers))
    len_zero_frame = torch.norm(rel_cloth_pos, dim=-1, keepdim=True)
    index = 0
    max_dif = -1
    max_dif_edge = -1
    for edge in range(len_mesh_space.shape[0]):
        if torch.allclose(len_mesh_space[edge], len_zero_frame[edge]) == False:
            dif_len = len_mesh_space[edge] - len_zero_frame[edge]
            print(edge, len_mesh_space[edge], len_zero_frame[edge])
            index = index + 1
            if dif_len > max_dif:
                max_dif = dif_len
                max_dif_edge = edge
            #print("foo")
    print("total not equal edge: ")
    print("max dif, max dif edge:  ", max_dif, max_dif_edge)


    print("foo")

def debug_npy2obj(face, tar_npy, folder_name):
    #extract face information
    garment_f = face
   
    #print(len(garment_f))

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for n in tqdm(range(tar_npy.shape[0])):
        obj_pos = tar_npy[n]

        obj_name = folder_name + '/' + 'obj_%06d.obj' %(n) 
        with open(obj_name, 'w') as file:
            for j in range(obj_pos.shape[0]):
                file.write('v %f %f %f\n' % (obj_pos[j][0], obj_pos[j][1], obj_pos[j][2]))
            for k in range(garment_f.shape[0]):
                file.write('f %s %s %s\n' % (str(int(garment_f[k][0].item())), str(int(garment_f[k][1].item())), str(int(garment_f[k][2].item()))))

def debug_analyze_data(trajectory, filepath):
    fields = ['cloth_pos']
    #node features
    #velocity
    out = {}
    for key, val in trajectory.items():
        out[key] = val[1:-1]
        if key in fields:
            out['prev|' + key] = val[:-2]
            out['target|' + key] = val[2:]
    trajectory = out

    cloth_pos = trajectory['cloth_pos']
    prev_cloth_pos = trajectory['prev|cloth_pos']

    velocity = cloth_pos - prev_cloth_pos

    face = trajectory['face'][0]
    decomposed_face = triangles_to_edges(face)
    senders, receivers = decomposed_face['two_way_connectivity']

    relative_cloth_pos = (torch.index_select(input=cloth_pos, dim=0, index=senders) -
                              torch.index_select(input=cloth_pos, dim=0, index=receivers))

    len_real_pos = torch.norm(relative_cloth_pos, dim=-1, keepdim=True)

    mesh_pos = trajectory['mesh_pos']

    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                torch.index_select(mesh_pos, 0, receivers))
    
    len_mesh_pos = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)


    result = {}
    result['velocity'] = velocity
    result['rel_cloth_pos'] = relative_cloth_pos
    result['len_real_pos'] = len_real_pos
    result['rel_mesh_pos'] = relative_mesh_pos
    result['len_mesh_pos'] = len_mesh_pos

    np.save(filepath, result)


    #edge features
    #relative mesh position
    #relative world position

def accel2pos(acceleration, inputs, model, use_fps, fps):
    cloth_pos = inputs['cloth_pos']
    prev_cloth_pos = inputs['prev|cloth_pos'][0]
    target_cloth_pos = inputs['target|cloth_pos']
    node_type = inputs['node_type'].to(device)
    loss_mask = ~torch.eq(node_type[:, 0], torch.tensor([get_handle_label()], device=device).int())
    if use_fps == False:
        pred_position = 2 * cloth_pos + model.get_output_normalizer().inverse(acceleration) - prev_cloth_pos
    else:
        frame_time = 1/fps
        pred_position = 2 * cloth_pos + (model.get_output_normalizer().inverse(acceleration) * frame_time * frame_time) - prev_cloth_pos
    pred_position = torch.where(loss_mask.reshape((-1,1)), torch.squeeze(pred_position), torch.squeeze(target_cloth_pos))

    return pred_position

def vel2pos(velocity, inputs, model, use_fps, fps):
    cloth_pos = inputs['cloth_pos']
    target_cloth_pos = inputs['target|cloth_pos']
    node_type = inputs['node_type'].to(device)
    loss_mask = ~torch.eq(node_type[:, 0], torch.tensor([get_handle_label()], device=device).int())
    if use_fps == False:
        pred_position = cloth_pos + model.get_output_normalizer().inverse(velocity)
    else:
        frame_time = 1/fps
        pred_position = cloth_pos + (model.get_output_normalizer().inverse(velocity) * frame_time)
    pred_position = torch.where(loss_mask.reshape((-1,1)), torch.squeeze(pred_position), torch.squeeze(target_cloth_pos))

    return pred_position

def get_p_from_epoch(epoch, supervised_epochs, regression_epochs, total_epochs, p_mode, low_p_thresh):
    p = 0.0
    if p_mode == "sup_reg_dec":
        if epoch < supervised_epochs:
            p = 1.0
        elif epoch >= supervised_epochs and epoch < supervised_epochs + regression_epochs:
            p = 0.0
        else:
            a = supervised_epochs + regression_epochs
            b = total_epochs - 1
            p = (epoch - epoch*low_p_thresh + a*low_p_thresh - b)/(a-b)
    elif p_mode == "sup_dec_reg":
        if epoch < supervised_epochs:
            p = 1.0
        elif epoch >= supervised_epochs and epoch < total_epochs - regression_epochs:
            a = supervised_epochs
            b = total_epochs - regression_epochs - 1
            p = (epoch - epoch*low_p_thresh + a*low_p_thresh - b)/(a-b)
        else:
            p = 0.0
        
    return random.random() < p, p

def save_tensor_to_txt(tnsr, filename):
    with open(filename, 'w') as file:
        for row in tnsr:
            file.write('%f %f %f\n' %(row[0].item(), row[1].item(), row[2].item()))
        
def generate_obj_from_npy(ref_obj_path, npy_path, folder_name):
    tar_npy = np.load(npy_path, allow_pickle=True)
    tar_npy = tar_npy.tolist()['pred_pos']
    npy2obj(ref_obj_path, tar_npy, folder_name)

def save_opposite_v(filepath):
    obj_filepath = os.path.join(filepath, "cloth.obj")

    #extract the face information
    face_data = torch.from_numpy(obj_to_face_data(obj_filepath))

    decomposed_face = triangles_to_edges_with_adjf(face_data)
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']

    np.save(os.path.join(filepath, "opposite_v.npy"), opposite_v)

def debug_graph_to_matplotlib(data, node_type_list, labels=None):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    x = data[:,0]
    y = data[:,1]

    color_list = ['navy', 'gray', 'r', 'g','b','black', 'cyan', 'magenta', 'yellow']

    c_list = []
    for node_type in node_type_list:
        c_list.append(color_list[node_type])
        
    plt.scatter(x, y, c=c_list)
    if labels is not None:
        for ind, label in enumerate(labels):
            plt.text(x[ind], y[ind], label)
    
    plt.show()

    print("foo")


def compute_target_acceleration(cloth_pos, prev_cloth_pos, target_cloth_pos):
    cur_position = cloth_pos
    prev_position = prev_cloth_pos
    target_position = target_cloth_pos
    target_acceleration = target_position - 2 * cur_position + prev_position
    
    return target_acceleration

def npy_loader_processor(npy_loader, run_step_config, input_file_path, is_train):

    cloth_mass_npy = np.load(os.path.join(input_file_path, "cloth_m.npy"))

    face = torch.from_numpy(np.load(os.path.join(input_file_path, "face.npy")))
    edge_group = triangles_to_edges_with_adjf(face)
    senders, receivers, opposite_v = edge_group['two_way_connectivity']

    num_frames = npy_loader['cloth_pos'].shape[0]

    npy_loader['face'] = face.repeat(num_frames, 1, 1)
    npy_loader['senders'] = senders.repeat(num_frames, 1)
    npy_loader['receivers'] = receivers.repeat(num_frames, 1)

    train_handle_ind_list = [0,19] ##---IMPORTANT ASSUMPTION----##
    node_type = torch.zeros((npy_loader['cloth_pos'].shape[1],1)).to(dtype=torch.int32)
    for handle_ind in train_handle_ind_list:
        node_type[handle_ind][0] = 1
    npy_loader['node_type'] = node_type.repeat(num_frames,1,1)

    
        
    if (run_step_config['has_sf_ratio'] == True or 
        'sf_ratio' in run_step_config['loss'].keys() or 
        run_step_config['has_sf_ratio_2'] == True or 
        'stretch_e' in run_step_config['loss'].keys()):

        rest_state_obj_filename = os.path.join(input_file_path, "rest.obj")

        train_len_zero_frame = get_len_zero_frame(npy_loader['cloth_pos'].shape[0], 
                    senders, receivers, rest_state_obj_filename, 
                    run_step_config['mesh_pos_mode'])
        npy_loader['len_zero_frame'] = train_len_zero_frame


    if run_step_config['use_fext'] == True or is_train == False:
        npy_loader['node_mass'] = torch.from_numpy(cloth_mass_npy).repeat(num_frames, 1, 1)
        npy_loader['face_area'] = torch.from_numpy(np.load(os.path.join(input_file_path, "cloth_f_area.npy"))).repeat(num_frames, 1, 1)
        #assert the size of node mass and face area
        assert(npy_loader['node_mass'].shape[1] == npy_loader['cloth_pos'].shape[1])
        assert(npy_loader['face_area'].shape[1] == npy_loader['face'].shape[1])
        
    if run_step_config['use_ke'] == True or 'ke' in run_step_config['loss']:
        npy_loader['node_mass'] = torch.from_numpy(cloth_mass_npy).repeat(num_frames, 1, 1)
        #assert the size of node mass and face area
        assert(npy_loader['node_mass'].shape[1] == npy_loader['cloth_pos'].shape[1])

    if run_step_config['is_mesh_space'] == True:
        rest_state_obj_filename = os.path.join(input_file_path, "rest.obj")
        npy_loader['mesh_pos'] = get_mesh_space(rest_state_obj_filename,
                                                npy_loader['cloth_pos'].shape[0],  
                                                run_step_config['mesh_pos_mode'])
                                                
    if (run_step_config['with_theta'] == True or 
        "theta_rest" in run_step_config['loss'] or 
        "bend_e" in run_step_config['loss'] or
        run_step_config['has_bend_e_feature']):

        npy_loader['opposite_v'] = opposite_v.repeat(num_frames, 1, 1)
        
        
    return npy_loader

def get_rel_cloth_pos(cloth_pos, senders, receivers, dim=0):
    relative_cloth_pos = (torch.index_select(input=cloth_pos, dim=dim, index=senders) -
                            torch.index_select(input=cloth_pos, dim=dim, index=receivers))
    return relative_cloth_pos

def get_std_mean_gt(dataset):
    print("foo")

def sigmoid_function(data):
    a = 30
    b = 2000
    c = 0.0075

    exp_torch = torch.exp(-b*(data-c))
    result = (-a/(1+exp_torch)) + a
    return result

def open_mj_data(root):
    gru_path = os.path.join(root, "gru_rough.pt")
    gru_rough_data = torch.load(gru_path).type(torch.float32)

    gnn_path = os.path.join(root, "gnn_full.npy")
    gnn_full_data = torch.from_numpy(np.load(gnn_path)).type(torch.float32)[7:]

    obj_path = os.path.join(root, "rest_state_v2.obj")
    face_obj = torch.from_numpy(obj_to_face_data(obj_path)).type(torch.int32)
    mesh_pos_obj = torch.from_numpy(obj_to_real_pos(obj_path)).type(torch.float32)

    npy_loader = {}
    npy_loader['cloth_pos'] = gru_rough_data
    npy_loader['full_cloth_pos'] = gnn_full_data
    npy_loader['face'] = face_obj.repeat([gru_rough_data.shape[0], 1, 1])
    npy_loader['mesh_pos'] = mesh_pos_obj.repeat([gru_rough_data.shape[0], 1, 1])

    decomposed_face = triangles_to_edges_with_adjf(face_obj)
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']
    ##---get senders and receivers--##
    npy_loader['senders'] = senders.repeat([gru_rough_data.shape[0], 1])
    npy_loader['receivers'] = receivers.repeat([gru_rough_data.shape[0], 1])
    npy_loader['node_type'] = torch.zeros((gru_rough_data.shape[0], gru_rough_data.shape[1], 1))

    # mesh_pos_obj = obj_to_real_pos(obj_path)

    # #get the obj sequence
    # gru_obj_folder_path = os.path.join("debug", "motion_2", "gru_obj_seq")
    # npy2obj(obj_path, gru_data, gru_obj_folder_path)

    # gnn_obj_folder_path = os.path.join("debug", "motion_2", "gnn_obj_seq")
    # npy2obj(obj_path, gnn_data, gnn_obj_folder_path)

    # test_data_path = os.path.join("input", "processed_npy", "dataset_v3_mesh_1024_v16_changed_global", "train.npy")
    # test_data_npy = np.load(test_data_path, allow_pickle=True)
    #print("foo")
    return npy_loader

def compute_kinetic_energy(cur_pos, tar_pos, node_mass, node_mode):
    velocity = tar_pos - cur_pos
    ke = 0.5 * node_mass * velocity * velocity
    ke_norm = torch.norm(ke, dim=1)

    if node_mode == -1:
        ke_mean = torch.mean(ke_norm)
    else:
        ke_mean = ke_norm[node_mode]
    return ke_mean



if __name__=="__main__":
    '''
    filepath = os.path.join("input", "cape_template.obj")
    mesh_space = get_mesh_space(filepath)
    np.save(os.path.join("input","mesh_space.npy"),mesh_space)
    print(mesh_space)
    '''
    '''
    dv = np.load(os.path.join("input","raw_npy", "dv.npy"))
    fext = np.load(os.path.join("input", "basic_cloth", "original", "fext.npy"))
    make_global_feature(os.path.join("input","raw_npy", "gv.npy"))
    '''
    '''
    name = "with_global"
    best_epoch = 146
    folder_name_name = name + "_" + str(best_epoch)
    ref_obj_path = os.path.join("input", "cape_template.obj")
    tar_npy = np.load(os.path.join("output", "basic_cloth", name + ".npy"), allow_pickle=True)
    tar_npy = tar_npy.tolist()['pred_pos']
    folder_name = os.path.join("D:", "meshgraphnets_v2", "obj_results", folder_name_name)
    npy2obj(ref_obj_path, tar_npy, folder_name)
    '''
    
    ###make pkl file
    #base_output_folders = ["long_cloth_motion_1"]
    '''
    base_output_folders = ["long_cloth_motion_1", 
                            "long_cloth_motion_2",
                            "long_cloth_motion_3",
                            "long_cloth_motion_4",
                            "long_cloth_motion_5",
                            "long_cloth_motion_6",
                            "long_cloth_motion_7",]
    '''
    '''
    base_output_folders = ["wide_cloth_motion_1", 
                            "wide_cloth_motion_2",
                            "wide_cloth_motion_3",
                            "wide_cloth_motion_4",
                            "wide_cloth_motion_5",
                            "wide_cloth_motion_6",
                            "wide_cloth_motion_7",]
    '''
    '''
    base_output_folders = ["quarter_cloth_motion_1", 
                            "quarter_cloth_motion_2",
                            "quarter_cloth_motion_3",
                            "quarter_cloth_motion_4",
                            "quarter_cloth_motion_5",
                            "quarter_cloth_motion_6",
                            "quarter_cloth_motion_7",]
    '''
    
    
    # base_output_folders = ["cylinder_cloth_motion_1", "triangle_motion_1", "one_square_motion_1"]
    # handle_inds = [[833, 71, 2356, 1594], [51, 27], [72, 48]]
    # base_output_folders = ["cylinder_cloth_v2_motion_1_3d"]
    # handle_inds = [[119, 6511, 4380, 2250]]

    # base_output_folders = ["mesh_1024_v18", "mesh_1024_v19", "mesh_1024_v20", 
    #                         "mesh_1024_v21", "mesh_1024_v22", "mesh_1024_v23"]
    # handle_inds = [[0, 19], [0, 19], [0, 19],[0, 19],[0, 19],[0, 19]]
    base_output_folders = ["11oz-black-denim", "camel-ponte-roma", "tango-red-jet-set", "white-dots-on-blk",
        "white-swim-solid"]
    handle_inds = [[0,19],[0,19],[0,19],[0,19],[0,19]]
    frame_cut = 178 #178
    last_frame_cut = 0 #59

    frame_steps = 8
    frame_time = 0.016667
    with_forces = False
    with_theta = False

    #output_folder_prefix = "new_topology_"
    output_folder_prefix = "dataset_v3_"
    mesh_pos_mode = "3d" #2d or 3d
    for ind, base_output_folder in enumerate(base_output_folders):
        folder_path = os.path.join("input", "raw_npy", base_output_folder, "npy_dir")

        save_opposite_v(folder_path)
        
        handle_ind = handle_inds[ind]

        train_test_mode = "dataset_v2"

        modes = ["changed_global"]
        #modes = ["changed_global"]
        #modes = ["localized"]
        for mode in modes:
            
            output_folder = output_folder_prefix + base_output_folder + "_" + mode 
            
            full_output_folder = os.path.join("input", "processed_npy", output_folder)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            if mode == "localized":
                make_pkl_file(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time,
                            full_output_folder, with_forces, with_theta,mesh_pos_mode)
            elif mode == "changed_global":
                make_pkl_file_v2(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time,
                            full_output_folder, with_forces, with_theta, mesh_pos_mode)
            
            #make train test npy
            #train_test_ratio = 0.9
            #make_train_test_pkl(full_output_folder, train_test_ratio, full_output_folder, train_test_mode)
            # if base_output_folder == "quarter_cloth_motion_1":
            # if (ind == 0):
            #     #make ground truth obj from the full.npy
            #     full_npy = np.load(os.path.join("input", "processed_npy", output_folder,"full.npy"), allow_pickle=True)
            #     cloth_pos = full_npy.tolist()['cloth_pos']
            #     ref_obj_path = os.path.join("input", "raw_npy", "cylinder_cloth_motion_1", "cloth.obj")
            #     folder_name = os.path.join("D:", "meshgraphnets_v2", "obj_results", "gt_" + output_folder)
            #     npy2obj(ref_obj_path, cloth_pos, folder_name)
    
    '''
    train_npy = np.load(os.path.join("input", "processed_npy", "mesh_1024",
                    "train.npy"), allow_pickle=True)
    test_npy = np.load(os.path.join("input", "processed_npy", "mesh_1024",
                    "test.npy"), allow_pickle=True)
    '''
    '''
    test_npy = np.load(os.path.join("input", "basic_cloth", "original",
                        "train_original.npy"), allow_pickle=True)
    '''
    '''
    sample_v = torch.Tensor([[0.0382, -0.0155, 0.0009],[2,3,4]])
    norm = normalize(sample_v)
    print("foo")   
    '''
    
    # running the dihedral angle computation
    '''
    faces = [[0, 1, 2],[0, 1, 3],[1, 2, 3],[2,3,4]]
    face_data = torch.from_numpy(np.array(faces).astype(np.int32))

    cloth_pos = [[0,0,1],[0,0,0],[1,0,0],[0,1,0],[2,0,1]]
    cloth_pos = torch.from_numpy(np.array(cloth_pos).astype(np.float32))

    decomposed_face = triangles_to_edges_with_adjf(face_data)
    senders, receivers, opposite_v = decomposed_face['two_way_connectivity']

    relative_cloth_pos = (torch.index_select(input=cloth_pos, dim=0, index=senders) -
                              torch.index_select(input=cloth_pos, dim=0, index=receivers))
    
    dihedral_angle = compute_dihedral_angle_fast(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos)
    dihedral_angle_2 = compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos)
    '''
    # import time
    # npy_dic = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta", "train.npy"), allow_pickle=True).tolist()
    
    
    # frame_num = 987
    # faces = npy_dic['face'][frame_num]
    # decomposed_face = triangles_to_edges_with_adjf(faces)
    # senders, receivers, opposite_v = decomposed_face['two_way_connectivity']
    # senders_2 = npy_dic['senders'][frame_num]
    # receivers_2 = npy_dic['receivers'][frame_num]
    # cloth_pos = npy_dic['cloth_pos'][frame_num]
    # relative_cloth_pos = (torch.index_select(input=cloth_pos, dim=0, index=senders) -
    #                           torch.index_select(input=cloth_pos, dim=0, index=receivers))
    # gt_theta = npy_dic['dihedral_angle'][frame_num]

    # senders = senders.to(device)
    # receivers = receivers.to(device)
    # opposite_v = opposite_v.to(device)
    # relative_cloth_pos = relative_cloth_pos.to(device)
    # cloth_pos = cloth_pos.to(device)

    # mode = "unsigned"

    # start_1 = time.time()
    # dihedral_angle = compute_dihedral_angle_fast(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos, mode)
    # end_1 = time.time()
    # dihedral_angle_2 = compute_dihedral_angle(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos)
    # end_2 = time.time()

    # print("fast implementation: ", end_1 - start_1)
    # print("old implementation: ", end_2 - end_1)

    # # filepaths = ["dataset_v3_unrotated_square_motion_1_changed_global", "dataset_v3_unrotated_inv_v2_motion_1_changed_global"]
    # # for filepath in filepaths:
    # #     filepath_full = os.path.join("input", "processed_npy", filepath)
    # #     save_opposite_v(filepath_full)

    # # #read opposite v
    # # opposite_v = np.load(os.path.join(filepath, "opposite_v.npy"))

    # print("foo")

    #debug_dihedral_angle_symmetry()
    #edit_dihedral_angle()
    #debug_dihedral_angle_local_global()
    #debug_check_length()


    #test_torch = torch.tensor([[2,4,6], [8,10,12]])
    #test_torch_2 = torch.tensor([[2,2,2], [2,2,2]])


    #len_frame = get_len_zero_frame(5)
    
    '''
    files = ["wg_t1_best", "wg_sf1_best", "wg_t_ms1_best"]
    for file in files:
        ref_obj_path = os.path.join("input", "raw_npy", "mesh_1024", "cloth.obj")
        tar_npy = np.load(os.path.join("D:\meshgraphnets_v2\output\\results", file + ".npy"), allow_pickle=True)
        tar_npy = tar_npy.tolist()['pred_pos']
        folder_name = os.path.join("D:", "meshgraphnets_v2", "obj_results", file)
        npy2obj(ref_obj_path, tar_npy, folder_name)
    '''
    
    
    
    '''
    #read the input file of the meshgraphnets data
    meshgraphnet_train = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "train.npy"),
                            allow_pickle=True)
    meshgraphnet_test = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "test.npy"),
                            allow_pickle=True)
    my_data = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta", "test.npy"), allow_pickle=True)
    '''
    #visualize the train and test files
    '''
    meshgraphnet_train = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "train.npy"),
                            allow_pickle=True)
    meshgraphnet_test = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "test.npy"),
                            allow_pickle=True)
    
    test_cloth_pos = meshgraphnet_test.tolist()[0]['world_pos']

    cloth_pos = meshgraphnet_test.tolist()[0]['world_pos']
    faces = meshgraphnet_test.tolist()[0]['cells'][0] + 1
    folder_name = os.path.join("input", "processed_npy", "meshgraphnet_data", "debug_test_obj")
    debug_npy2obj(faces, cloth_pos, folder_name)
    '''
    
    #visualize meshgraphnets base
    '''
    #meshgraphnet_train = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "train.npy"),
    #                        allow_pickle=True)
    #faces = meshgraphnet_train.tolist()[0]['cells'][0] + 1
    meshgraphnet_test = np.load(os.path.join("input", "processed_npy", "meshgraphnet_data", "test.npy"),
                            allow_pickle=True)
    faces = meshgraphnet_test.tolist()[0]['cells'][0] + 1
    #files = ["meshgraphnets_base4_epoch_300", "meshgraphnets_base4_epoch_400"]
    files = ["test_epoch_2000"]
    base_folder = "mg_max"
    for file in files:
        tar_npy = np.load(os.path.join("output", "basic_cloth", base_folder, file + ".npy"),
                            allow_pickle=True)
        tar_npy = tar_npy.tolist()['pred_pos']
        folder_name = os.path.join("output", "basic_cloth", base_folder, file)
        debug_npy2obj(faces, tar_npy, folder_name)
    '''
    '''
    run_step_config={}
    run_step_config['input_file_path'] = os.path.join("input", "")
    train_npy_loader = np.load(run_step_config['input_file_path'], allow_pickle=True).tolist()
    if run_step_config['has_sf_ratio'] == True:
        train_len_zero_frame = utils.get_len_zero_frame(train_npy_loader['fext'].shape[0])
        train_npy_loader['len_zero_frame'] = train_len_zero_frame
    if run_step_config['is_meshgraphnet_data'] == True:
        train_npy_loader = train_npy_loader[0]
        train_npy_loader['cloth_pos'] =  train_npy_loader.pop('world_pos')
        train_npy_loader['face'] =  tra
        in_npy_loader.pop('cells')
    '''
    '''
    files = ["full_1_checkpoint"]
    ref_obj_path = os.path.join("input", "raw_npy", "mesh_1024", "cloth.obj")
    base_folders = ["5_with_global_and_theta_Sat-May--7-07-46-21-2022",
                    "6_wg_t_ms_Sun-May--8-12-28-00-2022",
                    "7_wg_sf_Sun-May--8-08-41-43-2022",
                    "8_wg_sf_t_Mon-May--9-02-36-12-2022",
                    "10_wg_sf_ms_Mon-May--9-23-08-25-2022" ]
    for file in files:
            for base_folder in base_folders:
                tar_npy = np.load(os.path.join("output", "basic_cloth", base_folder, file + ".npy"), allow_pickle=True)
                tar_npy = tar_npy.tolist()['pred_pos']
                folder_name = os.path.join("output", "basic_cloth", base_folder, file)
                npy2obj(ref_obj_path, tar_npy, folder_name)
    '''
    '''
    files = ["full_epoch_500", "full_epoch_800"]
    ref_obj_path = os.path.join("input", "raw_npy", "mesh_1024", "cloth.obj")
    base_folders = []
    base_folders.append(os.path.join("base_rerun_v2_Thu-Jun-16-20-23-19-2022", "from_mesh_1024_v22_changed_global"))
    base_folders.append(os.path.join("wgms_18", "from_mesh_1024_v22_localized"))
    #base_folders.append(os.path.join("base_rerun_v2_Thu-Jun-16-20-23-19-2022", "from_mesh_1024_v19_changed_global"))
    #base_folders.append(os.path.join("wgms_18", "from_mesh_1024_v19_localized"))
    
    for file in files:
        for base_folder in base_folders:
            tar_npy = np.load(os.path.join("output", "basic_cloth", base_folder, file + ".npy"), allow_pickle=True)
            tar_npy = tar_npy.tolist()['pred_pos']
            folder_name = os.path.join("output", "basic_cloth", base_folder, file)
            npy2obj(ref_obj_path, tar_npy, folder_name)
    '''


    '''
    data_mesh_1024 = np.load(os.path.join("input", "processed_npy", "mesh_1024_with_theta", "full.npy"),
                            allow_pickle=True)
    handle_v = data_mesh_1024.tolist()['handle_v'][1:-1]
    
    save_tensor_to_txt(handle_v, os.path.join("debug", "handle_v_mesh_1024.txt"))
    '''
    '''
    ###extract_handle_v
    folder_names = ["mesh_1024_v23"]
    for folder_name in folder_names:
        folder_path = os.path.join("input", "raw_npy", folder_name)
        frame_steps = 8
        frame_cut = 120
        last_frame_cut = 0
        handle_ind = [0, 19]
        frame_time = 0.016667
        extract_handle_v(folder_path, frame_steps, frame_cut, last_frame_cut, handle_ind, frame_time)
    print("foo")
    '''
    '''
    foo1 = np.load(os.path.join("input", "processed_npy", "mesh_1024_changed_global",
                    "train.npy"), allow_pickle=True)
    foo2 = np.load(os.path.join("input", "processed_npy", "mesh_1024_corrected_handle_v",
                    "train.npy"), allow_pickle=True)
    '''
    '''
    obj_filename = os.path.join("input", "input_mesh", "debug_dress.obj")
    obj_v = obj_to_real_pos(obj_filename)
    obj_vt = obj_to_mesh_pos(obj_filename)
    '''
    #dress_face = obj_to_face_data(os.path.join("output", "basic_cloth", "mj_gru", "000000.obj"))
    #np.save(os.path.join("output", "basic_cloth", "mj_gru", "dress_face.npy"), dress_face)
    #saved_trajectory = np.load(os.path.join("debug", "trajectory.npy"))
    #open_mj_data()
    print("foo")