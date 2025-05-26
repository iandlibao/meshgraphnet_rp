import os
import torch
from utils import obj_to_real_pos, obj_to_face_data, triangles_to_edges_with_adjf, obj_to_mesh_pos
import numpy as np



import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

class ClothState:
    
    def __init__(self, initial_pos, rest_pos, face_data, seq_length, handle_ind_list, mass_f_a_path):
        
        self.cloth_pos = initial_pos

        self.mesh_pos = rest_pos
    
        self.num_vertices = self.cloth_pos.shape[0]
        self.face = face_data
        edge_group = triangles_to_edges_with_adjf(self.face)
        self.senders, self.receivers, self.opposite_v = edge_group['two_way_connectivity']
        self.senders = self.senders.to(device)
        self.receivers = self.receivers.to(device)
        self.opposite_v = self.opposite_v.to(device)
        self.set_len_zero_frame()
        self.set_mass_and_face_area(mass_f_a_path)
        self.set_prev_pos(seq_length)
        self.set_node_type(self.cloth_pos.shape[0], handle_ind_list)
        self.set_mask()
        self.set_initial_state_dict()
        self.saved_output = [initial_pos]
        self.handle_ind_list = handle_ind_list

        self.handle_ang_vel = 0.
        # self.handle_rot_axis = torch.tensor([1.,0.,0.]).to(device)
        self.set_radius(handle_ind_list)
        self.handle_transl_mag = 0.
        self.handle_transl_dir = torch.tensor([0.,0.,0.]).to(device)
        self.handle_ang_vel_tensor = torch.tensor([0.,0.,0.]).to(device)

    def set_radius(self, handle_ind_list):
        handle_positions = self.get_position_list(handle_ind_list)
        ref_handle_pos = (handle_positions[0] - handle_positions[1])
        self.radius = torch.norm(ref_handle_pos).item()/2

    def set_len_zero_frame(self):
        rel_cloth_pos = (torch.index_select(self.mesh_pos, 0, self.senders) -
                                torch.index_select(self.mesh_pos, 0, self.receivers))
        self.len_zero_frame = torch.norm(rel_cloth_pos, dim=-1, keepdim=True).to(device)

    def set_mass_and_face_area(self, file_path):
        self.mass = torch.from_numpy(np.load(os.path.join(file_path, "cloth_m.npy"))).to(device)
        self.face_area = torch.from_numpy(np.load(os.path.join(file_path, "cloth_f_area.npy"))).to(device)

    def set_prev_pos(self, seq_length):
        self.prev_pos = []

        for ind in range(seq_length):
            self.prev_pos.append(self.cloth_pos)

    def set_mask(self):
        mask = ~torch.eq(self.node_type[:, 0], torch.tensor([1]).int().to(device))
        self.mask = torch.stack((mask, mask, mask), dim=1).to(device)
    
    def set_node_type(self, num_vertices, handle_ind_list):
        self.node_type = torch.zeros((num_vertices,1)).to(dtype=torch.int32).to(device)
        for handle_ind in handle_ind_list:
            self.node_type[handle_ind][0] = 1

    def set_initial_state_dict(self):
        self.state_dict = {
            "cloth_pos": self.cloth_pos,
            "mesh_pos": self.mesh_pos,
            "face": self.face,
            "senders": self.senders,
            "receivers": self.receivers,
            "opposite_v": self.opposite_v,
            "len_zero_frame": self.len_zero_frame,
            "node_mass": self.mass,
            "face_area": self.face_area,
            "prev|cloth_pos": self.prev_pos,
            "node_type": self.node_type
        }
    
    def get_state_dict(self):
        return self.state_dict

    def set_next_positions(self, next_pos, next_prev_pos):
        self.cloth_pos = next_pos
        self.prev_pos = next_prev_pos
        self.state_dict['cloth_pos'] = self.cloth_pos
        self.state_dict['prev|cloth_pos'] = self.prev_pos

    def save_position(self, pos):
        self.saved_output.append(pos)

    def get_unity_position(self):
        unity_position = self.cloth_pos
        #unity_position[:,0] *= -1
        return unity_position.tolist()

    def get_unity_face(self):
        unity_face = self.face
        unity_face[:, [1,2]] = unity_face[:, [2,1]]
        unity_face = unity_face.flatten()
        return unity_face.tolist()

    def get_position_list(self, vertex_list):
        pos_list = []
        for vertex in vertex_list:
            pos_list.append(self.cloth_pos[vertex])
        
        return pos_list

    def get_vel_list(self, vertex_list):
        vel_list = []
        for vertex in vertex_list:
            vel_list.append(self.cloth_pos[vertex] - self.prev_pos[0][vertex])

        return vel_list