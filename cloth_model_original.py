# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model for FlagSimple."""

from asyncio import run
import torch
from torch import nn as nn
import torch.nn.functional as F
# from torch_cluster import random_walk
import functools

import torch_scatter
import utils
import normalization
import encode_process_decode
from compute_forces import compute_external_force

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, run_step_config):
        super(Model, self).__init__()
        
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        ########TODO: use params to change the size of the node normalizer

        if run_step_config['with_rnn_encoder'] is not None:
            node_features_size = 3 + run_step_config['node_type_length']
        else: 
            node_features_size = (run_step_config['velocity_history'] * 3) + run_step_config['node_type_length']
        if run_step_config['is_force_included'] == True:
            node_features_size = node_features_size + 9
        if run_step_config['use_fext'] == True:
            node_features_size = node_features_size + 3
        if run_step_config['use_ke'] == True:
            if run_step_config['use_ke_version'] == 1:
                node_features_size = node_features_size + 1
            elif run_step_config['use_ke_version'] == 2:
                node_features_size = node_features_size + 2
        
        
        edge_features_size = 4
        if run_step_config['is_mesh_space'] == True:
            if run_step_config['mesh_pos_mode'] == "2d":
                edge_features_size = edge_features_size + 3
            elif run_step_config['mesh_pos_mode'] == "3d":
                if run_step_config['no_3d_rest_vector'] == True:
                    edge_features_size = edge_features_size + 1
                else:
                    edge_features_size = edge_features_size + 4
        if run_step_config['with_theta'] == True:
            edge_features_size = edge_features_size + 1
        if run_step_config['has_sf_ratio'] == True:
            edge_features_size = edge_features_size + 1
        if run_step_config['has_sf_ratio_2'] == True:
            edge_features_size = edge_features_size + 1
        if run_step_config['with_rel_mesh_pos'] == True:
            edge_features_size = edge_features_size + 2
        if run_step_config['use_ke'] == True:
            if run_step_config['use_ke_version'] == 3:
                edge_features_size = edge_features_size + 1
            elif run_step_config['use_ke_version'] == 4:
                edge_features_size = edge_features_size + 2
        if run_step_config['has_stretch_e_feature'] == True:
            edge_features_size = edge_features_size + 1
        if run_step_config['has_bend_e_feature'] == True:
            edge_features_size = edge_features_size + 1
        
        global_features_size = run_step_config['global_features_size']

        self._node_normalizer = normalization.Normalizer(
            size=node_features_size, name='node_normalizer') # [feature dim= 9] # 만약 original space에서 해볼거면 common 을 써야함! subspace는 common_edit
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=edge_features_size, name='mesh_edge_normalizer')  # 3D coord + 1*length = 4 - > [feature dim= 9] + 1 = 10
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer') ## ripple일때만 사용하는거 가틈
        self._global_normalizer = normalization.Normalizer(size=global_features_size, name='global_normalizer')
        
        for loss_key in run_step_config['loss'].keys():
            if run_step_config['loss'][loss_key]['norm'] != None:
                if loss_key == "rel_pos":
                    self._rl_loss_normalizer = normalization.Normalizer(
                        size=3, name='rl_loss_normalizer'
                    )
                elif loss_key == "edge_length":
                    self._el_loss_normalizer = normalization.Normalizer(
                        size=1, name='el_loss_normalizer'
                    )
                elif loss_key == "ke":
                    self._ke_loss_normalizer = normalization.Normalizer(
                        size=3, name='ke_loss_normalizer'
                    )
                elif loss_key == "theta":
                    self._theta_loss_normalizer = normalization.Normalizer(
                        size=1, name='theta_loss_normalizer'
                    )
                elif loss_key == "theta_rest":
                    self._theta_rest_loss_normalizer = normalization.Normalizer(
                            size=1, name='theta_rest_loss_normalizer'
                    )
                elif loss_key == "ke_rest":
                    self._ke_rest_loss_normalizer = normalization.Normalizer(
                        size=3, name='ke_rest_loss_normalizer'
                    )
                elif loss_key == "accel_hist":
                    self._accel_hist_loss_normalizer = normalization.Normalizer(
                        size=3, name='accel_hist_loss_normalizer'
                    )
                elif loss_key == "vel_hist":
                    self._vel_hist_loss_normalizer = normalization.Normalizer(
                        size=3, name='vel_hist_loss_normalizer'
                    )
                elif loss_key == "gt_accel_hist":
                    self._gt_accel_hist_loss_normalizer = normalization.Normalizer(
                        size=3, name='gt_accel_hist_loss_normalizer'
                    )
                elif loss_key == "gt_vel_hist":
                    self._gt_vel_hist_loss_normalizer = normalization.Normalizer(
                        size=3, name='gt_vel_hist_loss_normalizer'
                    )
                elif loss_key == "a_from_pred_v":
                    self._a_from_pred_v_loss_normalizer = normalization.Normalizer(
                        size=3, name='a_from_pred_v_loss_normalizer'
                    )
                elif loss_key == "gt_pos_hist":
                    self._gt_pos_hist_loss_normalizer = normalization.Normalizer(
                        size=3, name='gt_pos_hist_loss_normalizer'
                    )
                elif loss_key == "ke_fixed":
                    self._ke_fixed_loss_normalizer = normalization.Normalizer(
                        size=1, name="ke_fixed_loss_normalizer"
                    )
                elif loss_key == "stretch_e":
                    self._stretch_e_loss_normalizer = normalization.Normalizer(
                        size=1, name="stretch_e_loss_normalizer"
                    )
                elif loss_key == "bend_e":
                    self._bend_e_loss_normalizer = normalization.Normalizer(
                        size=1, name="bend_e_loss_normalizer"
                    )
            

        self.core_model = encode_process_decode
        self.message_passing_steps = run_step_config['message_passing_steps']
        self.message_passing_aggregator = run_step_config['message_passing_aggregator']
        self._attention = run_step_config['attention']

        if run_step_config['with_rnn_encoder'] is not None:
            is_rnn_encoder = True
        else:
            is_rnn_encoder = False

        self.learned_model = self.core_model.EncodeProcessDecode(
            output_size=run_step_config['output_size'],
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator, 
            attention=self._attention, 
            has_global=run_step_config['has_global'],
            global_model_in_processor=run_step_config['global_model_in_processor'],
            is_rnn_encoder=is_rnn_encoder,
            node_features_length=node_features_size,
            edge_features_length=edge_features_size,
            global_latent_size=run_step_config['global_latent_size'])

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long()#.to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])#.to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, params):
        
        graph_list = []
        if params['with_rnn_encoder'] is None:
            velocity_history = params['velocity_history']
            graph_length = 1
            cloth_pos_list = [inputs['cloth_pos']]
            prev_cloth_pos_list = [inputs['prev|cloth_pos']]
        else:
            velocity_history = 1
            graph_length = params['with_rnn_encoder']
            cloth_pos_list = []
            prev_cloth_pos_list = []
            for ind in range(graph_length):
                if ind == graph_length-1:
                    cloth_pos_list.append(inputs['cloth_pos'])
                    prev_cloth_pos_list.append([inputs['prev|cloth_pos'][0]])
                else:
                    cloth_pos_list.append(inputs['prev|cloth_pos'][graph_length-ind-2])
                    prev_cloth_pos_list.append([inputs['prev|cloth_pos'][graph_length-ind-1]])

        
        debug_ind = 0
        for cloth_pos, prev_cloth_pos in zip(cloth_pos_list, prev_cloth_pos_list):
            node_type = inputs['node_type']

            frame_time = 1/params['fps']

            velocity = []
            if params['use_fps'] == False:
                velocity.append(cloth_pos - prev_cloth_pos[0])
                for prev_v_ind in range(velocity_history - 1):
                    velocity.append(prev_cloth_pos[prev_v_ind] - prev_cloth_pos[prev_v_ind + 1])
            else:
                velocity.append((cloth_pos - prev_cloth_pos[0])/frame_time)
                for prev_v_ind in range(velocity_history - 1):
                    velocity.append((prev_cloth_pos[prev_v_ind] - prev_cloth_pos[prev_v_ind + 1])/frame_time)

            one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), params['node_type_length']).to(device)
                
            
            node_features_list = []

            node_features_list.append(torch.cat((velocity), dim=-1))

            if params['is_force_included'] == True:
                fext = inputs['fext']
                sf = inputs['sf']
                bf = inputs['bf']
                node_features_list.append(fext)
                node_features_list.append(sf)
                node_features_list.append(bf)

            #computing external force
            if params['use_fext'] == True:
                if params['use_fps'] == False:
                    computed_fext = compute_external_force(inputs['node_mass'], inputs['face_area'], inputs['face'], velocity[0]/frame_time, cloth_pos)
                else:
                    computed_fext = compute_external_force(inputs['node_mass'], inputs['face_area'], inputs['face'], velocity[0], cloth_pos)
                node_features_list.append(computed_fext)
            
            if params['use_ke'] == True:
                if params['use_ke_version'] == 1:
                    ke_term = 0.5 * inputs['node_mass'] * torch.sum(velocity[0] * velocity[0], dim=1, keepdim=True) 
                    node_features_list.append(ke_term.float())
                if params['use_ke_version'] == 2:
                    ke_term = 0.5 * inputs['node_mass'] * torch.sum(velocity[0] * velocity[0], dim=1, keepdim=True) 
                    node_features_list.append(torch.cat((ke_term.float(), torch.zeros((ke_term.shape)).to(device)), dim=-1))
            
            node_features_list.append(one_hot_node_type)
            node_features = torch.cat(node_features_list, dim=-1)

            senders = inputs['senders']
            receivers = inputs['receivers']
            

            edge_features_list = []
            
            relative_cloth_pos = (torch.index_select(input=cloth_pos, dim=0, index=senders) -
                                torch.index_select(input=cloth_pos, dim=0, index=receivers))
            
            edge_features_list.append(relative_cloth_pos)
            len_real_pos = torch.norm(relative_cloth_pos, dim=-1, keepdim=True)
            edge_features_list.append(len_real_pos)

            if(params['is_mesh_space'] == True):
                mesh_pos = inputs['mesh_pos']
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                    torch.index_select(mesh_pos, 0, receivers))
                if params['no_3d_rest_vector'] == False:
                    edge_features_list.append(relative_mesh_pos)
                edge_features_list.append(torch.norm(relative_mesh_pos, dim=-1, keepdim=True))
            
            if(params['has_sf_ratio'] == True):
                len_zero_frame = inputs['len_zero_frame']
                sf_ratio = (len_real_pos - len_zero_frame)/len_real_pos

                edge_features_list.append(sf_ratio)
            
            if(params['has_sf_ratio_2'] == True):
                len_zero_frame = inputs['len_zero_frame']
                sf_ratio = (len_real_pos - len_zero_frame)

                edge_features_list.append(sf_ratio)
            
            if(params['has_stretch_e_feature'] == True):
                stretch_e = (len_real_pos - inputs['len_zero_frame'])**2

                edge_features_list.append(stretch_e)
                
            if(params['with_theta'] == True or params['has_bend_e_feature']):
                opposite_v = inputs['opposite_v']
                dihedral_angle = utils.compute_dihedral_angle_fast(senders, receivers, opposite_v, relative_cloth_pos, cloth_pos, params['theta_mode'])
                if params['with_theta'] == True:
                    edge_features_list.append(dihedral_angle.reshape((-1,1)))
                if params['has_bend_e_feature'] == True:
                    bend_e = (dihedral_angle.reshape((-1,1)) - inputs['theta_rest'].reshape((-1,1)))**2
                    edge_features_list.append(bend_e)

            
            if(params['with_rel_mesh_pos']== True):
                mesh_pos = inputs['mesh_pos']
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                    torch.index_select(mesh_pos, 0, receivers))
                edge_features_list.append(relative_mesh_pos)

            if params['use_ke'] == True:
                if params['use_ke_version'] == 3:
                    edge_velocity = (torch.index_select(input=velocity[0], dim=0, index=senders) + torch.index_select(input=velocity[0], dim=0, index=receivers))/2
                    edge_mass = (torch.index_select(input=inputs['node_mass'], dim=0, index=senders) + torch.index_select(input=inputs['node_mass'], dim=0, index=receivers))/2
                    ke_term = 0.5 * edge_mass * torch.sum(edge_velocity * edge_velocity, dim=1, keepdim=True) 
                    edge_features_list.append(ke_term.float())
                if params['use_ke_version'] == 4:
                    edge_velocity = (torch.index_select(input=velocity[0], dim=0, index=senders) + torch.index_select(input=velocity[0], dim=0, index=receivers))/2
                    edge_mass = (torch.index_select(input=inputs['node_mass'], dim=0, index=senders) + torch.index_select(input=inputs['node_mass'], dim=0, index=receivers))/2
                    ke_term = 0.5 * edge_mass * torch.sum(edge_velocity * edge_velocity, dim=1, keepdim=True) 
                    edge_features_list.append(torch.cat((ke_term.float(), torch.zeros((edge_velocity.shape)).to(device)), dim=-1))

            edge_features = torch.cat(edge_features_list, dim=-1)

            if(params['has_global'] == True):
                if params['global_version'] == 'handle_v':
                    #global_features = inputs['handle_v'].reshape((-1))
                    global_features = torch.zeros(3).to(device)
                elif params['global_version'] == 'mat_pca':
                    global_features = inputs['mat_pca']
                elif params['global_version'] == 'thickness':
                    global_features = torch.tensor([inputs['thickness']])
                if(params['no_global_normalization'] == False):
                    global_features = self._global_normalizer(global_features)
            else:
                global_features = -1


            mesh_edges = self.core_model.EdgeSet(
                name='mesh_edges',
                features=self._mesh_edge_normalizer(edge_features, params['is_training']),
                receivers=receivers,
                senders=senders)

            
            graph_list.append(self.core_model.MultiGraph(node_features=self._node_normalizer(node_features),
                                                edge_sets=[mesh_edges], global_features=global_features))
            debug_ind = debug_ind + 1
        if params['with_rnn_encoder'] is None:
            #assert(len(graph_list) == 1)
            return graph_list[0]
        else:
            return graph_list
            
    def forward(self, inputs, params):
        graph = self._build_graph(inputs, params)
        #if params['is_training']:
        return self.learned_model(graph,
                        world_edge_normalizer=self._world_edge_normalizer, is_training=params['is_training'])
        # else:
        #     return self._update(inputs, self.learned_model(graph,
        #                                                    world_edge_normalizer=self._world_edge_normalizer,
        #                                                    is_training=params['is_training']))

    def _update(self, inputs, per_node_network_output, use_fps, fps):
        """Integrate model outputs."""

        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['cloth_pos']
        prev_position = inputs['prev|cloth_pos'][0].to(device)

        if use_fps == False:
            position = 2 * cur_position + acceleration - prev_position
        else:
            frame_time = 1/fps
            position = 2 * cur_position + acceleration * frame_time * frame_time - prev_position

        return position, acceleration
    
    def _update_vel(self, inputs, per_node_network_output, use_fps, fps):
        """Integrate model outputs."""

        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['cloth_pos']

        if use_fps == False:
            position = cur_position + velocity 
        else:
            frame_time = 1/fps
            position = cur_position + velocity * frame_time
        
        return position, velocity

    def get_output_normalizer(self):
        return self._output_normalizer

    def get_loss_normalizer(self, norm_mode):
        if norm_mode == "ke":
            return self._ke_loss_normalizer
        elif norm_mode == "edge_length":
            return self._el_loss_normalizer
        elif norm_mode == "rel_pos":
            return self._rl_loss_normalizer
        elif norm_mode == "theta":
            return self._theta_loss_normalizer
        elif norm_mode == "theta_rest":
            return self._theta_rest_loss_normalizer
        elif norm_mode == "ke_rest":
            return self._ke_rest_loss_normalizer
        elif norm_mode == "accel_hist":
            return self._accel_hist_loss_normalizer
        elif norm_mode == "vel_hist":
            return self._vel_hist_loss_normalizer
        elif norm_mode == "gt_accel_hist":
            return self._gt_accel_hist_loss_normalizer
        elif norm_mode == "gt_vel_hist":
            return self._gt_vel_hist_loss_normalizer
        elif norm_mode == "a_from_pred_v":
            return self._a_from_pred_v_loss_normalizer
        elif norm_mode == "gt_pos_hist":
            return self._gt_pos_hist_loss_normalizer
        elif norm_mode == "ke_fixed":
            return self._ke_fixed_loss_normalizer
        elif norm_mode == "stretch_e":
            return self._stretch_e_loss_normalizer
        elif norm_mode == "bend_e":
            return self._bend_e_loss_normalizer

    def save_model(self, path, loss_dict):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        torch.save(self._global_normalizer, path + "_global_normalizer.pth")

        for loss_key in loss_dict.keys():
            if loss_dict[loss_key]['norm'] != None:
                torch.save(self.get_loss_normalizer(loss_key), path + "_" + loss_key + "_loss_normalizer.pth")
        

    def load_model(self, path, has_global, loss_dict, new_json=True):
        self.learned_model = torch.load(path + "_learned_model.pth", map_location=device)
        self._output_normalizer = torch.load(path + "_output_normalizer.pth", map_location=device)
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth", map_location=device)
        self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth", map_location=device)
        self._node_normalizer = torch.load(path + "_node_normalizer.pth", map_location=device)
        if has_global:
            self._global_normalizer = torch.load(path + "_global_normalizer.pth", map_location=device)
        
        for loss_key in loss_dict.keys():
            if loss_dict[loss_key]['norm'] != None:
                if new_json:
                    if loss_key == "ke":
                        self._ke_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "rel_pos":
                        self._rl_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "edge_length":
                        self._el_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "theta":
                        self._theta_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "theta_rest":
                        self._theta_rest_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "ke_rest":
                        self._ke_rest_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "accel_hist":
                        self._accel_hist_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "vel_hist":
                        self._vel_hist_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "gt_accel_hist":
                        self._gt_accel_hist_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    # elif loss_key == "gt_vel_hist":
                    #     self._gt_vel_hist_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                    #                 map_location=device)
                    elif loss_key == "a_from_pred_v":
                        self._a_from_pred_v_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "gt_pos_hist":
                        self._gt_pos_hist_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "ke_fixed":
                        self._ke_fixed_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "stretch_e":
                        self._stretch_e_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                                    map_location=device)
                    # elif loss_key == "bend_e":
                    #     self._bend_e_loss_normalizer = torch.load(path + "_" + loss_key + "_loss_normalizer.pth",
                    #                 map_location=device)
                else:
                    if loss_key == "ke":
                        self._ke_loss_normalizer = torch.load(path + "_proper_ke_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "rel_pos":
                        self._rl_loss_normalizer = torch.load(path + "_rl_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "edge_length":
                        self._el_loss_normalizer = torch.load(path + "_el_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "theta":
                        self._theta_loss_normalizer = torch.load(path + "_theta_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "theta_rest":
                        self._theta_rest_loss_normalizer = torch.load(path + "_theta_rest_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "ke_rest":
                        self._ke_rest_loss_normalizer = torch.load(path + "_ke_rest_loss_normalizer.pth",
                                    map_location=device)
                    elif loss_key == "accel_hist":
                        self._accel_hist_loss_normalizer = torch.load(path + "_accel_hist_loss_normalizer.pth",
                                    map_location=device)
    

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
