'''
    Handles dataset preprocessing, model definition, training process definition and model training
'''


import torch
import debug_transform
from utils import get_handle_label

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        if k != "prev|cloth_pos" and k != 'prev_gt|cloth_pos':
            data_frame[k] = torch.squeeze(v, 0)
        else:
            for ind, prev_cloth_pos in enumerate(data_frame[k]):
                data_frame[k][ind] = torch.squeeze(prev_cloth_pos, 0)
                
    return data_frame


def add_targets(params):
    """Adds target and optionally history fields to dataframe."""
    fields = params['field']
    add_history = params['history']
    velocity_history = params['velocity_history']
    assert(velocity_history >= 1)

    def fn(trajectory, is_gt_accel_hist):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[velocity_history:-1].to(device)
            if is_gt_accel_hist and key == 'cloth_pos':
                out['cloth_pos_gt'] = val[velocity_history:-1].to(device)
            if key in fields:
                if add_history:
                    out['prev|' + key] = []
                    if is_gt_accel_hist:
                        out['prev_gt|' + key] = []
                    for ind in range(velocity_history):
                        out['prev|' + key].append(val[velocity_history - 1 - ind: -2 - ind].to(device))
                        if is_gt_accel_hist:
                            out['prev_gt|' + key].append(val[velocity_history - 1 - ind: -2 - ind].to(device))
                    #out['prev|' + key] = val[:-2].to(device)
                out['target|' + key] = val[velocity_history + 1:].to(device)
                if is_gt_accel_hist:
                    out['target_gt|' + key] = val[velocity_history + 1:].to(device)
        return out
    return fn


def split_and_preprocess(params, model_phase):
    """Splits trajectories into frames, and adds training noise."""
    noise_field = params['field']
    noise_scale = params['noise']
    noise_gamma = params['gamma']
    

    def add_noise(frame):
        zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=noise_scale).to(device)
        other = torch.Tensor([get_handle_label()]).to(device)
        mask = ~torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(steps):
            trajectory_step = {}
            for key, value in trajectory.items():
                if key != "prev|cloth_pos" and key != 'prev_gt|cloth_pos':
                    trajectory_step[key] = value[i]
                else:
                    trajectory_step[key] = []
                    for prev_cloth_pos in value:
                        trajectory_step[key].append(prev_cloth_pos[i])
            if (model_phase == 'train' and params['have_noise'] == True):
                noisy_trajectory_step = add_noise(trajectory_step)
            else:
                noisy_trajectory_step = trajectory_step

            trajectory_steps.append(noisy_trajectory_step)
        return trajectory_steps

    return element_operation


def process_trajectory(trajectory, params, model_phase):

    global steps

    steps = trajectory['cloth_pos'].shape[0] - 2 - params['velocity_history'] + 1
    

    
    bool_flag = ('gt_accel_hist' in params['loss'] or 
        'gt_vel_hist' in params['loss'] or 
        'gt_pos_hist' in params['loss'] or 
        'ke_fixed' in params['loss'] or 
        'theta' in params['loss'])
    trajectory = add_targets(params)(trajectory, bool_flag)
    
    #-------local preprocessing-------
    if params['local_preprocessing'] == True and model_phase == "train":

        
        rot_mat, trans_mat = debug_transform.make_rot_mat_trans_mat(trajectory['cloth_pos'], (0,19))

        trajectory['cloth_pos'] = debug_transform.transform_positions(trajectory['cloth_pos'].transpose(1,2),
                        rot_mat, trans_mat)
        
        for ind in range(params['velocity_history']):
            trajectory['prev|cloth_pos'][ind] = debug_transform.transform_positions(trajectory['prev|cloth_pos'][ind].transpose(1,2),
                            rot_mat, trans_mat)

        trajectory['target|cloth_pos'] = debug_transform.transform_positions(trajectory['target|cloth_pos'].transpose(1,2),
                        rot_mat, trans_mat)

        if bool_flag == True:
            trajectory['cloth_pos_gt'] = trajectory['cloth_pos']
            trajectory['prev_gt|cloth_pos'] = trajectory['prev|cloth_pos']
            trajectory['target_gt|cloth_pos'] = trajectory['target_gt|cloth_pos']
            
            #make an assertion that prev_gt is same with prev
            for ind in range(params['velocity_history']):
                assert(torch.equal(trajectory['prev_gt|cloth_pos'][ind], trajectory['prev|cloth_pos'][ind]))

        #--------------debug: visualize prev_gt---------------------
        # from render_plot import render_single
        # position_list = [trajectory['target|cloth_pos'], trajectory['cloth_pos']]
        # for ind in range(params['velocity_history']):
        #     position_list.append(trajectory['prev|cloth_pos'][ind])
        # face_info = trajectory['face'][0]
        # viewport = (-60,30)
        # fps = 60
        # result_path = os.path.join("debug", "vel_hist_5.mp4")
        # render_single(position_list, face_info, viewport, result_path, fps)
        # print("foo")
        #----------------------------------------------------------------


    if model_phase == "train":
        trajectory = split_and_preprocess(params, model_phase)(trajectory)
    
    return trajectory


