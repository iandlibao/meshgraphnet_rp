import os
from utils import read_json_file, obj_to_real_pos, obj_to_face_data, triangles_to_edges_with_adjf, obj_to_mesh_pos
import torch
import numpy as np
from generate_json_conf import generate_from_code_for_gen_v3
import cloth_model_original
from debug_transform import make_rot_mat_trans_mat, transform_positions
from cloth_state import ClothState
from tqdm import tqdm
import json

model_dir_root = os.path.join("output", "basic_cloth")

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')



motion_traj_to_mcode = {
    "forwardBack": "fwd"
}


obj_name_to_handle_ind_list_dict = {
    "square_1024": [0, 19],
    "longObj": [31,39],
    "big": [34, 44],
    "quarter": [13, 17],
    "wide": [21, 29],
    "triangle": [1, 2],
    "diamond": [1, 3, 2],
    "wide3Handles": [21,29,25],
    "triangleManyHandles": [1, 2, 53, 21, 54, 8, 60, 20, 52, 4, 59, 18, 58, 7, 56, 19, 55]
}

def setup_handle_traj(initial_mesh, motion_traj, handle_ind_list, fps):


    motion_code = get_motion_code(motion_traj)

    handle_traj = generate_from_code_for_gen_v3(motion_code, handle_ind_list, initial_mesh, fps)

    return handle_traj

def get_path_from_gt_input(gt_input):
    root_path = os.path.join("input", "gt_data")
    gt_folders = [folder for folder in os.listdir(root_path) if \
        os.path.isdir(os.path.join(root_path, folder))]
    
    gt_path = None
    for gt_folder in gt_folders:
        json_path = os.path.join(root_path, gt_folder, "meta.json")

        assert(os.path.exists(json_path))
        with open(json_path, 'r') as json_file:
            json_dict = json.load(json_file)

        equal_flag = True
        for key, val in json_dict.items():
            if gt_input[key] != val:
                equal_flag = False
                break

        if equal_flag:
            gt_path = os.path.join(root_path, gt_folder)
            break
    
    assert(gt_path is not None)
    return gt_path


def get_motion_code(motion_traj):
    if motion_traj in motion_traj_to_mcode.keys():
        return motion_traj_to_mcode[motion_traj]
    else:
        return motion_traj

def setup_model(run_step_config):
    run_epoch_number = run_step_config['epoch_number']
    output_folder = run_step_config['output_folder']
    output_number = run_step_config['output_number']
    last_run_step_dir = os.path.join(model_dir_root, output_folder, output_number)


    model = cloth_model_original.Model(run_step_config)

    if run_epoch_number == -1:
        model_mode = "/epoch_model_checkpoint"
    else:
        model_mode = "/epoch_model_checkpoint_" + str(run_epoch_number)
    model.load_model(os.path.join(last_run_step_dir, 'checkpoint')+ model_mode, run_step_config['has_global'], 
                        run_step_config['loss'], run_step_config['new_json'])
    model.to(device)

    #if(run_step_config['has_global'] == False):
    model.learned_model.encoder.has_global = run_step_config['has_global']
    model.learned_model.processor.has_global = run_step_config['has_global']
    for block in model.learned_model.processor.graphnet_blocks:
        block.has_global = run_step_config['has_global']
        block.global_model_in_processor = run_step_config['global_model_in_processor']

    return model


    
def step_fn(model, state, run_step_config, handle_traj_pos):
    prev_pos = state.prev_pos
    cur_pos = state.cloth_pos
    
    #---localize-------
    global_cur_pos = cur_pos.reshape(1, cur_pos.shape[0], -1)
    global_prev_pos_list = []
    for one_prev_pos in prev_pos:
        global_prev_pos = one_prev_pos.reshape(1, one_prev_pos.shape[0], -1)
        global_prev_pos_list.append(global_prev_pos)

    handle_ind = run_step_config['handle_ind']
    
    
    rot_mat, trans_mat, g2l_rot_mat, g2l_trans_mat = make_rot_mat_trans_mat(global_cur_pos, handle_ind, True)
    cur_pos = transform_positions(global_cur_pos.transpose(1,2),
            rot_mat, trans_mat).squeeze()
    
    for ind, global_prev_pos in enumerate(global_prev_pos_list):
        prev_pos[ind] = transform_positions(global_prev_pos.transpose(1,2),
                rot_mat, trans_mat).squeeze()
    #----------------

    #set the local positions of the cur_pos and prev_pos
    state.set_next_positions(cur_pos, prev_pos)


    with torch.no_grad():
        network_output = model(state.get_state_dict(), run_step_config)


    if run_step_config['loss_params']['pred_vel'] == False:
        prediction, accel = model._update(state.get_state_dict(), network_output, 
                                run_step_config['use_fps'], run_step_config['fps'])
    else:
        prediction, vel = model._update_vel(state.get_state_dict(), network_output,
                                run_step_config['use_fps'], run_step_config['fps'])
        accel = vel

    #---------globalize-------------
    prediction = transform_positions(prediction.reshape(1, prediction.shape[0], -1).transpose(1,2),
                g2l_rot_mat, g2l_trans_mat).squeeze()
    #----------------------------------

    
    next_pos = torch.where(state.mask, torch.squeeze(prediction), torch.squeeze(handle_traj_pos))
    

    next_prev_pos = []
    for ind in range(len(global_prev_pos_list)):
        if ind == 0:
            global_prev_pos = global_cur_pos
        else:
            global_prev_pos = global_prev_pos_list[ind - 1] 
        next_prev_pos.append(global_prev_pos.squeeze())
    

    state.set_next_positions(next_pos, next_prev_pos)
    state.save_position(next_pos)

def get_output_directory(file_name, folder_name, json_params):
    output_file_suff=""
    if json_params['starting_frame'] != 0:
        output_file_suff += output_file_suff + '_start_' + str(json_params['starting_frame'])
    if json_params['rollout_length'] != -1:
        output_file_suff += '_rollout_' + str(json_params['rollout_length'])
    if json_params['epoch_number'] != -1:
        output_file_suff += '_e_' + str(json_params['epoch_number'])
    else:
        output_file_suff += '_e_' + str(json_params['epoch_last_checkpoint'])
    output_file = file_name + output_file_suff

    output_dir_root = os.path.join("output", "npy_results")
    model_npy_output_folder = os.path.join(output_dir_root, json_params['name_code'])
    if not os.path.exists(model_npy_output_folder):
        os.makedirs(model_npy_output_folder)
    npy_output_folder = os.path.join(model_npy_output_folder, folder_name)
    if not os.path.exists(npy_output_folder):
        os.makedirs(npy_output_folder)

    output_dir = os.path.join(npy_output_folder, output_file)

    return output_dir

def get_render_directory(file_name, folder_name, json_name, viewport):
    result_root = os.path.join("output", "renders")
    result_model_folder_root = os.path.join(result_root, json_name)
    if not os.path.exists(result_model_folder_root):
        os.makedirs(result_model_folder_root)
    result_path = os.path.join(result_model_folder_root, folder_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    suf_name = '_' + viewport
    mp4_target_filename = file_name + suf_name + '.mp4'
    final_result_path = os.path.join(result_path, mp4_target_filename)
    
    return final_result_path

def set_model_cloth_state_and_params(json_name, initial_pos, rest_obj_file_path, handle_ind_list, mass_f_a_path):
    json_params = read_json_file(json_name, 'test')
    json_params['handle_ind'] = handle_ind_list
    
    model = setup_model(json_params)

    mesh_pos_mode = json_params['mesh_pos_mode']
    if mesh_pos_mode == '3d':
        rest_pos = torch.from_numpy(obj_to_real_pos(rest_obj_file_path)).to(device)
    elif mesh_pos_mode == '2d':
        rest_pos = torch.from_numpy(obj_to_mesh_pos(rest_obj_file_path)).to(device)

    if json_params['with_rnn_encoder'] is not None:
        seq_length = json_params['with_rnn_encoder']
    else:
        seq_length = 1

    face_data = torch.from_numpy(obj_to_face_data(rest_obj_file_path)).to(device)

    cloth_state = ClothState(initial_pos, rest_pos, face_data, seq_length, handle_ind_list, 
        mass_f_a_path)


    return model, cloth_state, json_params


def setup_handle_traj_gt(gt_path, handle_ind_list):
    
    cloth_pos = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy")))

    handle_traj = torch.zeros(cloth_pos.shape).to(device)
    for handle_ind in handle_ind_list:
        handle_traj[:,handle_ind,:] = cloth_pos[:,handle_ind,:]
    
    return handle_traj[1:]

def setup_arbitrary_motion(json_name, input_data):
    obj_name = input_data['obj_code']
    motion_traj = input_data['motion_code']

    obj_root_dir = os.path.join("input", "unity_demo")
    

    rest_obj_file_path = os.path.join(obj_root_dir, obj_name + ".obj")
    initial_pos = torch.from_numpy(obj_to_real_pos(rest_obj_file_path)).to(device)

    obj_name_to_mass_f_area_dict = {
        "wide3Handles": "wide",
        "triangleManyHandles": "triangle"
    }
    if obj_name in obj_name_to_mass_f_area_dict.keys():
        mass_f_a_path_fname = obj_name_to_mass_f_area_dict[obj_name]
    else:
        mass_f_a_path_fname = obj_name
    mass_f_a_path = os.path.join(obj_root_dir, mass_f_a_path_fname)

    handle_ind_list = obj_name_to_handle_ind_list_dict[obj_name]

    model, cloth_state, json_params = set_model_cloth_state_and_params(json_name, initial_pos, rest_obj_file_path,
        handle_ind_list, mass_f_a_path)

    if motion_traj == 'userFeed':
        handle_traj = None
    else:
        handle_traj = setup_handle_traj(cloth_state.cloth_pos, motion_traj, handle_ind_list,
            json_params['fps'])

    return model, cloth_state, handle_traj, json_params

def setup_motion_with_gt(json_name, gt_input):
    gt_path = get_path_from_gt_input(gt_input)
    rest_obj_file_path = os.path.join(gt_path, "rest.obj")
    full_cloth_pos = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
    initial_pos = full_cloth_pos[0]
    mass_f_a_path = gt_path

    handle_ind_list = obj_name_to_handle_ind_list_dict[gt_input['obj_code']]
    

    model, cloth_state, json_params = set_model_cloth_state_and_params(json_name, initial_pos, rest_obj_file_path,
        handle_ind_list, mass_f_a_path)


    handle_traj = setup_handle_traj_gt(gt_path, handle_ind_list)

    return model, cloth_state, handle_traj, json_params

def initial_setup(json_name, input_data):
    if input_data["mode"] == 'arbitrary':
        return setup_arbitrary_motion(json_name, input_data)
    elif input_data["mode"] == 'with_gt':
        return setup_motion_with_gt(json_name, input_data)


def save_output(input_data, output_pos, face_data, json_params):
    folder_name = input_data["mode"]
    
    file_name = input_data['obj_code'] + '_' + input_data['motion_code']

    output_dir = get_output_directory(file_name, folder_name, json_params)

    npy_output = {
        "position": output_pos,
        "face": face_data,
        "input_data": input_data,
        "model_name": json_params['name_code']
    }
    np.save(output_dir, npy_output)


if __name__ == "__main__":
    
    json_name_list = [
        "mesh_graph_nets", 
        "mlp_encoder",
        "final_model"    
    ]
    input_data_list = [
        {
            "mode": "arbitrary",
            "obj_code": "square_1024",
            "motion_code": "fwd"
        },
        {
            "mode": "with_gt",
            "obj_code": "square_1024",
            "motion_code": "fwd"
        }
    ]

    for json_name in json_name_list:

        for input_data in input_data_list:

            model, cloth_state, handle_traj, json_params = initial_setup(json_name, input_data)

            for frame_handle_traj in tqdm(handle_traj):
                step_fn(model, cloth_state, json_params, frame_handle_traj)

            output_pos = torch.stack(cloth_state.saved_output).cpu().numpy()

            save_output(input_data, output_pos, cloth_state.face.cpu().numpy(), json_params)
