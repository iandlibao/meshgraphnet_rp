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
"""Runs the learner/evaluator."""

import wandb
import os
from pathlib import Path
from absl import app
from absl import flags
import torch
import gnn_preprocess as preprocess
import gnn_summary as summary
import cloth_model_original
import utils
import logging
import time
import datetime
import debug_transform
import random
from preprocess_full_dataset import get_data

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')
    

#################################################################################################################################

steps = None




FLAGS = flags.FLAGS
flags.DEFINE_string('params', 'final_model', 'the json file')
#################################################################################################################################


def learner(model, run_step_config):

    if run_step_config['is_save_output_on'] == True:
        #--------change your wandb credentials here --------------#
        wandb.login(key='put_key')
        wandb.init(project='put_project_name', entity="put_username", name=run_step_config['name_code'], config=run_step_config)
    
    #set up the node_type_new once
    utils.set_handle_label(is_new_labels=run_step_config['is_new_labels'])

    
    root_logger = logging.getLogger()


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
    
    trained_epoch = 0

    if run_step_config['last_run_dir'] is not None:
        if run_step_config['epoch_checkpoint'] == -1:
            if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
                optimizer.load_state_dict(
                    torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_optimizer_checkpoint.pth")
                        #,map_location=device)
                        )
                )
                scheduler.load_state_dict(
                    torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_scheduler_checkpoint.pth")
                        #,map_location=device)
                        )
                )
            elif platform.machine() == 'arm64':
                optimizer.load_state_dict(
                    torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_optimizer_checkpoint.pth")
                        ,map_location=device)
                )
                scheduler.load_state_dict(
                    torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_scheduler_checkpoint.pth")
                        ,map_location=device)
                )
            
            epoch_checkpoint = torch.load(
                os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_checkpoint.pth"))
            trained_epoch = epoch_checkpoint['epoch'] + 1
            root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")
        else:
            optimizer.load_state_dict(
                torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_optimizer_checkpoint" + str(run_step_config['epoch_checkpoint']) + ".pth")))
            scheduler.load_state_dict(
                torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_scheduler_checkpoint" + str(run_step_config['epoch_checkpoint']) + ".pth")))
            trained_epoch = run_step_config['epoch_checkpoint']
            root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")

    

    count = 0

    epoch_training_losses = []
    all_trajectory_train_losses = []
    epoch_run_times = []

    wandb_step = 0
    step_offset = run_step_config['sample_start_point']

    train_npy_loader = []
    test_npy_loader = []

    assert run_step_config['full_data_version'] is not None
        
    full_input_file_path = os.path.join("input", "gt_data", "square_1024_fwd")
        

    train_dataset = get_data(run_step_config['full_data_version'], "train")
    for key, val in train_dataset.items():
        npy_loader = utils.npy_loader_processor(val, run_step_config, full_input_file_path, True)
        train_npy_loader.append(npy_loader)
    
    test_dataset = get_data(run_step_config['eval_data_ver'], "train")
    for key, val in test_dataset.items():
        npy_loader = utils.npy_loader_processor(val, run_step_config, full_input_file_path, True)
        test_npy_loader.append(npy_loader)


    train_trajectory_length = len(train_npy_loader)
    eval_trajectory_length = len(test_npy_loader)

    
    gt_mean_std = None

    if "accel_hist" not in run_step_config['loss'].keys():
        prev_pred_accel = None
        prev_tar_accel = None
        prev_pred_accel_eval = None
        prev_tar_accel_eval = None
    if "vel_hist" not in run_step_config['loss'].keys():
        prev_pred_vel = None
        prev_tar_vel = None
        prev_pred_vel_eval = None
        prev_tar_vel_eval = None
    if "gt_accel_hist" not in run_step_config['loss'].keys():
        prev_pred_accel = None
        prev_gt_tar_accel = None
        prev_pred_accel_eval = None
        prev_gt_tar_accel_eval = None
    if "gt_vel_hist" not in run_step_config['loss'].keys():
        prev_pred_vel = None
        prev_gt_tar_vel = None
        prev_pred_vel_eval = None
        prev_gt_tar_vel_eval = None
    if "a_from_pred_v" not in run_step_config['loss'].keys():
        prev_pred_vel = None
        prev_pred_vel_eval = None


    max_regress_count = run_step_config['max_regress_count']
    
    for epoch in range(run_step_config['epochs'])[trained_epoch:]:
        # check whether the rest time is sufficient for running a whole epoch; stop running if not
        hpc_current_time = time.time()
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
        epoch_training_loss = 0.0
        epoch_eval_loss = 0.0

        epoch_training_loss_dict = {}
        epoch_eval_loss_dict = {}
        for key in run_step_config['loss']:
            epoch_training_loss_dict[key] = 0.0
            epoch_eval_loss_dict[key] = 0.0



        if run_step_config['is_save_output_on'] == True:
            wandb.log({"epoch": epoch}, step=wandb_step + step_offset)
        for model_phase in ['train', 'val']:
            with torch.set_grad_enabled(model_phase == 'train'):
            
                if model_phase =='train':
                    

                    model.train()
                    ds_iterator = iter(train_npy_loader)
                    for trajectory_index in range(train_trajectory_length):
                        root_logger.info(
                            "    trajectory index " + str(trajectory_index + 1) + "/" + str(train_trajectory_length))
                        trajectory = next(ds_iterator)
                        trajectory = preprocess.process_trajectory(trajectory, run_step_config, model_phase)
                        trajectory_loss = 0.0


                        traj_training_loss = {}
                        for key in run_step_config['loss']:
                            traj_training_loss[key] = 0.0

                        use_output_as_input = False
                        regress_count = 0

                        if("accel_hist" in run_step_config['loss'].keys()):
                            prev_pred_accel = None
                            prev_tar_accel = None

                        if "vel_hist" in run_step_config['loss'].keys():
                            prev_pred_vel = None
                            prev_tar_vel = None

                        if "gt_accel_hist" in run_step_config['loss'].keys():
                            prev_pred_accel = None
                            prev_gt_tar_accel = None
                        
                        if "gt_vel_hist" in run_step_config['loss'].keys():
                            prev_pred_vel = None
                            prev_gt_tar_vel = None

                        if "a_from_pred_v" in run_step_config['loss'].keys():
                            prev_pred_vel = None

                        for idx, data_frame in enumerate(trajectory): # frame by frame learning!
                            count += 1
                            data_frame = preprocess.squeeze_data_frame(data_frame)

                            #overtake the cur and prev
                            if use_output_as_input == True: #regression
                                for ind in range(run_step_config['velocity_history']):
                                    if ind == 0:
                                        data_frame['prev|cloth_pos'][ind] = last_frame_cur_pos
                                    else:
                                        data_frame['prev|cloth_pos'][ind] = last_frame_prev_pos[ind - 1]
                                data_frame['cloth_pos'] = last_frame_pred
                                use_output_as_input = False

                                if run_step_config['local_preprocessing'] == True:
                                    cur_pos = data_frame['cloth_pos'].reshape(1,-1,3)
                                    prev_pos_list = []
                                    for ind in range(run_step_config['velocity_history']):
                                        prev_pos_list.append(data_frame['prev|cloth_pos'][ind].reshape(1,-1,3))
                                    

                                    rot_mat, trans_mat = debug_transform.make_rot_mat_trans_mat(cur_pos, (0,19))

                                    data_frame['cloth_pos'] = debug_transform.transform_positions(cur_pos.transpose(1,2),
                                            rot_mat, trans_mat).squeeze()
                                    
                                    for ind in range(run_step_config['velocity_history']):
                                        data_frame['prev|cloth_pos'][ind] = debug_transform.transform_positions(prev_pos_list[ind].transpose(1,2),
                                                rot_mat, trans_mat).squeeze()
                            torch.autograd.set_detect_anomaly(True)
                            network_output = model(data_frame, run_step_config)

                            if run_step_config['p_mode'] != "dec_loss":
                                p_bool, p_float = utils.get_p_from_epoch(epoch, run_step_config['supervised_epochs'],
                                                            run_step_config['regression_epochs'],
                                                            run_step_config['epochs'],
                                                            run_step_config['p_mode'],
                                                            run_step_config['low_p_thresh'])
                            else:
                                if epoch < run_step_config['supervised_epochs']:
                                    p_float = 1.0
                                    p_bool = True
                                else:
                                    p_bool = random.random() < p_float

                            if  p_bool == False and regress_count < max_regress_count: #if use regression
                                if run_step_config['loss_params']['pred_vel'] == False:
                                    last_frame_pred = utils.accel2pos(network_output.detach(), data_frame, model, 
                                            run_step_config['use_fps'], run_step_config['fps'])
                                else:
                                    last_frame_pred = utils.vel2pos(network_output.detach(), data_frame, model,
                                            run_step_config['use_fps'], run_step_config['fps'])
                                last_frame_cur_pos = data_frame['cloth_pos']
                                last_frame_prev_pos = data_frame['prev|cloth_pos']
                                use_output_as_input = True
                                regress_count += 1
                            else:
                                regress_count = 0

                            
                            loss, loss_dict = loss_fn(data_frame, network_output, model, 
                                                    run_step_config['loss'], 
                                                    run_step_config['loss_params'],
                                                    idx,
                                                    gt_mean_std, 
                                                    prev_tar_accel, 
                                                    prev_pred_accel,
                                                    prev_tar_vel,
                                                    prev_pred_vel,
                                                    prev_gt_tar_accel,
                                                    prev_gt_tar_vel,
                                                    use_output_as_input)


                        
                            
                            if("accel_hist" in run_step_config['loss'].keys()):
                                prev_pred_accel = network_output.detach()
                                cur_tar_accel = data_frame['target|cloth_pos'] - 2 * data_frame['cloth_pos'] + data_frame['prev|cloth_pos'][0]
                                prev_tar_accel = model.get_output_normalizer()(cur_tar_accel).to(device)

                            if "vel_hist" in run_step_config['loss'].keys():
                                prev_pred_vel = network_output.detach()
                                cur_tar_vel = data_frame['target|cloth_pos'] - data_frame['cloth_pos']
                                prev_tar_vel = model.get_output_normalizer()(cur_tar_vel).to(device)

                            if "gt_accel_hist" in run_step_config['loss'].keys():
                                prev_pred_accel = network_output.detach()
                                prev_gt_tar_accel = data_frame['target_gt|cloth_pos'] - 2 * data_frame['cloth_pos_gt'] + data_frame['prev_gt|cloth_pos'][0]

                            if "gt_vel_hist" in run_step_config['loss'].keys():
                                if run_step_config['loss_params']['pred_vel'] == True:
                                    prev_pred_vel = network_output.detach()
                                else:
                                    prev_pred_vel = model.get_output_normalizer().inverse(network_output.detach()) + data_frame['cloth_pos'] - data_frame['prev|cloth_pos'][0]
                                prev_gt_tar_vel = data_frame['target_gt|cloth_pos'] - data_frame['cloth_pos_gt']
                            
                            if "a_from_pred_v" in run_step_config['loss'].keys():
                                #asserted that pred_vel is true
                                prev_pred_vel = network_output.detach()


                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            trajectory_loss += loss.detach().cpu()

                            for key, value in loss_dict.items():
                                traj_training_loss[key] += value
                            
                            if run_step_config['is_save_output_on'] == True:
                                wandb.log({"train loss": loss}, step=wandb_step + step_offset)
                                wandb.log({"p float": p_float}, step=wandb_step + step_offset)
                                wandb.log({"regress count": regress_count}, step=wandb_step + step_offset)
                                if run_step_config["name_code"] == "min_model_2_v86_debug_3":
                                    for key, val in loss_dict.items():
                                        wandb_title = "frame " + key + " loss"
                                        wandb.log({wandb_title: val}, step=wandb_step + step_offset)
                                wandb_step = wandb_step + 1
                             
                        # one trajectory finished!
                        all_trajectory_train_losses.append(trajectory_loss)

                        epoch_training_loss += trajectory_loss

                        for key, value in traj_training_loss.items():
                            epoch_training_loss_dict[key] += value
                        
                        model.save_model(os.path.join(run_step_config['checkpoint_dir'],
                                        "trajectory_model_checkpoint"), run_step_config['loss'])

                        torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'],
                                                "trajectory_optimizer_checkpoint" + ".pth"))
                        torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'],
                                                "trajectory_scheduler_checkpoint" + ".pth"))
                    # one epoch finished!
                    epoch_training_losses.append(epoch_training_loss)
                    root_logger.info("        epoch_training_loss")
                    root_logger.info("        " + str(epoch_training_loss))
                    if run_step_config['is_save_output_on'] == True:
                        wandb.log({"epoch train loss": epoch_training_loss}, step=wandb_step + step_offset)
                    
                    if epoch >= run_step_config['supervised_epochs'] and run_step_config['p_mode'] == "dec_loss":
                        if epoch == run_step_config['supervised_epochs']:
                            prev_loss = epoch_training_loss
                            p_float=1.0
                        else:
                            cur_loss = epoch_training_loss
                            if cur_loss < prev_loss:
                                p_float = p_float - 0.01
                            assert(p_float > 0.0)
                            #prev_loss = cur_loss


                    for key, value in epoch_training_loss_dict.items():
                        root_logger.info("        epoch_training_" + key + "_loss")
                        root_logger.info("        " + str(value))
                        if run_step_config['is_save_output_on'] == True:
                            wandb_loss_title = "epoch train " + key + " loss"
                            if key == 'rel_pos':
                                wandb_loss_title = "epoch train rc loss"
                            elif key == 'edge_length':
                                wandb_loss_title = "epoch train el loss"
                            wandb.log({wandb_loss_title: value}, step=wandb_step + step_offset)

                   
                    model.save_model(os.path.join(run_step_config['checkpoint_dir'],
                            "epoch_model_checkpoint"), run_step_config['loss'])
                    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'],
                                    "epoch_optimizer_checkpoint" + ".pth"))
                    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'],
                                    "epoch_scheduler_checkpoint" + ".pth"))
                    torch.save({'epoch': epoch}, os.path.join(run_step_config['checkpoint_dir'], "epoch_checkpoint.pth"))
                
        if (epoch%run_step_config['save_epoch']==0):
            print('run_save_epoch')
            model.save_model(os.path.join(run_step_config['checkpoint_dir'],
                            "epoch_model_checkpoint_" + str(epoch)), run_step_config['loss'])
            torch.save(optimizer.state_dict(),
                    os.path.join(run_step_config['checkpoint_dir'],
                                    "epoch_optimizer_checkpoint" + str(epoch) + ".pth"))
            torch.save(scheduler.state_dict(),
                    os.path.join(run_step_config['checkpoint_dir'],
                                    "epoch_scheduler_checkpoint" + str(epoch) + ".pth"))

        epoch_run_times.append(time.time() - hpc_current_time)
        if epoch == 13:
            scheduler.step()
            root_logger.info("Call scheduler in epoch " + str(epoch))

    
    model.save_model(os.path.join(run_step_config['checkpoint_dir'], "model_checkpoint"), run_step_config['loss'])
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "scheduler_checkpoint.pth"))
    



def normalize_torch(data):
    means = data.mean(dim=0, keepdim=True)
    stds = data.std(dim=0, keepdim=True)
    norm_data = (data - means)/stds
    return norm_data

def normalize_torch_gt(data, mean_std_data):
    
    means = mean_std_data[0]
    stds = mean_std_data[1]
    norm_data = (data - means)/stds
    return norm_data

def loss_fn(inputs, network_output, model, loss_mode, 
            loss_params, frame_num, mean_std_data=None,
            prev_tar_accel=None, prev_pred_accel=None,
            prev_tar_vel=None, prev_pred_vel=None, prev_gt_tar_accel=None,
            prev_gt_tar_vel=None, epoch_in_reg=False):
    """L2 loss on position."""
    # build target acceleration
    cloth_pos = inputs['cloth_pos']
    prev_cloth_pos = inputs['prev|cloth_pos'][0]
    target_cloth_pos = inputs['target|cloth_pos']

    cur_position = cloth_pos
    prev_position = prev_cloth_pos
    target_position = target_cloth_pos

    use_fps = loss_params['use_fps']
    frame_time = 1/loss_params['fps']

    if loss_params['pred_vel'] == False:
        if use_fps == False:
            target_acceleration = target_position - 2 * cur_position + prev_position
        else:
            target_acceleration = (target_position - 2 * cur_position + prev_position)/(frame_time * frame_time)
        target_normalized = model.get_output_normalizer()(target_acceleration).to(device)
    else:
        if use_fps == False:
            target_vel = target_position - cur_position
        else:
            target_vel = (target_position - cur_position)/frame_time
        target_normalized = model.get_output_normalizer()(target_vel).to(device)
    #define loss dictionary
    loss_dict = {}

    # build loss
    node_type = inputs['node_type'].to(device)
    loss_mask = ~torch.eq(node_type[:, 0], torch.tensor([utils.get_handle_label()], device=device).int())
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    total_loss = torch.mean(error[loss_mask])
    
    if loss_params['pred_vel'] == False:
        loss_dict['accel'] = total_loss.item()
    else:
        loss_dict['vel'] = total_loss.item()

    unnormalized_output = model.get_output_normalizer().inverse(network_output)
    
    if loss_params['pred_vel'] == False:
        if use_fps == False:
            pred_position = 2 * cloth_pos + unnormalized_output - prev_cloth_pos
        else:
            pred_position = 2 * cloth_pos + (unnormalized_output * frame_time * frame_time) - prev_cloth_pos
    else:
        if use_fps == False:
            pred_position = cloth_pos + unnormalized_output
        else:
            pred_position = cloth_pos + (unnormalized_output * frame_time)
    #mask the handles prediction position to target
    pred_position = torch.where(loss_mask.reshape((-1,1)), torch.squeeze(pred_position), torch.squeeze(target_position))
    pred_rel_cloth_pos = utils.get_rel_cloth_pos(pred_position, inputs['senders'], inputs['receivers'])
    tar_rel_cloth_pos = utils.get_rel_cloth_pos(target_position, inputs['senders'], inputs['receivers'])
    
    pred_edge_len = torch.norm(pred_rel_cloth_pos, dim=-1, keepdim=True)
    tar_edge_len = torch.norm(tar_rel_cloth_pos, dim=-1, keepdim=True)

    if 'rel_pos' in loss_mode:
        loss_key = 'rel_pos'
        if loss_mode[loss_key]['norm'] == None:
            if loss_params['use_fixed_mean_std'] == False:
                #compute loss
                error_rel_cloth_pos = torch.sum((normalize_torch(tar_rel_cloth_pos) - normalize_torch(pred_rel_cloth_pos))**2, dim=1)
            else: 
                error_rel_cloth_pos = torch.sum((normalize_torch_gt(tar_rel_cloth_pos, mean_std_data=mean_std_data[0]) 
                                        - normalize_torch_gt(pred_rel_cloth_pos, mean_std_data=mean_std_data[0]))**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v1":
            normalized_pred_rel_cloth_pos = model.get_loss_normalizer(loss_key)(pred_rel_cloth_pos)
            normalized_tar_rel_cloth_pos = model.get_loss_normalizer(loss_key)(tar_rel_cloth_pos, accumulate=False)
            error_rel_cloth_pos = torch.sum((normalized_tar_rel_cloth_pos - 
                                                normalized_pred_rel_cloth_pos)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v2":
            normalized_pred_rel_cloth_pos = model.get_loss_normalizer(loss_key)(pred_rel_cloth_pos)
            normalized_tar_rel_cloth_pos = model.get_loss_normalizer(loss_key)(tar_rel_cloth_pos)
            error_rel_cloth_pos = torch.sum((normalized_tar_rel_cloth_pos - 
                                                normalized_pred_rel_cloth_pos)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v3":
            normalized_tar_rel_cloth_pos = model.get_loss_normalizer(loss_key)(tar_rel_cloth_pos)
            normalized_pred_rel_cloth_pos = model.get_loss_normalizer(loss_key)(pred_rel_cloth_pos, accumulate=False)
            error_rel_cloth_pos = torch.sum((normalized_tar_rel_cloth_pos - 
                                                normalized_pred_rel_cloth_pos)**2, dim=1)
        
        loss_rel_cloth_pos = torch.mean(error_rel_cloth_pos) * loss_mode[loss_key]['weight']
        loss_dict[loss_key] = loss_rel_cloth_pos.item()
        total_loss = total_loss + loss_rel_cloth_pos

    if 'edge_length' in loss_mode:
        loss_key = "edge_length"
        if loss_mode[loss_key]['norm'] == None:
            if loss_params['use_fixed_mean_std'] == False:
                error_edge_len = torch.sum((normalize_torch(tar_edge_len) - normalize_torch(pred_edge_len))**2, dim=1)
            else:
                error_edge_len = torch.sum((normalize_torch_gt(tar_edge_len, mean_std_data=mean_std_data[1]) 
                                    - normalize_torch_gt(pred_edge_len, mean_std_data=mean_std_data[1]))**2, dim=1)
            #error_edge_len = normalize_torch(torch.sum(((tar_edge_len) - (pred_edge_len))**2, dim=1))
        elif loss_mode[loss_key]['norm'] == "v1":
            norm_pred_edge_len = model.get_loss_normalizer(loss_key)(pred_edge_len)
            norm_tar_edge_len = model.get_loss_normalizer(loss_key)(tar_edge_len, accumulate=False)
            error_edge_len = torch.sum((norm_tar_edge_len - 
                                        norm_pred_edge_len)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v2":
            norm_pred_edge_len = model.get_loss_normalizer(loss_key)(pred_edge_len)
            norm_tar_edge_len = model.get_loss_normalizer(loss_key)(tar_edge_len)
            error_edge_len = torch.sum((norm_tar_edge_len - 
                                        norm_pred_edge_len)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v3":
            norm_tar_edge_len = model.get_loss_normalizer(loss_key)(tar_edge_len)
            norm_pred_edge_len = model.get_loss_normalizer(loss_key)(pred_edge_len, accumulate=False)
            error_edge_len = torch.sum((norm_tar_edge_len - 
                                        norm_pred_edge_len)**2, dim=1)
        loss_edge_len = torch.mean(error_edge_len) * loss_mode[loss_key]['weight']
        loss_dict[loss_key] = loss_edge_len.item()
        total_loss = total_loss + loss_edge_len
    
    
    if 'theta' in loss_mode:
        loss_key = 'theta'
        tar_gt_rel_cloth_pos = utils.get_rel_cloth_pos(inputs['target_gt|cloth_pos'], inputs['senders'], inputs['receivers'])
        pred_theta = utils.compute_dihedral_angle_fast(inputs['senders'], inputs['receivers'],
                            inputs['opposite_v'], pred_rel_cloth_pos, pred_position, "signed").reshape([-1,1])
        tar_theta = utils.compute_dihedral_angle_fast(inputs['senders'], inputs['receivers'],
                            inputs['opposite_v'], tar_gt_rel_cloth_pos, inputs['target_gt|cloth_pos'], "signed").reshape([-1,1])
        if loss_mode[loss_key]['norm'] == None:
            error_theta= torch.sum((normalize_torch(tar_theta) - normalize_torch(pred_theta))**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v1":
            norm_pred_theta = model.get_loss_normalizer(loss_key)(pred_theta)
            norm_tar_theta = model.get_loss_normalizer(loss_key)(tar_theta, accumulate=False)
            error_theta = torch.sum((norm_tar_theta - 
                                        norm_pred_theta)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v2":
            norm_pred_theta = model.get_loss_normalizer(loss_key)(pred_theta)
            norm_tar_theta = model.get_loss_normalizer(loss_key)(tar_theta)
            error_theta= torch.sum((norm_tar_theta - 
                                        norm_pred_theta)**2, dim=1)
        elif loss_mode[loss_key]['norm'] == "v3":
            norm_tar_theta = model.get_loss_normalizer(loss_key)(tar_theta)
            norm_pred_theta = model.get_loss_normalizer(loss_key)(pred_theta, accumulate=False)
            error_theta = torch.sum((norm_tar_theta - 
                                        norm_pred_theta)**2, dim=1)
        
        loss_theta = torch.mean(error_theta) * loss_mode[loss_key]['weight']
        loss_dict[loss_key] = loss_theta.item()
        total_loss = total_loss + loss_theta

    
    if 'ke_fixed' in loss_mode:
        loss_key = "ke_fixed"
        if use_fps == False:
            tar_velocity = inputs['target_gt|cloth_pos'] - inputs['cloth_pos_gt']
        else:
            tar_velocity = (inputs['target_gt|cloth_pos'] - inputs['cloth_pos_gt'])/frame_time
        tar_ke = 0.5 * inputs['node_mass'] * torch.sum(tar_velocity * tar_velocity, dim=1, keepdim=True)
        
        if use_fps == False:
            pred_velocity = pred_position - cur_position
        else:
            pred_velocity = (pred_position - cur_position)/frame_time
        pred_ke_unnormalized = 0.5 * inputs['node_mass'] * torch.sum(pred_velocity * pred_velocity, dim=1, keepdim=True)

        if loss_mode[loss_key]['norm'] == "v3":
            tar_ke = model.get_loss_normalizer(loss_key)(tar_ke).to(device)
            pred_ke = model.get_loss_normalizer(loss_key)(pred_ke_unnormalized, accumulate=False).to(device)



        error_ke = torch.sum((tar_ke - pred_ke)**2, dim=1)
        if loss_params['ke_fixed_loss_mask_change'] == False:
            loss_ke = torch.mean(error_ke[loss_mask]) * loss_mode[loss_key]['weight']
        else:
            loss_ke = torch.mean(error_ke) * loss_mode[loss_key]['weight']
        loss_dict[loss_key] = loss_ke.item()
        total_loss = total_loss + loss_ke



    return total_loss, loss_dict


def main(argv):
    run_step_config = utils.read_json_file(FLAGS.params, 'train')
    


    # load config from previous run step if last run dir is specified
    last_run_dir = run_step_config['model_last_run_dir']
    
   

    # setup directory structure for saving checkpoint, train configuration, rollout result and log
    output_dir = os.path.join('output', "basic_cloth")
    run_step_dir = summary.prepare_files_and_directories(last_run_dir, output_dir, 
            run_step_config['is_save_output_on'], run_step_config['name_code'])
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')

    # setup logger
    root_logger = summary.logger_setup(os.path.join(log_dir, 'log.log'))

    
    if last_run_dir is None:
       
        run_step_config['last_run_dir'] = None

        root_logger.info("=========================================================")
        root_logger.info("Start new run in " + str(run_step_dir))
        root_logger.info("=========================================================")
    if last_run_dir is not None:
        run_step_config['last_run_dir'] = last_run_dir 
        run_step_config['last_run_step_dir'] = summary.find_nth_latest_run_step(last_run_dir, 2)
    
    
    run_step_config['checkpoint_dir'] = checkpoint_dir
    run_step_config['log_dir'] = log_dir

    run_step_config_save_path = os.path.join(log_dir, 'config.pkl')
    Path(run_step_config_save_path).touch()
    summary.pickle_save(run_step_config_save_path, run_step_config)



    # create or load model
    root_logger.info("Start training......")
    model = cloth_model_original.Model(run_step_config)
    if last_run_dir is not None:
        last_run_step_dir = summary.find_nth_latest_run_step(last_run_dir, 2)
        if(run_step_config['epoch_checkpoint'] == -1):
            model.load_model(os.path.join(last_run_step_dir, 'checkpoint')+ "/epoch_model_checkpoint", 
                        run_step_config['has_global'], 
                        run_step_config['loss'],
                        run_step_config['new_json'])
        else:
            model.load_model(os.path.join(last_run_step_dir, 'checkpoint')+ "/epoch_model_checkpoint_" + str(run_step_config['epoch_checkpoint']), 
                        run_step_config['has_global'], 
                        run_step_config['loss'],
                        run_step_config['new_json'])
        model.learned_model.encoder.has_global = run_step_config['has_global']
        model.learned_model.processor.has_global = run_step_config['has_global']
        for block in model.learned_model.processor.graphnet_blocks:
            block.has_global = run_step_config['has_global']
        # model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info("Loaded checkpoint file in " + str(os.path.join(last_run_step_dir, 'checkpoint')) + " and starting retraining...")
    model.to(device)

    # run summary
    summary.log_run_summary(root_logger, run_step_config)


    train_start = time.time()

    learner(model, run_step_config)

    train_end = time.time()

    train_elapsed_time_in_second = train_end - train_start
    train_elapsed_time = str(datetime.timedelta(seconds=train_elapsed_time_in_second))
    root_logger.info("total train elapsed time is! ", train_elapsed_time)
    root_logger.info("Finished training......")

if __name__ == '__main__':

    app.run(main)
