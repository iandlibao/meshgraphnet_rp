import os
def get_run_step_config(run_step_config, json_params, mode):

    if mode == 'train':
        run_step_config['is_training'] = True
        run_step_config['wandb_project_name'] = set_def_val('wandb_project_name', 'mesh_1024_min', json_params)
        run_step_config['epochs'] = json_params['epochs']
        run_step_config['have_noise'] = set_def_val('have_noise', True, json_params)
        run_step_config['supervised_epochs'] = json_params['schedule_sampling']['supervised_epochs']
        run_step_config['regression_epochs'] = json_params['schedule_sampling']['regression_epochs']
        run_step_config['max_regress_count'] = json_params['schedule_sampling']['max_regress_count']
        run_step_config['p_mode'] = json_params['schedule_sampling']['p_mode']
        run_step_config['low_p_thresh'] = json_params['schedule_sampling']['low_p_thresh']
        if json_params['model_last_run_dir'] is not None:
            run_step_config['epoch_checkpoint'] = json_params['epoch_checkpoint']
            run_step_config['model_last_run_dir'] = os.path.join("output", "basic_cloth", 
                    json_params['model_last_run_dir'])
        else:
            run_step_config['model_last_run_dir'] = json_params['model_last_run_dir']
        run_step_config["new_json"] = set_def_val("new_json", True, json_params)


    elif mode == 'test':
        run_step_config['is_training'] = False
        run_step_config['epoch_last_checkpoint'] = json_params['runtime']['epoch_last_checkpoint']
        run_step_config['output_folder'] = json_params['runtime']['output_folder']
        run_step_config['output_number'] = json_params['runtime']['output_number']

        run_step_config['epoch_number'] = set_def_val('epoch_number', -1, json_params['runtime'])
        run_step_config["runfilename"] = set_def_val("runfilename", None, json_params['runtime'])
        run_step_config["generalized_motion"] = set_def_val("generalized_motion", None, json_params['runtime'])
        run_step_config["time_acc"] = set_def_val("time_acc", False, json_params['runtime'])
        run_step_config["new_json"] = set_def_val("new_json", True, json_params)
        run_step_config["starting_frame"] = set_def_val("starting_frame", 0, json_params['runtime'])
        run_step_config["quick_render"] = set_def_val("quick_render", False, json_params['runtime'])
        run_step_config["save_runtime_npy"] = set_def_val("save_runtime_npy", True, json_params['runtime'])
        run_step_config['rollout_length'] = set_def_val("rollout_length", -1, json_params['runtime'])
        run_step_config['runtime_mode'] = set_def_val("runtime_mode", None, json_params['runtime'])
        run_step_config['dir_code'] = set_def_val('dir_code', '', json_params['runtime'])
        run_step_config['motion_code'] = set_def_val('motion_code', '', json_params['runtime'])
        run_step_config['set_own_obj'] = set_def_val('set_own_obj', False, json_params['runtime'])
        run_step_config['own_obj_folder'] = set_def_val('own_obj_folder', None, json_params['runtime'])
        run_step_config['handle_ind'] = set_def_val('handle_ind', (0,19), json_params['runtime'])
        run_step_config['custom_folder'] = set_def_val('custom_folder', None, json_params['runtime'])
        run_step_config['runtime_input_folder'] = set_def_val('runtime_input_folder', None, json_params['runtime'])
    
    run_step_config['is_force_included'] = set_def_val('is_force_included', False, json_params)
    run_step_config['is_mesh_space'] = set_def_val('is_mesh_space', True, json_params)
    run_step_config['has_global'] = set_def_val('has_global', False, json_params)
    run_step_config['sample_start_point'] = set_def_val('sample_start_point', 0, json_params)
    run_step_config['val_every_epoch'] = json_params['val_every_epoch']
    run_step_config['save_epoch'] = json_params['save_epoch']
    run_step_config['output_size'] = set_def_val('output_size', 3, json_params)
    run_step_config['noise'] = set_def_val('noise', 0.003, json_params)
    run_step_config['gamma'] = set_def_val('gamma', 0.1, json_params)
    run_step_config['field'] = set_def_val('field', 'cloth_pos', json_params)
    run_step_config['history'] = set_def_val('history', True, json_params)
    run_step_config['output_size'] = set_def_val('output_size', 3, json_params)

    run_step_config['is_save_output_on'] = json_params['is_save_output_on']
    run_step_config['with_theta'] = set_def_val('with_theta', False, json_params)
    run_step_config['has_sf_ratio'] = set_def_val('has_sf_ratio', False, json_params)
    run_step_config['with_rel_mesh_pos'] = set_def_val('with_rel_mesh_pos', False, json_params)
    
    run_step_config['is_meshgraphnet_data'] = set_def_val('is_meshgraphnet_data', False, json_params)
    
    run_step_config['message_passing_aggregator'] = set_def_val('message_passing_aggregator', 'sum', json_params)
        #['sum', 'max', 'min', 'mean', 'pna']

    run_step_config['message_passing_steps'] = set_def_val('message_passing_steps', 15, json_params)
    
    run_step_config['attention'] = set_def_val('attention', False, json_params)   
    run_step_config['no_global_normalization'] = set_def_val('no_global_normalization', False, json_params)
    run_step_config['name_code'] = json_params['name_code']
    run_step_config['velocity_history'] = set_def_val('velocity_history', 1, json_params)   

    run_step_config["mesh_pos_mode"] = set_def_val("mesh_pos_mode", "2d", json_params)
    
    run_step_config["use_fext"] = set_def_val("use_fext", False, json_params)

    run_step_config["use_ke"] = set_def_val("use_ke", False, json_params)

    run_step_config["use_ke_version"] = set_def_val("use_ke_version", -1, json_params)
    
    run_step_config["node_type_length"] = set_def_val("node_type_length", 9, json_params)

    run_step_config["theta_mode"] = set_def_val("theta_mode", "unsigned", json_params)

    run_step_config["has_sf_ratio_2"] = set_def_val("has_sf_ratio_2", False, json_params)

    run_step_config['has_stretch_e_feature'] = set_def_val('has_stretch_e_feature', False, json_params)

    run_step_config['has_bend_e_feature'] = set_def_val('has_bend_e_feature', False, json_params)

    run_step_config['full_data_version'] = set_def_val('full_data_version', None, json_params)
    
    run_step_config['local_preprocessing'] = set_def_val('local_preprocessing', False, json_params)

    run_step_config['ke_fixed_loss_mask_change'] = set_def_val('ke_fixed_loss_mask_change', False, json_params)

    run_step_config['use_fps'] = set_def_val('use_fps', False, json_params)

    run_step_config['fps'] = set_def_val('fps', 60., json_params)

    run_step_config['eval_data_ver'] = set_def_val('eval_data_ver', "eval", json_params)

    run_step_config['with_new_labels'] = set_def_val('with_new_labels', None, json_params)

    run_step_config['with_rnn_encoder'] = set_def_val('with_rnn_encoder', None, json_params)

    run_step_config['no_3d_rest_vector'] = set_def_val('no_3d_rest_vector', False, json_params)

    run_step_config['global_version'] = set_def_val('global_version', None, json_params)
    run_step_config['global_features_size'] = set_def_val('global_features_size', 3, json_params)
    run_step_config['global_latent_size'] = set_def_val('global_latent_size', 128, json_params)

    run_step_config['global_model_in_processor'] = set_def_val('global_model_in_processor', True, json_params)
    run_step_config['loss'] = json_params['loss']


    run_step_config = set_loss_params(run_step_config, json_params)

    
    run_step_config['loss'] = set_loss_dict_params(run_step_config['loss'])

    #some assertions
    if "a_from_pred_v" in run_step_config['loss'].keys():
        assert(run_step_config['loss_params']['pred_vel'] == True)
    if "gt_pos_hist" in run_step_config['loss'].keys():
        assert(run_step_config['loss_params']['pred_vel'] == True)
    if "bend_e" in run_step_config['loss'].keys():
        assert(run_step_config['theta_mode'] == "signed")
    if run_step_config['full_data_version'] is not None:
        assert(json_params['input_data_folders'] is None)
    else:
        assert(json_params['input_data_folders'] is not None)
    if run_step_config['full_data_version'] == 'v2':
        assert(run_step_config['local_preprocessing'] == True)
    if run_step_config['loss_params']['ke_fixed_loss_mask_change'] == True:
        assert("ke_fixed" in run_step_config['loss'].keys())
    if run_step_config['use_fps'] == True:
        assert("gt_vel_hist" not in run_step_config['loss'].keys())
        assert("gt_accel_hist" not in run_step_config['loss'].keys())
        assert("gt_pos_hist" not in run_step_config['loss'].keys())
        assert("a_from_pred_v" not in run_step_config['loss'].keys())

    if mode == "test":
        if run_step_config['set_own_obj'] == True:
            assert(run_step_config['own_obj_folder'] is not None)
            assert(run_step_config['handle_ind'] is not None)
    if(run_step_config['with_rel_mesh_pos'] == True):
        assert(run_step_config['is_mesh_space'] == False)
    if(run_step_config['is_mesh_space'] == True):
        assert(run_step_config['with_rel_mesh_pos'] == False)
    
    if run_step_config['with_rnn_encoder'] is not None:
        run_step_config['velocity_history'] = run_step_config['with_rnn_encoder']
    if run_step_config['with_new_labels'] is not None:
        run_step_config['is_new_labels'] = True
    else:
        run_step_config['is_new_labels'] = False

    
    if json_params['input_data_folders'] is not None:
        input_data_folders = []
        for input_data_folder in json_params['input_data_folders']:
            input_data_folders.append(os.path.join("input", "processed_npy", input_data_folder))
        run_step_config['input_data_folders'] = input_data_folders
    
    if "inertia_v1" in run_step_config['loss'].keys():
        assert(run_step_config['loss_params']['use_fps'] == False)


    # if run_step_config['global_model_in_processor'] is not None:
    #     assert(run_step_config['has_global'] == True)


    return run_step_config

def set_loss_dict_params(run_step_config_loss):
    for key,val in run_step_config_loss.items():
        if "in_reg" not in val:
            run_step_config_loss[key]["in_reg"] = False
    
    return run_step_config_loss

def set_loss_params(run_step_config, json_params):
    
    run_step_config['loss_params'] = {}
    run_step_config['loss_params']['fixed_sf_ratio'] = set_def_val("fixed_sf_ratio", False, json_params)
    run_step_config['loss_params']['decaying_loss_weight'] = set_def_val('decaying_loss_weight', False, json_params)
    run_step_config['loss_params']['frame_num_loss_weight'] = set_def_val('frame_num_loss_weight', False, json_params)
    run_step_config['loss_params']['use_fixed_mean_std'] = set_def_val('use_fixed_mean_std', False, json_params)
    run_step_config['loss_params']['ke_fixed_loss_mask_change'] = set_def_val('ke_fixed_loss_mask_change', False, json_params)
    run_step_config['loss_params']['use_fps'] = set_def_val('use_fps', False, json_params)
    run_step_config['loss_params']['fps'] = set_def_val('fps', 60., json_params)
    
    if 'vel' in run_step_config['loss'].keys():
        run_step_config['loss_params']['pred_vel'] = True
    elif 'accel' in run_step_config['loss'].keys():
        run_step_config['loss_params']['pred_vel'] = False
    else:
        assert(False)


    
    return run_step_config
        

def set_def_val(key, def_val, json_params):
    if key in json_params.keys():
        return json_params[key]
    else:
        return def_val