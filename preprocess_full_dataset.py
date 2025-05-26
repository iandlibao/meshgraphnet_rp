import numpy as np
import os
import torch
import logging
from new_runtime_code import get_path_from_gt_input

def get_npy_data_from_motion_code(motion_code):
    obj_code = "square_1024"
    gt_path = get_path_from_gt_input({"motion_code": motion_code, "obj_code": obj_code})

    cloth_pos = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy")))
    
    npy_data = {
        "cloth_pos": cloth_pos
    }
    
    return npy_data

def get_data(mode, test_train):

    if test_train == "train":
        num_frames = 1261
    elif test_train == "test":
        num_frames = 1261


    motion_list = []

    if mode == "v1":
        motion_list.append("fwd")
        motion_list.append("fwd_opp")
        motion_list.append("side")
        motion_list.append("side_opp")
        motion_list.append("updown")
        motion_list.append("updown_opp")
        motion_list.append("xy")
        motion_list.append("xy_opp")
        motion_list.append("xyz")
        motion_list.append("xyz_opp")
        motion_list.append("xz")
        motion_list.append("xz_opp")
        motion_list.append("yz")
        motion_list.append("yz_opp")
    elif mode == "v2":
        motion_list.append("fwd")
        motion_list.append("fwd_opp")
        motion_list.append("side")
        motion_list.append("side_opp")
        motion_list.append("updown")
        motion_list.append("updown_opp")
        motion_list.append("xy")
        motion_list.append("xy_opp")
        motion_list.append("xyz")
        motion_list.append("xyz_opp")
        motion_list.append("xz")
        motion_list.append("xz_opp")
        motion_list.append("yz")
        motion_list.append("yz_opp")
        motion_list.append("rot_h0_h1")
        motion_list.append("rot_h0_h1_opp")
    elif mode == "eval":
        motion_list.append("xy_v2")
        motion_list.append("xy_v2_opp")
        motion_list.append("yz_v2")
        motion_list.append("yz_v2_opp")
        motion_list.append("xz_v2")
        motion_list.append("xyz_v2")
        motion_list.append("xyz_v2_opp")
        motion_list.append("xyz_v3")
        motion_list.append("xyz_v3_opp")
        motion_list.append("xyz_v4")
        motion_list.append("xyz_v4_opp")
        motion_list.append("rot_h0")
        motion_list.append("rot_h0_opp")
        motion_list.append("rot_h1")
        motion_list.append("rot_h1_opp")
    


    full_dataset = {}

    for motion in motion_list:
        
            
        npy_file = get_npy_data_from_motion_code(motion)

       

        for key, val in npy_file.items():
            assert(val.shape[0] == num_frames)

        
            
        #for starting
        if motion == "fwd":
            full_dataset['ss_start'] = {}
            for key, val in npy_file.items():
                full_dataset["ss_start"][key] = val[0:52]
        
        #for rest to motion
        full_dataset_key = "r2m_" + motion
        full_dataset[full_dataset_key] = {}
        for key, val in npy_file.items():
            full_dataset[full_dataset_key][key] = val[50:162]   

        #for steady state motion
        if motion[-4:] != "_opp":
            full_dataset_key = "ssm_" + motion
            full_dataset[full_dataset_key] = {}
            for key, val in npy_file.items():
                full_dataset[full_dataset_key][key] = val[250:532]
        
        #for motion to rest
        full_dataset_key = "m2r_" + motion
        full_dataset[full_dataset_key] = {}
        for key, val in npy_file.items():
            full_dataset[full_dataset_key][key] = val[650:802]

        #for ending
        if motion == "fwd":
            full_dataset['ss_end'] = {}
            for key, val in npy_file.items():
                full_dataset['ss_end'][key] = val[1100:1202]
            


    #check the total frames
    num_frames = 0
    for key, val in full_dataset.items():
       num_frames += val['cloth_pos'].shape[0] - 2

    root_logger = logging.getLogger()
    # print("num_frames: ", num_frames)
    root_logger.info("num_frames: " + str(num_frames))
    return full_dataset


if __name__ == '__main__':
    full_dataset = get_data("v1", "train")
    print("foo")