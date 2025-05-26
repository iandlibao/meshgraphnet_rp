import torch
import numpy as np

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

def rotate_x_dir_values(x_values, direction):
    x_values = x_values.float()

    dir_vec = direction/torch.norm(direction)

    if not torch.equal(direction,torch.tensor([1.,0.,0.])) and not torch.equal(direction,torch.tensor([-1.,0.,0.])):

        x_vec = torch.tensor([1., 0., 0.])

        axis = torch.cross(x_vec, dir_vec)

        cosine = torch.dot(x_vec, dir_vec)

        skew_mat = torch.tensor([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]], 
                [-axis[1], axis[0], 0]])

        r_mat = torch.eye(3) + skew_mat + torch.matmul(skew_mat, skew_mat) * (1/(1+cosine))
        r_mat = r_mat.to(device)

    #rotated pos
    if not torch.equal(direction,torch.tensor([1.,0.,0.])) and not torch.equal(direction,torch.tensor([-1.,0.,0.])):
        rotated_x_values = torch.matmul(r_mat, x_values).transpose(0,1)
    elif torch.equal(direction,torch.tensor([1.,0.,0.])):
        rotated_x_values = x_values.transpose(0,1)
    elif torch.equal(direction,torch.tensor([-1.,0.,0.])):
        rotated_x_values = x_values.transpose(0,1) * -1

    return rotated_x_values

def new_translate_frames(pos_traj, direction):
    pos_traj = torch.stack([pos_traj, torch.zeros((pos_traj.shape[0])).to(device), 
            torch.zeros((pos_traj.shape[0])).to(device)])
        
    pos_traj = pos_traj.to(device)

    pos_traj = rotate_x_dir_values(pos_traj, direction)
    
    return pos_traj

def compute_theta_start(handle_orientation, center_of_rotation, handle_code):
    x = handle_code - center_of_rotation

    if x <= 0:
        theta_start = handle_orientation
    else:
        theta_start = handle_orientation + np.pi

    return theta_start

def compute_theta_start_from_theta_end(theta_end, next_cor, prev_cor, handle_code):
    if prev_cor < next_cor:
        if handle_code < next_cor and handle_code > prev_cor:
            theta_start = theta_end + np.pi
        else:
            theta_start = theta_end
    else:
        if handle_code < prev_cor and handle_code > next_cor:
            theta_start = theta_end + np.pi
        else:
            theta_start = theta_end
    return theta_start


def compute_handle_orientation(last_ref_h_pos):

    return torch.atan2(last_ref_h_pos[1], last_ref_h_pos[0])

def new_rotate_frames_get_3d(theta_traj, theta_start, handle_radius):
    theta_time_line = theta_traj + theta_start
    x_pos = handle_radius * torch.cos(theta_time_line) - handle_radius * torch.cos(theta_start) #+ initial_pos[0].item()
    y_pos = 1 * handle_radius * torch.sin(theta_time_line) - handle_radius * torch.sin(theta_start) #+ initial_pos[2].item()
    z_pos = 0 * theta_time_line #+ initial_pos[1].item()
    handle_pos = torch.stack((x_pos, z_pos, y_pos)).transpose(0,1)
    
    return handle_pos

def new_rotate_frames(pos_traj, center_of_rotation, radius, handle_code, theta_start, cw_ccw):
    #assume center_of_rotation lies on the segment of the topology
    
    if cw_ccw == "cw":
        pos_traj = -pos_traj
    
    relative_radius = abs(center_of_rotation - handle_code)
    handle_radius = radius * relative_radius

    #solve the max radius
    max_radius = solve_max_radius(center_of_rotation, radius)

    theta_traj = pos_traj / max_radius
    theta_traj = theta_traj.to(device)

    handle_pos = new_rotate_frames_get_3d(theta_traj, theta_start, handle_radius)

    
    return handle_pos


def solve_max_radius(cor, radius):
    he_radius_list = []
    handle_ends_code_list = [-1,1]
    for handle_end in handle_ends_code_list:
        he_relative_radius = abs(cor - handle_end)
        he_radius = radius * he_relative_radius
        he_radius_list.append(he_radius)
    max_radius = max(he_radius_list)
    return max_radius


def make_vel_traj(start, amplitude, time_range, mode, fps):
    time_skip = 1/fps
    time_start = 1/fps
    time_end = time_start + time_range
    time_line = np.arange(time_start, time_end, time_skip)

    if mode == 'linear':
        val_array = (amplitude/time_range) * time_line + start    
    elif mode == 'smooth':
        val_array = -(amplitude/2) * (np.cos((time_line*np.pi)/time_range)-1) + start

    return val_array

def block_postprocess(frame_pos_list, current_handle_pos_list,
        handle_code_list):
    adjusted_frame_pos_list = []
    for frame_pos, current_handle_pos in zip(frame_pos_list, current_handle_pos_list):
        adjusted_frame_pos = frame_pos + current_handle_pos
        adjusted_frame_pos_list.append(adjusted_frame_pos)

    current_handle_pos_list = []
    for frame_pos in adjusted_frame_pos_list:
        current_handle_pos_list.append(frame_pos[-1])

    #compute theta start based on the position of the end handles
    ref_handle_pos = (current_handle_pos_list[handle_code_list.index(-1)] - 
        current_handle_pos_list[handle_code_list.index(1)])
    ref_handle_pos = torch.tensor([ref_handle_pos[0], ref_handle_pos[2]])
    radius = torch.norm(ref_handle_pos).item()/2
    handle_orientation = compute_handle_orientation(ref_handle_pos)

    return adjusted_frame_pos_list, current_handle_pos_list, radius, handle_orientation

def compute_pos_traj(time, distance, velocity, fps):
    '''
    Given (2 of these):
        time
        distance
        velocity - "start" "end" "linear/smooth"
    Case 1:
        -given time and distance
            -fit in different types of velocity (linear, constant and smooth)
    Case 2:
        -given time and velocity
            -easy
    Case 3:
        -given distance and velocity
            -fit in the time that satisfies 3 types of velocity (linear, constant, smooth)
    '''
    if time == None:
        assert(distance is not None and velocity is not None)
        time_new = (2 * distance) / (fps* (velocity['start'] + velocity['end']))
        vel_traj = make_vel_traj(velocity['start'], velocity['end']-velocity['start'], time_new,
            velocity['mode'], fps)
        pos_traj = compute_pos_from_v_traj(vel_traj, 0.)
    elif distance == None:
        assert(time is not None and velocity is not None)
        #make the vel traj from velocity
        vel_traj = make_vel_traj(velocity['start'], velocity['end'] - velocity['start'], time,
                velocity['mode'], fps)
        #make pos traj from vel traj
        pos_traj = compute_pos_from_v_traj(vel_traj, 0.)
        
    elif "start" not in velocity.keys() and "end" not in velocity.keys() and "mode" in velocity.keys():
        assert(time is not None and distance is not None)
        pos_traj = make_vel_traj(0, distance, time, velocity['mode'], fps)
    
    return torch.from_numpy(pos_traj).to(device)


def block_transl_rot(handle_orientation, radius,
    current_handle_pos_list, handle_code_list, fps, input_dict):
    
    if input_dict['rot_dict'] is not None:
        center_of_rotation = input_dict['rot_dict']['cor']
        ccw_cw = input_dict['rot_dict']['ccw_cw']
    else:
        center_of_rotation = 0
        ccw_cw = "ccw"


    theta_start_list = []
    for handle_code in handle_code_list:
        theta_start_list.append(compute_theta_start(handle_orientation, center_of_rotation, handle_code))

    not_none_key_list = []
    output_dict = {}
    for input_key, input_val in input_dict.items():
        if input_val is not None:
            not_none_key_list.append(input_key)
            if input_key == 'transl_dict' or input_key == 'sec_transl_dict':
                transl_traj_list = []
                transl_dir_list = []
                for _ in handle_code_list:
                    transl_traj_list.append(compute_pos_traj(input_val['time'], input_val['distance'],
                        input_val['velocity'], fps))
                    transl_dir_list.append(input_val['direction'])
                output_dict[input_key] = {'transl_traj_list': transl_traj_list,
                    'transl_dir_list': transl_dir_list}
            elif input_key == 'rot_dict':
                #make rot traj
                if input_val['distance'] is not None:
                    rot_rad = input_val['distance'] * (np.pi / 180)
                    max_radius = solve_max_radius(center_of_rotation, radius)
                    rot_distance = rot_rad * max_radius
                else:
                    rot_distance = None
                rot_traj = compute_pos_traj(input_val['time'], rot_distance, input_val['velocity'],
                    fps)
                output_dict[input_key] = {'rot_traj': rot_traj}
        else:
            output_dict[input_key] = None

    assert(len(not_none_key_list) > 0)
    not_none_key = not_none_key_list[0]
    for output_key, output_val in output_dict.items():
        if output_val is None:
            if not_none_key == 'transl_dict' or not_none_key == 'sec_transl_dict':
                if output_key == 'transl_dict' or output_key == 'sec_transl_dict':
                    transl_traj_list = []
                    for transl_traj in output_dict[not_none_key]['transl_traj_list']:
                        transl_traj_list.append(transl_traj * 0)
                    output_dict[output_key] = {'transl_traj_list': transl_traj_list,
                        'transl_dir_list': output_dict[not_none_key]['transl_dir_list']}
                elif output_key == 'rot_dict':
                    output_dict[output_key] = {'rot_traj': output_dict[not_none_key]['transl_traj_list'][0] * 0}
            elif not_none_key == 'rot_dict':
                assert(output_key != 'rot_dict')
                if output_key == 'transl_dict' or output_key == 'sec_transl_dict':
                    transl_traj_list = []
                    transl_dir_list = []
                    for _ in handle_code_list:
                        transl_traj_list.append(output_dict[not_none_key]['rot_traj'] * 0)
                        transl_dir_list.append(torch.tensor([1.,0.,0.]))
                    output_dict[output_key] = {'transl_traj_list': transl_traj_list,
                        'transl_dir_list': transl_dir_list}
            
            

    frame_pos_list = []
    for (transl_traj, transl_dir, second_transl_traj, second_transl_dir, 
            handle_code, theta_start) in zip(output_dict['transl_dict']['transl_traj_list'], 
            output_dict['transl_dict']['transl_dir_list'], 
            output_dict['sec_transl_dict']['transl_traj_list'], 
            output_dict['sec_transl_dict']['transl_dir_list'],
            handle_code_list, theta_start_list):
        transl_frame_handle_pos = new_translate_frames(transl_traj, transl_dir)
        second_transl_frame_handle_pos = new_translate_frames(second_transl_traj, second_transl_dir)
        rot_frame_handle_pos= new_rotate_frames(output_dict['rot_dict']['rot_traj'], center_of_rotation,
                radius, handle_code, theta_start, ccw_cw)
        frame_handle_pos = transl_frame_handle_pos + rot_frame_handle_pos + second_transl_frame_handle_pos
        frame_pos_list.append(frame_handle_pos)

    adjusted_frame_pos_list, current_handle_pos_list, radius, handle_orientation = block_postprocess(frame_pos_list,
            current_handle_pos_list, handle_code_list)

    return adjusted_frame_pos_list, current_handle_pos_list, radius, handle_orientation


def make_input_dict_from_motion_code(motion_code):
    def rest_dict(time):
        return {"transl_dict": default_transl_dict(time), 
            "sec_transl_dict": None, 
            "rot_dict": None}
    def translate_dict(time, transl_dir, transl_d):
        return {"transl_dict": make_transl_dict(time, transl_d, {"mode": "linear"}, transl_dir), 
            "sec_transl_dict": None, 
            "rot_dict": None}
    def input_rot_dict(time, rot_d, cor, ccw_cw):
        return {"time": time, "rot_d": rot_d, "cor": cor, "ccw_cw": ccw_cw}
    def transl_rot_dict(time, transl_dir, transl_d, rot_d, cor, ccw_cw):
        return {"time": time, "transl_dir": transl_dir, "transl_d": transl_d,
            "rot_d": rot_d, "cor":cor, "ccw_cw": ccw_cw}

    def default_transl_dict(time):
        return make_transl_dict(time, None, {"start": 0, "end": 0, "mode": "linear"}, torch.tensor([1.,0.,0.]))
    def default_rot_dict(time):
        return make_rot_dict(time, None, {"start": 0, "end": 0, "mode": "linear"}, 0, "ccw")
    def make_transl_dict(time, distance, velocity, direction):
        return {"time": time, "distance": distance, "velocity": velocity, "direction": direction}
    def make_rot_dict(time, distance, velocity, cor, ccw_cw):
        return {"time": time, "distance": distance, "velocity": velocity, "cor": cor, "ccw_cw": ccw_cw}
    
    # def _transl_back_forth(period=1, mode=):
    #     print("foo")
    ####keys
    '''
    time -> seconds
    ----------translation params-----------------
    transl_dir -> direction of first translation
    transl_d -> distance of first translation
    transl_v_traj -> velocity traj of the first translation

    sec_transl_dir
    sec_transl_d
    sec_transl_v_traj

    ----------rotation params-----------------------
    rot_d -> distance of rotation
    cor -> center of rotation
    ccw_cw -> counter clockwise or clockwise
    rot_v_traj -> velocity traj

    **note: you either use distance or the velocity trajectory**
    '''
    
    direction_dict = {
        "fwd": torch.tensor([1., 0., 0.]),
        "side": torch.tensor([0.,0.,1.]),
        "updown": torch.tensor([0., 1., 0.]),
        "xy": torch.tensor([1.,0.,1.]),
        "yz": torch.tensor([0.,1.,1.]),
        "xz": torch.tensor([1.,1.,0.]),
        "xyz": torch.tensor([1.,1.,1.]),
        "xy_v2": torch.tensor([1.,0.,-1.]),
        "yz_v2": torch.tensor([0.,1.,-1.]),
        "xz_v2": torch.tensor([1.,-1.,0.]),
        "xyz_v2": torch.tensor([1.,1.,-1.]),
        "xyz_v3": torch.tensor([1.,-1.,1.]),
        "xyz_v4": torch.tensor([-1.,1.,1.]),
    }
    opp_direction_dict = {}
    for key, val in direction_dict.items():
        opp_direction_dict[key + "_opp"] = -1 * val
    direction_dict.update(opp_direction_dict)

    input_dict_list = []
    input_dict_list.append(rest_dict(0.5))
    
    if motion_code in direction_dict.keys():
        # period = 3
        # transl_dir = direction_dict[motion_code]
        
        # for _ in range(period):
        #     input_dict_list.append(translate_dict(1, transl_dir, 1))
        #     input_dict_list.append(translate_dict(1, -1 * transl_dir, 1))
        # transl_max_v = 0.02
        period_time = 1
        period_distance = 1
        num_translations = 3
        transl_v_info = {"mode": "smooth"}
        
        for ind in range(num_translations):
            if ind%2 == 0:
                transl_dir = direction_dict[motion_code]
            else:
                transl_dir = -1 * direction_dict[motion_code]

            bounce_dict = {"transl_dict": make_transl_dict(period_time, period_distance, transl_v_info, 
                transl_dir),
                "sec_transl_dict": None,
                "rot_dict": None}
            
            input_dict_list.append(bounce_dict)
    
    elif motion_code == "debug":
        transl_max_v = 0.02
        period_time = 4
        direction = torch.tensor([1.0, 1.0, -1.5])
        transl_v_info = {"start": 0, "end": transl_max_v, "mode": "smooth"}
        bounce_dict = {"transl_dict": make_transl_dict(period_time, None, transl_v_info, 
            direction),
            "sec_transl_dict": None,
            "rot_dict": None}
        input_dict_list.append(bounce_dict)
    
    elif motion_code == "demo_speed":
        training_speed = 0.02617
        fast_speed = 0.03
        slow_speed = 0.015
        transl_max_v = 0.02
        period_time = 4
        direction = torch.tensor([1.0, 0.0, 0.0])
        transl_v_info = {"start": 0, "end": transl_max_v, "mode": "smooth"}
        bounce_dict = {"transl_dict": make_transl_dict(period_time, None, transl_v_info, 
            direction),
            "sec_transl_dict": None,
            "rot_dict": None}
        input_dict_list.append(bounce_dict)

    elif motion_code == "bouncy":
        #bouncy params
        num_periods = 5
        period_time = 1
        transl_max_v = 0.01
        bounce_max_v = 0.02
        bounce_dir = "updown"
        transl_dir = "fwd"


        for per_num in range(num_periods):

            if per_num == 0:
                transl_v_info = {"start": 0, "end": transl_max_v, "mode": "smooth"}
            elif per_num == num_periods - 1:
                transl_v_info = {"start": transl_max_v, "end": 0, "mode": "smooth"}
            else:
                transl_v_info = {"start": transl_max_v, "end": transl_max_v, "mode": "linear"}

            sec_transl_v_info = {"start": bounce_max_v, "end": -1 * bounce_max_v, "mode": "linear"}
            
            bounce_dict = {"transl_dict": make_transl_dict(period_time, None, transl_v_info, 
                direction_dict[transl_dir]),
                "sec_transl_dict": make_transl_dict(period_time, None, sec_transl_v_info,
                direction_dict[bounce_dir]),
                "rot_dict": None}

            input_dict_list.append(bounce_dict)

    elif motion_code == "rotate_bounce":
        #bouncy params
        num_periods = 4
        period_time = 1
        transl_max_v = 0.02
        bounce_max_v = 0.02
        bounce_dir = "updown"
        transl_dir = "fwd"
        rot_cor = 2
        rot_ccw_cw = "ccw"


        for per_num in range(num_periods):

            if per_num == 0:
                transl_v_info = {"start": 0, "end": transl_max_v, "mode": "smooth"}
            elif per_num == num_periods - 1:
                transl_v_info = {"start": transl_max_v, "end": 0, "mode": "smooth"}
            else:
                transl_v_info = {"start": transl_max_v, "end": transl_max_v, "mode": "linear"}

            sec_transl_v_info = {"start": bounce_max_v, "end": -1 * bounce_max_v, "mode": "linear"}
            
            bounce_dict = {"transl_dict": None,
                "sec_transl_dict": make_transl_dict(period_time, None, sec_transl_v_info,
                direction_dict[bounce_dir]),
                "rot_dict": make_rot_dict(period_time, None, transl_v_info, rot_cor, rot_ccw_cw)}

            input_dict_list.append(bounce_dict)

    
    elif motion_code == "spiral":
        start_cor = 2.0
        end_cor = 0.0
        cor_skip = -0.1
        time_per_cor = 0.5
        spiral_vel = 0.02617
        ccw_cw = "ccw"

        cor_list = np.arange(start_cor, end_cor + cor_skip, cor_skip)

        for cor_ind, cor in enumerate(cor_list):
            if cor_ind == 0:
                rot_v_info = {"start": 0, "end": spiral_vel, "mode": "smooth"}
            elif cor_ind == cor_list.shape[0] - 1:
                rot_v_info = {"start": spiral_vel, "end": 0, "mode": "smooth"}
            else:
                rot_v_info = {"start": spiral_vel, "end": spiral_vel, "mode": "linear"}

            spiral_dict = {"transl_dict": None,
                "sec_transl_dict": None,
                "rot_dict": make_rot_dict(time_per_cor, None, rot_v_info, cor, ccw_cw)}

            input_dict_list.append(spiral_dict)

    elif motion_code == "snake":
        num_twists = 1
        twist_vel = 0.02617
        twist_degree = 270
        init_twist_cor = 1
        init_twist_deg = 135
        init_twist_ccw_cw = "ccw"


        total_twists= num_twists + 2

        for twist_ind in range(total_twists):
            if twist_ind == 0:
                rot_d = init_twist_deg
                rot_v_info = {"start": 0, "end": twist_vel, "mode": "smooth"}
                twist_ccw_cw = init_twist_ccw_cw
                twist_cor = init_twist_cor
            elif twist_ind == total_twists - 1:
                rot_d = init_twist_deg
                rot_v_info = {"start": twist_vel, "end": 0, "mode": "smooth"}
                twist_ccw_cw = get_opp_of_cw_or_ccw(twist_ccw_cw)
                twist_cor = -twist_cor
            else:
                rot_d = twist_degree
                rot_v_info = {"start": twist_vel, "end": twist_vel, "mode": "linear"}
                twist_ccw_cw = get_opp_of_cw_or_ccw(twist_ccw_cw)
                twist_cor = -twist_cor

            twist_dict = {
                "transl_dict": None,
                "sec_transl_dict": None,
                "rot_dict": make_rot_dict(None, rot_d, rot_v_info, twist_cor, twist_ccw_cw)
            }
            input_dict_list.append(twist_dict)

    elif motion_code == 'spiral_upward':
        cor = 1.0
        ccw_cw = 'ccw'
        spiral_vel = 0.02617 #0.02617
        translate_vel = 0.0175#0.02617
        spiral_time = 6

        #initial dict
        time=1
        spiral_dict = {
            "transl_dict": make_transl_dict(time=time, distance=None,
                velocity={"start":0, "end": translate_vel,"mode": "smooth"},
                direction=direction_dict["updown"]),
            "sec_transl_dict": None,
            "rot_dict": make_rot_dict(time=time, distance=None,
                velocity={"start":0, "end": spiral_vel,"mode": "smooth"},
                cor=cor, ccw_cw=ccw_cw)
        }
        input_dict_list.append(spiral_dict)

        #constant speed dict
        spiral_dict = {
            "transl_dict": make_transl_dict(time=spiral_time, distance=None,
                velocity={"start":translate_vel, "end": translate_vel,"mode": "smooth"},
                direction=direction_dict["updown"]),
            "sec_transl_dict": None,
            "rot_dict": make_rot_dict(time=spiral_time, distance=None,
                velocity={"start":spiral_vel, "end": spiral_vel,"mode": "smooth"},
                cor=cor, ccw_cw=ccw_cw)
        }
        input_dict_list.append(spiral_dict)

        #initial dict
        time=1
        spiral_dict = {
            "transl_dict": make_transl_dict(time=time, distance=None,
                velocity={"start":translate_vel, "end":0,"mode": "smooth"},
                direction=direction_dict["updown"]),
            "sec_transl_dict": None,
            "rot_dict": make_rot_dict(time=time, distance=None,
                velocity={"start":translate_vel, "end": 0,"mode": "smooth"},
                cor=cor, ccw_cw=ccw_cw)
        }
        input_dict_list.append(spiral_dict)
    else:
        assert False, "motion_code is not defined"

    input_dict_list.append(rest_dict(3))
    return input_dict_list

def get_opp_of_cw_or_ccw(ccw_cw):
    if ccw_cw == "cw":
        return "ccw"
    elif ccw_cw == "ccw":
        return "cw"

def generate_from_code_for_gen_v3(motion_code, handle_ind_list, 
    initial_mesh, fps):

    if len(handle_ind_list) == 3:
        handle_code_list = [-1,1,0]
    elif len(handle_ind_list) == 2:
        handle_code_list = [-1,1]

    input_dict_list = make_input_dict_from_motion_code(motion_code)

    #----------------------------------------------------------------
    handle_pos_list = []

    #get initial data
    initial_handle_pos = []
    for handle_ind in handle_ind_list:
        initial_handle_pos.append(initial_mesh[handle_ind])
    current_handle_pos_list = initial_handle_pos
    ref_handle_pos = (current_handle_pos_list[handle_code_list.index(-1)] - 
    current_handle_pos_list[handle_code_list.index(1)])
    ref_handle_pos = torch.tensor([ref_handle_pos[0], ref_handle_pos[2]])
    radius = torch.norm(ref_handle_pos).item()/2
    handle_orientation = compute_handle_orientation(ref_handle_pos)

    for input_dict in input_dict_list:
        (adjusted_frame_pos_list, current_handle_pos_list, 
        radius, handle_orientation) = block_transl_rot(handle_orientation, radius, 
            current_handle_pos_list, handle_code_list, fps, input_dict)

        handle_pos_list.append(adjusted_frame_pos_list)
    #--------------------------------------------------------------------


    #----------final processing of handle pos list ---------#
    new_hp_list = []
    for ind in range(len(handle_ind_list)):
        hp = [sublist[ind] for sublist in handle_pos_list]
        handle_pos = torch.cat(hp)

        new_hp_list.append(handle_pos)
    handle_pos_list = new_hp_list

    initial_handle_pos = []
    for handle_ind in handle_ind_list:
        initial_handle_pos.append(initial_mesh[handle_ind])

    num_frames = handle_pos_list[0].shape[0]
    fin_handle_pos = torch.cat([i for i in handle_pos_list], 1).reshape((num_frames,len(handle_pos_list),3))

    full_gt_gp = torch.zeros((fin_handle_pos.shape[0], initial_mesh.shape[0],3)).to(device)
    for ind, i in enumerate(handle_ind_list):
        full_gt_gp[:,i,:] = fin_handle_pos[:,ind,:]
    
    return full_gt_gp

def compute_pos_from_v_traj(v_traj, cur_pos):
    pos_list = []
    #pos_list.append(cur_pos)

    for v_val in v_traj:
        next_pos = v_val + cur_pos
        pos_list.append(next_pos)
        cur_pos = next_pos

    return np.array(pos_list)