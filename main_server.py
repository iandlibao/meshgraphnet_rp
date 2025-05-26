import zmq
import time
import torch
from new_runtime_code import initial_setup, step_fn
import numpy as np

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

port_number = "5555"



def make_json_message(mesh_pos, face_info, command_type, stop_signal):
    output = {
        "mesh_pos": mesh_pos,
        "command_type": command_type,
        "stop_signal": stop_signal    
    }
    if command_type == 'initial':
        output['face_info'] = face_info

    return output

def slerp_torch(vector_a, vector_b, scale_factor):
    dot_product = torch.dot(vector_a, vector_b)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Ensure dot product is within [-1, 1] range
    theta_0 = torch.acos(dot_product)  # Angle between the vectors
    sin_theta_0 = torch.sin(theta_0)
    # Slerp formula
    interpolated_tensor = (torch.sin((1 - scale_factor) * theta_0) / sin_theta_0) * vector_a + (torch.sin(scale_factor * theta_0) / sin_theta_0) * vector_b
    return interpolated_tensor

def get_transl_displacement_wrapper(direction, transl_mag, transl_dir, max_tv_factor, accel_time,
    decel_time, dir_change_time, option):

    if option == 'smooth':
        return get_transl_displacement(direction, transl_mag, transl_dir, max_tv_factor,
            accel_time, decel_time, dir_change_time)
    elif option == 'fast':
        return get_transl_displacement_old(direction, transl_mag, transl_dir, max_tv_factor,
            accel_time, decel_time)

def get_transl_displacement_old(direction, transl_mag, transl_dir, max_tv_factor, 
    accel_time, decel_time):

    fps = 60
    max_vel = 0.025
    max_vel *= max_tv_factor
    


    
    if direction == [0.,0.,0.]:
        #decelerate
        acceleration = -max_vel/(decel_time*fps)
        
        if transl_mag > 0:
            new_vel = max(transl_mag + acceleration, 0)
        else:
            new_vel = 0
        fin_new_vel = new_vel * transl_dir
        new_dir = transl_dir

    else:
        #accelerate
        direction_tensor = torch.tensor(direction).to(device)
        direction_tensor = direction_tensor/torch.norm(direction_tensor)
        acceleration = max_vel/(accel_time*fps)
        
        if transl_mag < max_vel:
            new_vel = min(transl_mag + acceleration, max_vel)
        else:
            new_vel = max_vel
        fin_new_vel = new_vel * direction_tensor
        new_dir = direction_tensor


    return fin_new_vel, new_vel, new_dir
def get_transl_displacement(direction, transl_mag, transl_dir, max_tv_factor, accel_time, decel_time,
    dir_change_time):
    
    fps = 60
    max_vel = 0.025
    max_vel *= max_tv_factor

    cur_vel_tensor = transl_mag * transl_dir

    
    if direction != [0.,0.,0.]:
        #acceleration
        direction_tensor = torch.tensor(direction).to(device)
        direction_tensor = direction_tensor/torch.norm(direction_tensor)
        
        acceleration = max_vel/(accel_time*fps)
        accel_tensor = acceleration * direction_tensor
        

        new_vel_tensor = accel_tensor + cur_vel_tensor
        if torch.norm(new_vel_tensor) > max_vel:
            if not torch.allclose(transl_dir, direction_tensor, atol=1e-4):

                dot_product = torch.clamp(torch.dot(direction_tensor, transl_dir), -1.0, 1.0)  # Ensure dot product is within [-1, 1] range
                theta = torch.acos(dot_product)
                angle_change = np.pi/(2 * dir_change_time * fps)
                scale_factor = min(angle_change/theta, 1.)

                slerp_dir = slerp_torch(transl_dir, direction_tensor, scale_factor)

            else:
                slerp_dir = transl_dir
            new_vel_tensor = max_vel * slerp_dir
        
        if torch.norm(new_vel_tensor) == 0:
            new_direction = torch.tensor([0.,0.,0.]).to(device)
        else:
            new_direction = new_vel_tensor/torch.norm(new_vel_tensor)

        new_vel = torch.norm(new_vel_tensor)

    else:
        #deceleration
        acceleration = -max_vel/(decel_time*fps)

        new_vel = max(transl_mag + acceleration, 0)

        new_vel_tensor = new_vel * transl_dir
        new_direction = transl_dir


    return new_vel_tensor, new_vel, new_direction

def compute_handle_orientation(handle_positions):
    ref_handle_pos = (handle_positions[0] - handle_positions[1])
    ref_handle_pos = torch.tensor([ref_handle_pos[0], ref_handle_pos[2]]).to(device)
    handle_orientation = torch.atan2(ref_handle_pos[1], ref_handle_pos[0])


    return handle_orientation

def compute_theta_start(handle_orientation, center_of_rotation, handle_code):
    x = handle_code - center_of_rotation

    if x <= 0:
        theta_start = handle_orientation
    else:
        theta_start = handle_orientation + np.pi

    return theta_start

def solve_max_radius(cor, radius):
    he_radius_list = []
    handle_ends_code_list = [-1,1]
    for handle_end in handle_ends_code_list:
        he_relative_radius = abs(cor - handle_end)
        he_radius = radius * he_relative_radius
        he_radius_list.append(he_radius)
    max_radius = max(he_radius_list)
    return max_radius

def solve_max_radius_new(cor_point, handle_positions):
    dist_list = []
    for h_pos in handle_positions:
        dist_list.append(torch.norm(h_pos - cor_point))
    return max(dist_list)

def get_rot_displacement_new(handle_positions, max_av_factor, accel_time, decel_time, 
    cor_point, input_axis, cur_ang_vel_tensor):

    max_linear_vel = 0.025 * max_av_factor
    max_vel = max_linear_vel/solve_max_radius_new(cor_point, handle_positions)
    fps = 60

    
    if not torch.equal(input_axis, torch.zeros((3)).to(device)):
        #accelerate
        acceleration = max_vel/(accel_time * fps)
        accel_tensor = acceleration * input_axis

        
        new_ang_vel_tensor = cur_ang_vel_tensor + accel_tensor
        new_ang_vel_mag = torch.norm(new_ang_vel_tensor)
        if new_ang_vel_mag > max_vel:
            new_ang_vel_tensor = (max_vel/new_ang_vel_mag) * new_ang_vel_tensor

    else:
        #decelerate
        acceleration = max_vel/(decel_time * fps)
        cur_ang_vel_mag = torch.norm(cur_ang_vel_tensor)
        new_ang_vel = max(cur_ang_vel_mag - acceleration, 0)
        if cur_ang_vel_mag == 0:
            new_ang_vel_tensor = cur_ang_vel_tensor
        else:
            new_ang_vel_tensor = (new_ang_vel/torch.norm(cur_ang_vel_tensor)) * cur_ang_vel_tensor 

    

    new_ang_vel_mag = torch.norm(new_ang_vel_tensor)
    if new_ang_vel_mag == 0:
        new_axis = torch.tensor([0.,1.,0.]).to(device)
    else:
        new_axis = new_ang_vel_tensor/new_ang_vel_mag

    rot_disp_list = rotate_points_around_axis(handle_positions, cor_point, new_axis, new_ang_vel_mag)

    return rot_disp_list, new_ang_vel_tensor

def rotate_points_around_axis(points, cor_point, rot_axis, rot_angle):
    rot_axis_norm = rot_axis/torch.norm(rot_axis)

    rot_points_displacement_list = []
    for point in points:
        offset = point - cor_point

        rotated_offset = offset * torch.cos(rot_angle) + \
                torch.cross(rot_axis_norm, offset) * torch.sin(rot_angle) + \
                rot_axis_norm * torch.dot(rot_axis_norm, offset) * (1 - torch.cos(rot_angle))
        
        rotated_point = cor_point + rotated_offset
        rot_point_displacement = rotated_point - point
        rot_points_displacement_list.append(rot_point_displacement)
    
    return rot_points_displacement_list

def get_rot_displacement(cor, radius, pressed_button, handle_ang_vel, handle_positions, 
    max_av_factor, accel_time, decel_time):

    max_linear_vel = 0.025 * max_av_factor
    max_vel = max_linear_vel/solve_max_radius(cor, radius)
    fps = 60

    if pressed_button is not None:
        #accelerate
        acceleration = max_vel/(accel_time * fps)
        if pressed_button == 'ccw':
            if handle_ang_vel < max_vel:
                new_ang_vel = min(handle_ang_vel + acceleration, max_vel)
            else:
                new_ang_vel = max_vel
        elif pressed_button == 'cw':
            if handle_ang_vel > -max_vel:
                new_ang_vel = max(handle_ang_vel - acceleration, -max_vel)
            else:
                new_ang_vel = -max_vel
    else:
        #decelerate
        acceleration = -max_vel/(decel_time * fps)
        if handle_ang_vel > 0:
            new_ang_vel = max(handle_ang_vel + acceleration, 0)
        elif handle_ang_vel < 0:
            new_ang_vel = min(handle_ang_vel - acceleration, 0)
        else:
            new_ang_vel = 0


    handle_orientation = compute_handle_orientation(handle_positions)
    handle_code_list = [-1,1]
    theta_start_list = []
    handle_radius_list = []
    for handle_code in handle_code_list:
        theta_start = compute_theta_start(handle_orientation, cor, handle_code)
        theta_start_list.append(theta_start)

        relative_radius = abs(cor - handle_code)
        handle_radius = radius * relative_radius
        handle_radius_list.append(handle_radius)

    rot_disp_list = []
    for theta_start, handle_radius in zip(theta_start_list, handle_radius_list):
        rot_disp_list.append(torch.tensor([handle_radius * (torch.cos(theta_start + new_ang_vel)-torch.cos(theta_start)),
            0., handle_radius * (torch.sin(theta_start + new_ang_vel)-torch.sin(theta_start))]).to(device))
    
    return rot_disp_list, new_ang_vel

def get_handle_pos_from_disp(handle_positions, rot_disp_list, transl_disp,
    num_vertices, handle_ind_list):
    full_positions = torch.zeros((num_vertices,3)).to(device)
    for handle_ind, handle_p, rot_disp in zip(handle_ind_list, handle_positions, rot_disp_list):
        total_disp = rot_disp + transl_disp
        # print('handle ind: ', handle_ind, ' disp_mag: ', torch.norm(total_disp))
        full_positions[handle_ind] = handle_p + total_disp

    return full_positions

def compute_initial_cor_point(cor, handle_positions, radius):
    origin = (handle_positions[0] + handle_positions[1])/2
    cor_offset = radius * cor
    return origin + cor_offset


if __name__ == "__main__":

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:' + port_number)

    while True:
        start = time.time()

        json_message = socket.recv_json()

        command_type = json_message['command_type']
        

        if command_type == 'initial':
            traj_ind = 0

            obj_name = json_message['obj_name']
            motion_traj = json_message['motion_traj']
            cor = json_message['cor']
            max_av_factor = json_message['max_av_factor']
            max_tv_factor = json_message['max_tv_factor']
            t_accel_time = json_message['t_accel_time']
            t_decel_time = json_message['t_decel_time']
            t_dir_change_time = json_message['t_dir_change_time']
            t_dir_change_option = json_message['t_dir_change_option']
            r_accel_time = json_message['r_accel_time']
            r_decel_time = json_message['r_decel_time']
            json_name = json_message['model_name']

            input_data = {
                "mode": "arbitrary",
                "obj_code": obj_name,
                "motion_code": motion_traj
            }

            model, cloth_state, handle_traj, json_params = initial_setup(json_name, input_data)
            mesh_pos = cloth_state.get_unity_position()
            face_info = cloth_state.get_unity_face()
            stop_signal = False

            cor_3d = json_message['cor_3d']
            cor_point = compute_initial_cor_point(torch.tensor(cor_3d).to(device),
                cloth_state.get_position_list(cloth_state.handle_ind_list), 
                cloth_state.radius)
            
            
        elif command_type == 'normal':

            user_direction = json_message['user_direction']
            axis = torch.tensor(json_message['rot_axis']).to(device)
            
            if motion_traj == 'userFeed':
                transl_disp, handle_transl_mag, handle_transl_dir = get_transl_displacement_wrapper(user_direction,
                    cloth_state.handle_transl_mag, cloth_state.handle_transl_dir, max_tv_factor,
                    t_accel_time, t_decel_time, t_dir_change_time, t_dir_change_option)
                cloth_state.handle_transl_mag = handle_transl_mag
                cloth_state.handle_transl_dir = handle_transl_dir
                
                # pressed_button = json_message['pressed_button']
                # rot_disp_list, handle_ang_vel = get_rot_displacement(cor, cloth_state.radius, 
                #     pressed_button, cloth_state.handle_ang_vel, 
                #     cloth_state.get_position_list(cloth_state.handle_ind_list), max_av_factor,
                #     r_accel_time, r_decel_time)
                # cloth_state.handle_ang_vel = handle_ang_vel

                rot_disp_list, handle_ang_vel_tensor = get_rot_displacement_new(
                    cloth_state.get_position_list(cloth_state.handle_ind_list), 
                    max_av_factor, r_accel_time, r_decel_time, cor_point, axis, 
                    cloth_state.handle_ang_vel_tensor)
                cloth_state.handle_ang_vel_tensor = handle_ang_vel_tensor

                
                

                #update cor_point from the transl_disp
                cor_point += transl_disp

                handle_pos = get_handle_pos_from_disp(cloth_state.get_position_list(
                    cloth_state.handle_ind_list), rot_disp_list,
                    transl_disp, cloth_state.num_vertices, cloth_state.handle_ind_list)
            else:
                handle_pos = handle_traj[traj_ind]
            step_fn(model, cloth_state, json_params, handle_pos)
            mesh_pos = cloth_state.get_unity_position()
            face_info = None
            traj_ind = traj_ind + 1

            if motion_traj != 'userFeed':
                if traj_ind == handle_traj.shape[0]:
                    stop_signal = True
                else:
                    stop_signal = False
            else:
                stop_signal = False
            

        reply_data = make_json_message(mesh_pos, face_info, command_type, stop_signal)


        socket.send_json(reply_data)

        end = time.time()
        # print('FPS: ', 1/(end-start))