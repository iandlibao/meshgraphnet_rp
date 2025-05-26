import os
import sys
import numpy as np
import torch
import tqdm
from matplotlib import animation
import matplotlib.pyplot as plt
from absl import flags, app
from utils import read_json_file
from new_runtime_code import get_path_from_gt_input

import platform
if platform.machine() == 'AMD64':
    if platform.node() == 'DESKTOP-REIB520':
        plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\ian\\Downloads\\ffmpeg-5.0.1-essentials_build\\bin\\ffmpeg.exe'
    else:
        plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\User\\Downloads\\ffmpeg-5.0.1-essentials_build\\bin\\ffmpeg.exe'
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    plt.rcParams['animation.ffmpeg_path'] = r'/opt/homebrew/bin/ffmpeg'
    device = torch.device('cpu')
elif platform.machine() == 'x86_64':
    #plt.rcParams['animation.ffmpeg_path'] = r'/home/ps3/Documents/ffmpeg-6.0/ffmpeg'
    device = torch.device('cuda')

    

#-----------------------------------#
# FLAGS = flags.FLAGS
# flags.DEFINE_string('render_params', 'final_model', 'the json file')
# flags.DEFINE_string('view', 'def', 'viewport')
#flags.DEFINE_integer('graph_mode', 0, '0: [[pred, gt]], 1:[[pred, gt], [gt]], 2:[[pred],[gt]], 3:[[pred]]')
#-----------------------------------#





def render_single(position_list, face_info, viewport, result_path,fps):
    plot_size = len(position_list) #plus 1 for the gt
    fig, axs = plt.subplots(1,plot_size, subplot_kw={'projection': '3d'})
    if len(position_list) == 1:
        axs = [axs]
    fig.set_size_inches(19.2,10.8)

    azim = viewport[0]
    elev = viewport[1]

    min_length = 99999999999
    for single_plot_list in position_list:
        for position in single_plot_list:
            if position.shape[0] < min_length:
                num_steps = position.shape[0]
                
    # num_steps = 120




    #compute bounds
    all_bounds_min = []
    all_bounds_max = []
    for single_plot_list in position_list:
        for plot in single_plot_list:
            plot = plot[:num_steps]
            bb_min = np.squeeze(plot).min(axis=(0, 1))
            bb_max = np.squeeze(plot).max(axis=(0, 1))
            all_bounds_min.append(bb_min)
            all_bounds_max.append(bb_max)
    final_bound_min = np.stack(all_bounds_min).min(axis=0)
    final_bound_max = np.stack(all_bounds_max).max(axis=0)
    #get the max range
    ran_val = (final_bound_max - final_bound_min).max()
    #get the mean
    mean_val = (final_bound_max + final_bound_min)/2
    bound = (mean_val - ran_val/2, mean_val + ran_val/2)

    #bound = (final_bound_min, final_bound_max)

    
    
    def animate(num):
        #print(num)
        for plot_group_index, plot_group in enumerate(position_list):
            axs[plot_group_index].cla()
            axs[plot_group_index].set_xlim([bound[0][0], bound[1][0]])
            axs[plot_group_index].set_ylim([bound[0][1], bound[1][1]])
            axs[plot_group_index].set_zlim([bound[0][2], bound[1][2]])

            axs[plot_group_index].azim = azim
            axs[plot_group_index].elev = elev
            for plot_index, position in enumerate(plot_group):
                pos = position[num]

                if plot_index == 0:
                    alpha = 1
                else:
                    alpha = 0.3
                # if ind < 7:
                #     alpha = 1
                #     color = 'blue'
                # else:
                #     alpha = 0.5
                #     color = 'red'

                axs[plot_group_index].plot_trisurf(pos[:, 0], pos[:, 1], face_info, pos[:, 2], shade=True, alpha=alpha)
        
        fig.suptitle("azim %d | elev %d | frame %d" %(azim, elev, num))
        
        return fig,
    
    anima = animation.FuncAnimation(fig, animate, frames=num_steps)
    pbar = tqdm.tqdm(total=num_steps)
    writervideo = animation.FFMpegWriter(fps=fps)
    anima.save(result_path, writer=writervideo,
                progress_callback=lambda i, n: pbar.update(1))

def render_json(json_name, viewport):
    json_path = os.path.join("output", "npy_results", json_name)
    folder_list = [folder for folder in os.listdir(json_path) if
        os.path.isdir(os.path.join(json_path, folder))]
    
    for folder in folder_list:
        render_json_folder(json_name, folder, viewport)

def render_json_folder(json_name, render_folder, viewport):
    json_params = read_json_file(json_name, 'test')

    npy_folder = os.path.join('output', 'npy_results', json_name, render_folder)
    
    npy_files = [npy_file for npy_file in os.listdir(npy_folder) if \
        npy_file[-1 * len(".npy"):] == ".npy"]

    for npy_file in npy_files:
        npy_path = os.path.join(npy_folder, npy_file)

        render_json_folder_path = os.path.join('output', 'renders', json_name)
        if not os.path.exists(render_json_folder_path):
            os.makedirs(render_json_folder_path)
        render_folder_path = os.path.join(render_json_folder_path, render_folder)
        if not os.path.exists(render_folder_path):
            os.makedirs(render_folder_path)
        render_filename = npy_file[:-1 * len(".npy")] + "_" + viewport
        
        result_path = os.path.join(render_folder_path, render_filename + ".mp4")

        
        if not os.path.exists(result_path):
            render_npy_file(viewport, npy_path, result_path, json_params['fps'])
        elif os.path.getsize(result_path) < 5:
            render_npy_file(viewport, npy_path, result_path, json_params['fps'])
        else:
            print("mp4 render exists")

def render_npy_file(viewport, npy_path, result_path, fps):
    viewport_dict = {'def': (-60,30), 'side': (-90,90), 'front': (0,0)}

    npy_data = np.load(npy_path, allow_pickle=True).tolist()
    
    data_keys = ['position', 'face', 'input_data']
    for data_key in data_keys:
        if not data_key in npy_data.keys():
            print("npy path ", npy_path, " is invalid")
            return -1

    output_pos = npy_data['position']
    face_data = npy_data['face']
    mode = npy_data['input_data']['mode']
    input_data = npy_data['input_data']
    

    if mode == 'arbitrary':
        position_list = [[output_pos]]
    elif mode == 'with_gt':
        gt_pos = get_gt_cloth_pos(input_data)
        position_list = [[output_pos, gt_pos]]

    render_dir = result_path
    
    render_single(position_list, face_data, viewport_dict[viewport], 
        render_dir, fps)

def get_gt_cloth_pos(gt_input):
    gt_path = get_path_from_gt_input(gt_input)
    return np.load(os.path.join(gt_path, "cloth_pos.npy"))

# def main(argv):

#     json_filename = FLAGS.render_params
#     viewport = FLAGS.view

#     render_json(json_filename, viewport)

    

if __name__ == '__main__':
    json_name_list = [
        "mesh_graph_nets", 
        "mlp_encoder",
        "final_model"    
    ]
    viewport_list = [
        'def',
        # 'side',
        # 'front'
    ]
    for json_name in json_name_list:
        for viewport in viewport_list:
            render_json(json_name, viewport)
    # app.run(main)


    