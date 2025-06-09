# MeshGraphNetRP: Improving Generalization of GNN-based Cloth Simulation

- Paper: https://dl.acm.org/doi/10.1145/3623264.3624441
- Project: https://lava.kaist.ac.kr/?page_id=7246

## File Locations

### input

- download here: https://kaist.gov-dooray.com/share/drive-files/kkmh2qu2awzo.4fpYw3WzRKqYVxbJtkWNYg
- contains the gt_data
- gt_data folder includes the data gathered from Arcsim, it includes the training dataset as well as test dataset
- unity_demo includes different data for the arbitrary mode (refer to Runtime code). It includes the data of different unseen topologies


### output

- download here: https://kaist.gov-dooray.com/share/drive-files/kkmh2qu2awzo.B0z0QN1PRZmbquL6csUVhw
- contains the basic_cloth, npy_results, and renders
- basic_cloth folder has the saved trained models of (1) final RNN-10 model (2) MLP encoder model and (3) original MeshGraphNets model
- npy_results contains the runtime results saved as npy. These npy files would be the input for the rendering
- renders contains the videos of the output

### params

- these include the json files that describes the configurations of the different models
- it has the json file for (1) final RNN-10 model (2) MLP encoder model and (3) original MeshGraphNets model that could be used for training

## Runtime code

- run the new_runtime_code.py for running the runtime
- it basically needs the name of the model (”json_name”) and the configuration of the runtime (”input_data”)
    - input_data
        - includes the mode, obj_code, and motion_code
        - mode
            - either “arbitrary” or “with_gt”
            - “arbitrary” is for configurations without gt
            - “with_gt” is for configurations with gt
                - refer to input>gt_data folder to see thek different ground truth data available
        - obj_code
            - the code for the cloth topology
            - for “with_gt” mode only “square_1024” as obj_code is available
            - for “arbitrary” refer to obj_name_to_handle_ind_list_dict variable in new_runtime_code.py for the list of different available obj_code
        - motion_code
            - the code for the motion
            - refer to input>gt_data folder to see the different motion available
                - specifically, the meta.json includes the motion code of these gt_data
            - for arbitrary motions, refer to make_input_dict_from_motion_code in generate_json_conf.py for the different available motion_code
                - example motion codes:
                    - directional code: "fwd", "side", "updown", "xy", "yz", "xz", "xyz"
                    - "bouncy"
                    - "rotate_bounce"
                    - "spiral"
                    - "snake"
                    - "spiral_upward"
- it would output an npy file to output>npy_results
- properly edit the "runtime" section of the json files if you want to run a different saved epoch or different model name

## Rendering

- run render_plot.py to render the output>npy_results
- it needs (1) folder name in output>npy_results and (2) viewport:
- viewport could be either:
    - ‘def’ for default view
    - ‘side’
    - ‘front’
- configure the ffmpeg directory to run the code properly

## Training

- run the train.py to train a model given a json file containing the configuration
- it saves the losses using wandb
    - change the wandb.login and wandb.init in train.py with your proper wandb credentials
- once you properly assign the wandb credentials, change the ‘is_save_output_on’ in the params folder json files from false to true
- to train simply run with the following format:
    - python train.py --params final_model
        - params is the name of the json_file
- the models would be saved to output>basic_cloth

## Other

- please refer to environment.yml to see the different libraries used for the python environment

## Acknowledgments
- This repository contains pieces of code from the following repository:
- [PyTorch version of Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)](https://github.com/wwMark/meshgraphnets/tree/main)