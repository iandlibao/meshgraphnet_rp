import os
from pathlib import Path
import pickle

import logging
import time
import datetime





def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def find_nth_latest_run_step(run_dir, n):
    all_run_step_dirs = os.listdir(run_dir)
    all_run_step_dirs = map(lambda d: os.path.join(run_dir, d), all_run_step_dirs)
    all_run_step_dirs = [d for d in all_run_step_dirs if os.path.isdir(d)]
    print(all_run_step_dirs)
    nth_latest_run_step_dir = sorted(all_run_step_dirs, key=os.path.getmtime)[-n]
    return nth_latest_run_step_dir


def prepare_files_and_directories(last_run_dir, output_dir, is_save_output_on, name_code):
    '''
        The following code is about creating all the necessary files and directories for the run
    '''
    # if last run dir is not specified, then new run dir should be created, otherwise use run specified by argument
    if last_run_dir is not None:
        run_dir = last_run_dir
        run_create_datetime_datetime_dash = run_dir
    else:
        run_create_time = time.time()
        run_create_datetime = datetime.datetime.fromtimestamp(run_create_time).strftime('%c')
        if(is_save_output_on == True):
            run_create_datetime_datetime_dash = run_create_datetime.replace(" ", "-").replace(":", "-")
            run_create_datetime_datetime_dash = name_code + "_" + run_create_datetime_datetime_dash
        else:
            run_create_datetime_datetime_dash = "temp"
        run_dir = os.path.join(output_dir, run_create_datetime_datetime_dash)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    # check for last run step dir and if exists, create a new run step dir with incrementing dir name, otherwise create the first run step dir
    all_run_step_dirs = os.listdir(run_dir)
    if not all_run_step_dirs:
        run_step_dir = os.path.join(run_dir, '1')
    else:
        latest_run_step_dir = find_nth_latest_run_step(run_dir, 1)
        run_step_dir = str(int(Path(latest_run_step_dir).name) + 1)
        run_step_dir = os.path.join(run_dir, run_step_dir)

    # make all the necessary directories
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


    return run_step_dir


def logger_setup(log_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    root_logger.addHandler(file_log_handler)
    return root_logger


def log_run_summary(root_logger, run_step_config):
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Name code is " + run_step_config['name_code'])
    root_logger.info("=========================================================")
    root_logger.info("")

