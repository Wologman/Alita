''' Aquires settings files and determines:
The locations to be used for the independent dataset
The locations to be used for the model training
Runs MegaDetector over everything to produce a json file
Saves the json file to the destination
Checks if RELOAD_INDEPENDENT_SET = 'True'
    If True, Deletes any existing folder
    Loads each of the folders into the local directory
Next steps is to run Interpret_JSON, and Preprocess_Images, to load up the crops
'''

import os, sys
from pathlib import Path
import time
import yaml
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm


class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_01'
        self.RELOAD_INDEPENDENT = False
        self.RERUN_MD_ON_ALL = False
        self.RUN_MD_NEW_FILES_ONLY = False
        self.RELOAD_NEW_TRAIN_IMAGES = False #This is for debugging, would only be false if the new images are already loaded to temp
        self.TEMP_DATA_SSD = 'E:\\'
        self.SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
        self.INDEPENDENT_TEST_ONLY = ['N01', 'BWS', 'EBF', 'EM1', 'ES1']
        
        # Atrubutes that shouldn't need changing below
        self.DATA_FOLDER_NM = 'Data'
        self.INPUT_FOLDER_NM = 'Inputs'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.SETUP_FOLDER_NM = 'Setup'
        self.IND_IMGS_FOLDER_NM = 'Independent_Images'
        self.RUN_MD_ON_TRAINING_NM = 'MD_Train_Crops.ps1'
        self.TRAIN_DATA_JSON_NM = 'MD_output.json'
        self.LAST_MD_FOLDER_NM = 'MD_Last_Run'
        self.TEMP_DATA_STORE = 'temp_images'

def get_config(settings_pth = None):
    #Config attributes with values that need evaluating from strings, e.g. '1e-5', singletons, instances:
    evaluate_list = ['RELOAD_INDEPENDENT', 'RELOAD_NEW_TRAIN_IMAGES', 'RERUN_MD_ON_ALL',
                     'RUN_MD_NEW_FILES_ONLY', 'INDEPENDENT_TEST_ONLY']
    cfg = DefaultConfig()
    
    if settings_pth:
        with open(settings_pth, 'r') as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        for key, value in yaml_data.items():
            if hasattr(cfg, key):
                if (key in evaluate_list) and (isinstance(value, str)):
                    setattr(cfg, key, eval(value))
                else:
                    setattr(cfg, key, value)
    return cfg


def get_folder_paths(root_dir, fldr_nms):
    matching_paths = []
    for item in root_dir.iterdir():
        if item.is_dir() and item.name in fldr_nms:
            matching_paths.append(item)
    return matching_paths


def run_mega_detector(process_folder, json_pth, run_md_pth):
    # Start the subprocess and redirect its output
    print(f'Running the MegaDetector on {process_folder}')
    cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(run_md_pth), str(process_folder), str(json_pth)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line_count = 0
    for line in process.stdout:
        if line_count < 10:
            print(line.strip())  # Print the first 10 lines normally
        else:
            sys.stdout.write("\r" + line.strip())  # Overwrite the current line
            sys.stdout.flush()
        line_count += 1
    for line in process.stdout:
        sys.stdout.write("\r" + line.strip())  # Overwrite the current line
        sys.stdout.flush()  
    return


def copy_old_json_files(archive_fldr, dstn_fldr):
    json_files = list(archive_fldr.glob("*.json"))
    if json_files:
        for json_file in json_files:
            destination_path = dstn_fldr / json_file.name
            shutil.copy(json_file, destination_path)
            print(f'Existing .jsons file copied from {archive_fldr} to {destination_path}')
    else:
        print(f'No .json files found in {archive_fldr}')
    return


def copy_last_parquet(archive_fldr, dstn_fldr):
    parquet_files = list(archive_fldr.glob("*.parquet"))
    if parquet_files:
        for parquet_file in parquet_files:
            destination_path = dstn_fldr / parquet_file.name
            shutil.copy(parquet_file, destination_path)
            print(f'Existing .parquets file copied from {archive_fldr} to {destination_path}')
    else:
        print(f'No .parquet files found in {archive_fldr}')
    return


def copy_independent_set(fldr_list, dstn):
    if dstn.exists() and dstn.is_dir():
        shutil.rmtree(dstn)
    os.mkdir(dstn)
    print(f'The list of folders to move for an independent set: {fldr_list}')
    start_time = time.time()
    print(f'Copying the independent image datasets to {dstn}')
    for folder in tqdm(fldr_list):
        destination = dstn / folder.name
        shutil.copytree(folder, destination)
    num_copied = sum(1 for _ in dstn.rglob('*') if _.is_file())
    end_time = time.time()
    print(f'Transfer of independent images completed in {end_time-start_time:.2f} seconds')
    print(f'{num_copied}, files were copied from the source drive to: {dstn} for a holdout test set from unique locations')
    return


def get_already_detected(src_fldr):
    """Searches the MD_Last_Run directory for all the parquet files
       Finds the most recent parquet file
       If one is found, returns a list of all the file-paths in it, otherwise returns None
       It doesn't matter if there are two files there all_labels and last_exif_data, so long as they have matching rows"""
    print('Sourcing parqet file of images previously processed by MegaDetector')
    parquet_files = list(src_fldr.glob("*.parquet"))
    if parquet_files:
        parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent_parquet = parquet_files[0]
        print(f'Most recent .parquet was {most_recent_parquet}')
        df = pd.read_parquet(most_recent_parquet)
        processed_paths_strs = df['File_Path'].to_list()
        processed_paths = [Path(path_string) for path_string in processed_paths_strs]
    else:
        print(f'No .parquet files found in {src_fldr}')
        processed_paths = None
    return processed_paths


def get_unprocessed(root_fldr, exclude_fldrs, finished_pths):
    search_folders = [fold for fold in root_fldr.iterdir() if fold.is_dir() and fold not in exclude_fldrs]
    all_paths = [file for folder in search_folders for file in tqdm(folder.rglob('*.[jJ][pP][gG]'), desc=f"Searching in {folder}")]
    print(f'There are a total of {len(all_paths)} image files for training')
    new_paths = list(set(all_paths) - set(finished_pths))
    print(f'There are {len(new_paths)} new images to be moved then processed by MegaDetector')
    return new_paths


def find_and_move_new_images(temp, last_md_folder, source_folder, independent_folders):
    """
    1. Removes any existing temp folder storing images from last time this process was run
    2. 
    """
    if temp.is_dir():
        shutil.rmtree(temp)
        os.mkdir(temp)

    processed_paths = get_already_detected(last_md_folder)
    #Now find which files exist but don't have entries in the dataframe
    unprocessed_paths = get_unprocessed(source_folder, independent_folders, processed_paths)
    
    for source_fpath in tqdm(unprocessed_paths):
        relative_path = source_fpath.relative_to(source_folder)
        destination_path = temp / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_fpath, destination_path)
    return

# ----------------------------------- Main Script-----------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_pth = None):
    cfg = get_config(settings_pth)
    script_dir = Path(__file__).resolve().parent
    project_folder = script_dir.parent
    source_folder = Path(cfg.SOURCE_IMAGES_PTH)
    data_folder = project_folder / cfg.DATA_FOLDER_NM
    ind_images_folder = data_folder / cfg.IND_IMGS_FOLDER_NM
    exp_folder = data_folder / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME
    last_md_folder = data_folder / cfg.EXPS_FOLDER_NM / cfg.LAST_MD_FOLDER_NM
    inputs_folder = exp_folder / cfg.INPUT_FOLDER_NM 
    json_pth = inputs_folder / cfg.TRAIN_DATA_JSON_NM
    run_md_pth = script_dir /  cfg.SETUP_FOLDER_NM / cfg.RUN_MD_ON_TRAINING_NM
    independent_folders = get_folder_paths(source_folder, cfg.INDEPENDENT_TEST_ONLY)
    temp_fldr = Path(cfg.TEMP_DATA_SSD) / cfg.TEMP_DATA_STORE

    if not os.path.exists(inputs_folder):
        os.makedirs(inputs_folder) 
    if cfg.RELOAD_INDEPENDENT:
        copy_independent_set(independent_folders, ind_images_folder)   
    if cfg.RERUN_MD_ON_ALL:
        run_mega_detector(source_folder, json_pth, run_md_pth)
    elif cfg.RUN_MD_NEW_FILES_ONLY:
        if cfg.RELOAD_NEW_TRAIN_IMAGES:
            #Moves all images not found in the most recent .parquet file in MD_Last_Run folder
            find_and_move_new_images(temp_fldr, last_md_folder, source_folder, independent_folders)
        run_mega_detector(temp_fldr, json_pth, run_md_pth)
        copy_old_json_files(last_md_folder, inputs_folder)
        copy_last_parquet(last_md_folder, inputs_folder)
    else:
        copy_old_json_files(last_md_folder, inputs_folder)
        copy_last_parquet(last_md_folder, inputs_folder)
    return

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()