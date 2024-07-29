'''This is the master file that:
- Makes a list of existing run_IDs and experiment names.
- Otherwise it checks for a single file sitting in the settings
- If no settings file found at all, a prompt will ask if everyhing should run with script defaults
- The individual scripts should also run with defults if no settings file provided if required
- Otherwise makes a list of all settings files in the directory, and processes oldest first
- Checks for eperiment update, vs just run_ID update
- Any directories needed should be set up by the individual scripts being run
- Runs all the preprocessing, either just a retrain with new settings, or a new dataset (New exp ID)
- Runs training script
- Runs evaluation notebooks
- Logs the results to a csv for later analysis & saves the settings file in the results folder for that run.
- If all the settings files on the list have been processed, looks for new ones added since the start
- If no new settings files found, processing stops
'''
#General libraries
import yaml
import os
import sys
import json
import torch
import shutil
from pathlib import Path

#Load up the other scripts
import Clean_Data
import Preprocess_Images
import Training
import Reload_Images
import Interpret_JSON

#For running the notebooks
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter


class Config:
    def __init__(self):
        self.DATA_FOLDER_NM = 'Data'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.RUNS_FOLDER_NM = 'Runs'
        self.INPUT_FOLDER_NM = 'Inputs'
        self.SETTINGS_FOLDER_NM =  'Settings'
        self.RESULTS_FOLDER_NM = 'Results'
        self.DATA_FOLDER_NM = 'Data' 
        self.SETUP_DIR_NM = 'Setup'
        self.DEBUG_SETTINGS_FLDR_NM = 'Debug'
        self.DESCRIPTION ='Running Evaluation Only'
        self.DEBUG_SETTINGS = 'Debug_Settings.yaml'
        self.DATA_EXP_NB = 'Data_Exploration.ipynb'
        self.DATA_EXP_NB_HTML = 'Data_Exploration.html'
        self.EVAL_NB = 'Model_Evaluation.ipynb'
        self.EVAL_NB_HTML = 'Model_Eval.html'
        self.FINISHED_SETTINGS = 'Finished_Settings_Files'
        self.RUN_PREPROCESS = True # Only effects new experiments, not new runs or EVAL_ONLY 
        self.EVAL_ONLY = None 
        #r'E:\Project\Settings\Exp_26_Run_02.yaml' #capacity_vals[np.isnan(capacity_vals)] = 1000   #intersting that this is needed! # Should skip all training, just run the eval notebook on this file
        #r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Predator_AI_Project\Settings\Exp_36_Run_17.yaml'

def check_for_settings(cfg):
    '''Takes in the configuration instance, looks in project settings directory, 
    and returns a list of all the .yaml file path objects, with the oldest first'''
    project_dir = Path(__file__).resolve().parent.parent
    settings_dir = project_dir / cfg.SETTINGS_FOLDER_NM
    settings_paths = list(settings_dir.glob("*.yaml"))
    file_info_tuples = [(path, os.path.getmtime(path)) for path in settings_paths]
    sorted_file_info = sorted(file_info_tuples, key=lambda x: x[1])
    ordered_settings_paths = [file_info[0] for file_info in sorted_file_info]
    return ordered_settings_paths


def list_past_runs(cfg):
    '''Take the configuration instance and looks through the specified directories
    for past project experiments (each dataset change) or chang of run_ID 
    (hyperparameter change without re-creating a dataset)'''
    project_dir = Path(__file__).resolve().parent.parent
    exps_dir = project_dir / cfg.DATA_FOLDER_NM / cfg.EXPS_FOLDER_NM

    exp_run_list = []
    experiments =  [folder for folder in exps_dir.iterdir() if folder.is_dir() and folder.name not in ['Debug', 'MD_Last_Run']]
    for experiment in experiments:
        runs_folder = experiment / cfg.RUNS_FOLDER_NM
        exp_nm = str(experiment.name)
        if runs_folder.is_dir():
            runs = [(exp_nm, str(run.name)) for run in runs_folder.iterdir() if run.is_dir()]
        else:
            runs=None
        if not runs:
            runs=[(exp_nm,'No_Runs_Completed')]
        exp_run_list.extend(runs)
    return exp_run_list


def extract_run_exp(settings_pth):
    with open(settings_pth, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        exp = yaml_data['EXPERIMENT_NAME']
        run = yaml_data['RUN_ID']
    return exp, run


def check_for_new_run_exp(settings_pths, exp_run_list):
    '''Compares the RUN_ID and EXPERIMENT_NAME from past runs and decides its a new
    experiment or new run only.  Returns:
    new_exps {settings_pth:run_name} For settings files that need new exeriments
    new_runs {settings_pth:run_name} For settings files that only need a new run'''
    exp_list = [item[0] for item in exp_run_list]
    new_exps = {}
    new_runs = {}
    for settings in settings_pths:
        exp_nm, run_id = extract_run_exp(settings)
        if exp_nm not in exp_list:
            new_exps[str(settings)] = (exp_nm, run_id)
        elif (exp_nm, run_id) not in exp_run_list:
            new_runs[str(settings)] = (exp_nm, run_id)
    return new_exps, new_runs


def check_for_unprocessed(cfg):
    settings_list = check_for_settings(cfg)
    if settings_list:
        print(Colour.S + 'Settings files found: '+ Colour.E, [f.name for f in settings_list])
    all_past_runs = list_past_runs(cfg)
    todo_exps, todo_runs = check_for_new_run_exp(settings_list, all_past_runs)
    return todo_exps, todo_runs, settings_list


def get_nb_paths(cfg, inputs_folder, results_dest):
    '''Sets path objects to point towards notebook locations, and save locations'''
    script_dir = Path(__file__).resolve().parent
    data_explore = script_dir / cfg.DATA_EXP_NB
    data_exp_html =  inputs_folder /  cfg.DATA_EXP_NB_HTML
    model_eval = script_dir / cfg.EVAL_NB
    model_eval_html = results_dest / cfg.EVAL_NB_HTML
    return data_explore, data_exp_html, model_eval, model_eval_html


def execute_save_notebook(nb_path, nb_save_pth, settings_path):
    '''Sets up a temporary json file, to pass the settings path on to a notebook
    converts the notebook to a .py script, runs it, saves the output as a html'''
    script_dir = Path(__file__).resolve().parent
    data = {'settings_path' : str(settings_path)}
    print(f'Writing this to temp settings file: {data}')
    with open(script_dir / 'temp_settings_path.json', 'w') as f:
        json.dump(data, f)
    with open(nb_path) as f:
        nb_content = nbformat.read(f, as_version=4) 
    execute_preprocessor = ExecutePreprocessor(timeout=None)
    finished_nb = execute_preprocessor.preprocess(nb_content)[0]
    html_exporter = HTMLExporter()
    (finished_html, _) = html_exporter.from_notebook_node(finished_nb)
    with open(nb_save_pth, 'w', encoding="utf-8") as html_file:
        html_file.write(finished_html)


class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'
# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------
def main():
    cfg = Config()
    todo_exps, todo_runs, settings_list = check_for_unprocessed(cfg)
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir.parent / cfg.DATA_FOLDER_NM 
    project_dir = Path(__file__).resolve().parent.parent
    settings_dir = project_dir / cfg.SETTINGS_FOLDER_NM
    debug_settings_pth = script_dir.parent / cfg.SETTINGS_FOLDER_NM / cfg.DEBUG_SETTINGS_FLDR_NM / cfg.DEBUG_SETTINGS
    print(Colour.S + 'New Experiments: '+ Colour.E, todo_exps)
    print(Colour.S + 'New Runs: '+ Colour.E, todo_runs)
    
    if cfg.EVAL_ONLY:
        settings = Path(cfg.EVAL_ONLY)
        settings_list = [settings]
        run, exp = extract_run_exp(settings)
        todo_runs = {str(settings): (run, exp)}
        print(f'Running evaluation only, on the settings at {settings} \n' 
              f'using experiment folder {todo_runs[str(settings)][0]}\n'
              f'and run folder {todo_runs[str(settings)][1]}')

    while True:
        if not todo_exps and not todo_runs:
            response = input("There are no new settings files to process, do you want to run with the debug settings file? (y/n): ")
            if response == "y":
                print("Continuing...")
                cuda_data = torch.cuda.max_memory_allocated()
                print(f'There is {cuda_data} allocated data left in the GPU memory at the beginning of the script')
                todo_exps = {str(debug_settings_pth) :('Debug', 'Run_01')}
                settings_list = [debug_settings_pth]
                print(f'Todo exps: {todo_exps}')
            elif response == "n":
                sys.exit()

        for settings_path in settings_list:      
            if str(settings_path) in list(todo_exps.keys()):   # Make a new derived dataset
                exp_folder = data_folder / cfg.EXPS_FOLDER_NM / todo_exps[str(settings_path)][0]
                run_folder = exp_folder / cfg.RUNS_FOLDER_NM / todo_exps[str(settings_path)][1]
                results_folder = run_folder / cfg.RESULTS_FOLDER_NM
                inputs_folder = exp_folder / cfg.INPUT_FOLDER_NM
                settings_destn = inputs_folder / settings_path.name
                finished_destn = data_folder / cfg.FINISHED_SETTINGS / settings_path.name
                for fldr in [results_folder, inputs_folder]:
                    if not os.path.exists(fldr):
                        os.makedirs(fldr)
                data_nb, data_nb_out, eval_nb, eval_nb_out = get_nb_paths(cfg, inputs_folder, results_folder)
                print(data_nb, data_nb_out)

                Reload_Images.main(settings_path) #Runs megadetector or just moves the latest parquet and json files from the last run.
                if cfg.RUN_PREPROCESS:
                    print(Colour.S + f'Making a new dataset {todo_exps[str(settings_path)][0]}' + Colour.E)
                    print('Interpreting JSON MegaDetector Outputs')
                    Interpret_JSON.main(settings_path)
                    print('Running the data exploration notebook')            
                    execute_save_notebook(data_nb, data_nb_out, settings_path)
                    Clean_Data.main(settings_path)
                    Preprocess_Images.main(settings_path)

            elif str(settings_path) in list(todo_runs.keys()):
                exp_folder = data_folder / cfg.EXPS_FOLDER_NM / todo_runs[str(settings_path)][0]
                run_folder = exp_folder / cfg.RUNS_FOLDER_NM / todo_runs[str(settings_path)][1]
                results_folder = run_folder / cfg.RESULTS_FOLDER_NM
                inputs_folder = exp_folder / cfg.INPUT_FOLDER_NM
                settings_destn = inputs_folder / settings_path.name
                finished_destn = data_folder / cfg.FINISHED_SETTINGS / settings_path.name
                for fldr in [results_folder, inputs_folder]:
                    if not os.path.exists(fldr):
                        os.makedirs(fldr)
                _, _, eval_nb, eval_nb_out = get_nb_paths(cfg, inputs_folder, results_folder)

            if not cfg.EVAL_ONLY:
                Training.main(settings_path)
            max_cuda_data = torch.cuda.max_memory_allocated()
            now_cuda_data = torch.cuda.memory_allocated()
            print(f'There is {now_cuda_data} Bytes allocated data left in the GPU memory after training')
            print(f'There was a maximum of {max_cuda_data} bytes allocated during training')
            torch.cuda.empty_cache()
            execute_save_notebook(eval_nb, eval_nb_out, settings_path)

            if settings_path.name != 'Debug_Settings.yaml':
                if settings_destn.exists():  
                    settings_destn.unlink()
                shutil.copy(settings_path, finished_destn)
                settings_path.rename(settings_destn)

        torch.cuda.empty_cache()
        if cfg.EVAL_ONLY:
            break
        cfg = Config()
        todo_exps, todo_runs, settings_list = check_for_unprocessed(cfg)
        if not todo_exps and not todo_runs:
            print(f'Processing complete, there are no more settings files in {settings_dir}')
            break

if __name__ == '__main__':
    main()