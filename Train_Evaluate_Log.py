'''This is the master file that:
- Makes a list of existing run_IDs and experiment names.
- Makes a list of all settings files in the directory, and processes oldest first
- Checks for eperiment update, vs just run_ID update
- Any directories needed should be set up by the individual scripts being run
- Runs all the preprocessing to make a new dataset (If a new exp ID)
- Runs training script (New run ID only)
- Runs evaluation notebooks on input data, and final model
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
        self.RUN_PREPROCESS = True # Only effects new experiments, not new runs or EVAL_ONLY 
        self.EVAL_ONLY = None #'/media/olly/Red_SSD/Alita/Settings/Exp_40_Run_01.yaml'
        #None # r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Predator_AI_Project\Settings\Exp_39_Run_02.yaml' #None 
        #r'E:\Project\Settings\Exp_26_Run_02.yaml' #capacity_vals[np.isnan(capacity_vals)] = 1000   #intersting that this is needed! # Should skip all training, just run the eval notebook on this file
        #r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Predator_AI_Project\Settings\Exp_36_Run_17.yaml'


class Paths:
    DATA_FOLDER_NM = 'Data'
    EXPS_FOLDER_NM = 'Experiments'
    RUNS_FOLDER_NM = 'Runs'
    INPUT_FOLDER_NM = 'Inputs'
    SETTINGS_FOLDER_NM =  'Settings'
    RESULTS_FOLDER_NM = 'Results'
    DATA_FOLDER_NM = 'Data' 
    SETUP_DIR_NM = 'Setup'
    DESCRIPTION ='Running Evaluation Only'
    #DEBUG_SETTINGS = 'Debug_Settings.yaml'
    DATA_EXP_NB = 'Data_Exploration.ipynb'
    DATA_EXP_NB_HTML = 'Data_Exploration.html'
    EVAL_NB = 'Model_Evaluation.ipynb'
    EVAL_NB_HTML = 'Model_Eval.html'
    FINISHED_SETTINGS = 'Finished_Settings_Files'
    
    def __init__(self, experiment=None, run=None, settings_path=None):
        _script_dir = Path(__file__).resolve().parent
        self.project_dir = _script_dir.parent
        self.data_explore = _script_dir / self.DATA_EXP_NB
        self.model_eval = _script_dir / self.EVAL_NB
        self.data_folder = self.project_dir / self.DATA_FOLDER_NM
        self.settings_dir = self.project_dir / self.SETTINGS_FOLDER_NM
        #self.debug_settings_pth = self.project_dir / self.SETTINGS_FOLDER_NM / self.DEBUG_SETTINGS_FLDR_NM / self.DEBUG_SETTINGS
        if experiment is not None and run is not None:
            print(f'Experiment {experiment}, run {run}')
            self.exp_folder = self.data_folder / self.EXPS_FOLDER_NM / experiment
            print(f'the experiment folder is {str(self.exp_folder)}')
            self.run_folder = self.exp_folder / self.RUNS_FOLDER_NM / run
            
            self.results_folder = self.run_folder / self.RESULTS_FOLDER_NM
            self.inputs_folder = self.exp_folder / self.INPUT_FOLDER_NM
            self.settings_destn = self.inputs_folder / settings_path.name
            self.finished_destn = self.data_folder / self.FINISHED_SETTINGS / settings_path.name
            self.data_nb_out =  self.inputs_folder /  self.DATA_EXP_NB_HTML
            self.eval_nb_out = self.results_folder / self.EVAL_NB_HTML
            for fldr in [self.results_folder, self.inputs_folder]:
                if not os.path.exists(fldr):
                    print(f'Making a new folder {str(fldr)}')
                    os.makedirs(fldr)

            print(f'the run folder is {str(self.run_folder)}')
            #/media/olly/Red_SSD/Alita/Data/Experiments/Run_02/Runs/Exp_500
            print(f'the results folder is {str(self.results_folder)}')
            print(f'The output path for the eval notebook is {self.eval_nb_out}')
            #/media/olly/Red_SSD/Alita/Data/Experiments/Run_02/Runs/Exp_500/Results/Model_Eval.html


def check_for_settings(paths):
    '''Takes in the configuration instance, looks in project settings directory, 
    and returns a list of all the .yaml file path objects, with the oldest first'''
    project_dir = Path(__file__).resolve().parent.parent
    settings_dir = project_dir / paths.SETTINGS_FOLDER_NM
    settings_paths = list(settings_dir.glob("*.yaml"))
    file_info_tuples = [(path, os.path.getmtime(path)) for path in settings_paths]
    sorted_file_info = sorted(file_info_tuples, key=lambda x: x[1])
    ordered_settings_paths = [file_info[0] for file_info in sorted_file_info]
    return ordered_settings_paths


def list_past_runs(paths):
    '''Take the configuration instance and looks through the specified directories
    for past project experiments (each dataset change) or chang of run_ID 
    (hyperparameter change without re-creating a dataset)'''
    project_dir = Path(__file__).resolve().parent.parent
    exps_dir = project_dir / paths.DATA_FOLDER_NM / paths.EXPS_FOLDER_NM
    exp_run_list = []
    experiments =  [folder for folder in exps_dir.iterdir() if folder.is_dir() and folder.name not in ['Debug', 'MD_Last_Run']]
    for experiment in experiments:
        runs_folder = experiment / paths.RUNS_FOLDER_NM
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


def check_for_unprocessed(paths):
    settings_list = check_for_settings(paths)
    if settings_list:
        print(Colour.S + 'Settings files found: '+ Colour.E, [f.name for f in settings_list])
    all_past_runs = list_past_runs(paths)
    todo_exps, todo_runs = check_for_new_run_exp(settings_list, all_past_runs)
    return todo_exps, todo_runs, settings_list


def execute_save_notebook(nb_path, nb_save_pth, settings_path):
    '''Sets up a temporary json file, to pass the settings path on to a notebook
    converts the notebook to a .py script, runs it, saves the output as a html'''
    script_dir = Path(__file__).resolve().parent
    data = {'settings_path' : str(settings_path)}
    print(f'Writing this to temp settings file: {data}')
    with open(script_dir / 'temp_settings_path.json', 'w', encoding="utf-8") as f:
        json.dump(data, f)
    with open(nb_path) as f:
        nb_content = nbformat.read(f, as_version=4)

    if "widgets" in nb_content.metadata:
        del nb_content.metadata["widgets"]

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
    paths = Paths()
    todo_exps, todo_runs, settings_list = check_for_unprocessed(paths)
    
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
            sys.exit()

        for settings_path in settings_list:      
            if str(settings_path) in list(todo_exps):   # Make a new derived dataset
                experiment = todo_exps[str(settings_path)][0]
                run = todo_exps[str(settings_path)][1]
                exp_paths = Paths(experiment=experiment, run=run, settings_path=settings_path)
                print(exp_paths.data_nb_out, exp_paths.data_nb_out)

                Reload_Images.main(settings_path) #Runs megadetector or just moves the latest parquet and json files from the last run.
                
                ###########################################################################################################################
                #There is a bug here when calling the MD on windows to update on any new images
                #joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated.
                # # This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System
                # #to kill the worker.
                ###########################################################################################################################

                if cfg.RUN_PREPROCESS:
                    print(Colour.S + f'Making a new dataset {experiment}' + Colour.E)
                    print('Interpreting JSON MegaDetector Outputs')
                    Interpret_JSON.main(settings_path)
                    print(f'Executing the data exploration notebook, then saving to {exp_paths.data_nb_out}')        
                    execute_save_notebook(exp_paths.data_explore, exp_paths.data_nb_out, settings_path)
                    Clean_Data.main(settings_path)
                    Preprocess_Images.main(settings_path)

            elif str(settings_path) in list(todo_runs.keys()):
                experiment = todo_runs[str(settings_path)][0]
                run = todo_runs[str(settings_path)][1]
                exp_paths = Paths(experiment=experiment, run=run, settings_path=settings_path)
                print(exp_paths.data_explore, exp_paths.data_nb_out)

            if not cfg.EVAL_ONLY:
                Training.train(settings_path)
                max_cuda_data = torch.cuda.max_memory_allocated()
                now_cuda_data = torch.cuda.memory_allocated()
                print(f'There is {now_cuda_data} Bytes allocated data left in the GPU memory after training')
                print(f'There was a maximum of {max_cuda_data} bytes allocated during training')
            torch.cuda.empty_cache()
            print(f'Executing the evaluation notebook, then saving to {exp_paths.eval_nb_out}')
            execute_save_notebook(exp_paths.model_eval, exp_paths.eval_nb_out, settings_path)

            if exp_paths.settings_destn.exists():  
                exp_paths.settings_destn.unlink()
            shutil.copy(settings_path, exp_paths.finished_destn)
            settings_path.rename(exp_paths.settings_destn)

        torch.cuda.empty_cache()
        if cfg.EVAL_ONLY:
            break
        cfg = Config()
        todo_exps, todo_runs, settings_list = check_for_unprocessed(paths)
        if not todo_exps and not todo_runs:
            print(f'Processing complete, there are no more settings files in {paths.settings_dir}')
            break

if __name__ == '__main__':
    main()