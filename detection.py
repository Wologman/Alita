'''
This script is called up by the inference script to run a detection model and save detections.json to predict bounding boxes.
Currently using Dan Moris's version of MegaDetector forked from https://github.com/agentmorris/MegaDetector
Some minor changes were made, and saved in the same location as as a separate script: run_dectector_batch_modified
Commented out is the Pytorch Wildlife fork  https://github.com/microsoft/CameraTraps
At the time the Microsoft version seemed to lack adequate exception handling.  Hopefully this improves.
'''

from pathlib import Path
import os
import yaml
import argparse
from typing import Optional
#from test_cuda import test_cuda
#import json
#from PytorchWildlife.models import detection as pw_detection
#from PytorchWildlife import utils as pw_utils
#from supervision import Detections
from megadetector.detection.run_detector_batch_modified import run_megadetector


class DefaultConfig:
    def __init__(self):
        cpu_cores = os.cpu_count() or 1
        self.NUM_WORKERS = max(cpu_cores // 2, 0)  #Only use lots of cores on grunty machines
        if self.NUM_WORKERS <= 4:
            print(Colour.S + f'Grrr, using only {self.NUM_WORKERS} of the {self.NUM_WORKERS} total threads to reduce the risk of crushing your puny earth operating system' + Colour.E)
        else:
            print(Colour.S + f'There are {cpu_cores} logical threads avalaible, using {self.NUM_WORKERS}' + Colour.E)
        self.BATCH_SIZE = 4 if self.NUM_WORKERS == 0 else 8 #Giving it the best chance of working on puny machines


def get_config(settings_pth=None):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list =  []  #If anything in the config list needs evaluating, eg True, None etc.
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

def make_paths_relative(results, base_stem):
    base_path = Path(base_stem)
    for entry in results:
        img_path = Path(entry['img_id'])
        entry['img_id'] = str(img_path.relative_to(base_path))
    return results
    
# --------------------------- Functions & Classes-----------------------------------------
# ----------------------------------------------------------------------------------------
class Warn:  #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Colour: 
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'

# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------
def detect(image_dir: Path, 
           json_out_path: Path,
           detector_weights_pth: Path,
           model_name: str = 'md_V5',
           num_workers: int = 0, 
           threshold: float = 0.05,
           batch_size=2):
    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', 'tif')
    total_images = len([f.name for f in Path(image_dir).rglob('*') if f.suffix.lower() in exts])
    print(f'Running Detection over {total_images} images')

    print(f'the detector path is {detector_weights_pth}')

    args = [str(detector_weights_pth),
            str(image_dir),
            str(json_out_path),
            '--threshold', str(threshold),
            '--recursive',
            '--output_relative_filenames',
            '--quiet',
            '--loader_workers', str(num_workers),
            ]

    run_megadetector(args)

    #DEVICE = test_cuda()  # Use "cuda" if GPU is available "cpu" if no GPU is available
    #model_full_name = 'MegaDetector_V5'
    #detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
    #detection_model = model_full_name #placeholder
    #python $mdScriptPath $mdModel $imagesPath $mdOutputPath --recursive --output_relative_filenames --quiet



    #results = detection_model.batch_image_detection(image_dir, batch_size=batch_size, conf_thres=threshold)

    #with open(json_out_path, 'r') as f:
    #    results = json.load(f)

    #print('reloaded the results json')
    #results = put_back_empties(results, image_dir)   #Needed for PytorchWildlife
    #print('empties ahve been put back')
    #results = make_paths_relative(results, str(image_dir))  #Needed for PytorchWildlife
    #print('results made relative')

    #The default, save the detections.json to the highest level folder
    #output_file_timelapse = image_dir / "detections.json"
    #pw_utils.save_detection_timelapse_json(results, 
    #                                       output_file_timelapse,
    #                                       categories=detection_model.CLASS_NAMES,
    #                                       exclude_category_ids=[],
    #                                       exclude_file_path=None,
    #                                       info={"detector": model_full_name})
    
    #if json_out_path is not None:
    #    pw_utils.save_detection_timelapse_json(results, 
    #                                           json_out_path,
    #                                           categories=detection_model.CLASS_NAMES,
    #                                           exclude_category_ids=[],
    #                                           exclude_file_path=None,
    #                                           info={"detector": model_full_name})

    #print(f'Completed running detection model, saving predictions to {str(output_file_timelapse)}')
    return

if __name__ == '__main__':
    temp_images = '/media/olly/Red_SSD/Alita/data/demos'
    temp_destn = '/media/olly/Red_SSD/Alita/data/detections.json'
    weights_pth = '/media/olly/Red_SSD/Alita/models/md_v5a.0.0.pt' #fallback for development

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", type=str, default=temp_images, help="Filepath to the root directory of imagery to be processed")
    parser.add_argument("--detectionPath", type=str, default=temp_destn, help="Filepath for the destination for the detections")
    parser.add_argument("--detectWeights", type=str, default=weights_pth, help="Filepath for the megadetector weights")
    parser.add_argument("--modelName", type=str, default='mdV5', help="MD_V5, MD_V6 etc.")
    parser.add_argument("--numWorkers", type=int, default=0, help="Parallel processes to run (Currently not implemented)")
    parser.add_argument("--detectThreshold", type=int, default=0.05, help="min confidence for recording a bounding box")
    args = parser.parse_args()
    if (args.dataPath is not None) and (args.settingsPath is not None): 
        print(f'Running Inference.py on {args.dataPath}, with the settings file {args.settingsPath}')

    detect(image_dir=args.dataPath, 
           detector_weights_pth=args.detectWeights,
           json_out_path=args.detectionsPath,
           model_name=args.modelName, 
           num_workers=args.numWorkers,
           threshold=args.detectThreshold)