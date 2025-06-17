from pathlib import Path
import os
import yaml
import argparse
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

'''
def put_back_empties(detections, image_fldr):
    #Extracts a list of all images with detections, 
    #compares to a list of all images recursively in the folder
    #adds new entries for the missing images, with bounding box around all, and a confidence of 0
    empty_detection= Detections(
        xyxy = np.array([[0, 0, 1064, 1064]], dtype=np.float32),
        mask = None,
        #category=
        confidence = np.array([0], dtype=np.float32),
        class_id =  np.array([0]),
        tracker_id = None,
        data = {}
        )
    empty_label = ['empty 0']
    empty_coords = [[0.0, 0.0, 1.0, 1.0]]

    detection_ids = {Path(entry['img_id']) for entry in detections}
    extensions = {'.jpg', '.jpeg', '.mp4', '.avi', '.mov'}
    all_paths = {file_path for file_path in image_fldr.rglob('*') if file_path.suffix.lower() in extensions}
    no_boxes = list(all_paths - detection_ids)
    new_entries = [{'img_id': str(img_id), 
                    'detections': empty_detection, 
                    'normalized_coords':empty_coords,
                    'labels': empty_label} 
                        for img_id in no_boxes]
    updated_results = detections + new_entries
    return updated_results
    '''

# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------
def detect(external_image_dir, 
           model_name='md_V5',
           detector_weights_pth=None,
           num_workers=0, 
           threshold=0.05,
           json_out_path=None,
           batch_size=2):
    image_dir = Path(external_image_dir)
    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', 'tif')
    total_images = len([f.name for f in Path(image_dir).rglob('*') if f.suffix.lower() in exts])
    print(f'Running Detection over {total_images} images')
    
    #DEVICE = test_cuda()  # Use "cuda" if GPU is available "cpu" if no GPU is available

    if model_name == 'yolov9c':
        model_full_name = 'MegaDetector_V6'
        #detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="yolov9c")
    else: 
        model_full_name = 'MegaDetector_V5'
        #detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
        detection_model = model_full_name #placeholder

        #python $mdScriptPath $mdModel $imagesPath $mdOutputPath --recursive --output_relative_filenames --quiet

        if detector_weights_pth is None:
            detector_weights_pth = r'F:/Alita/Models/md_v5a.0.0.pt' #fallback for development

        print(f'the detector path is {detector_weights_pth}')

        args = [str(detector_weights_pth),
                str(external_image_dir),
                str(json_out_path),
                '--threshold', str(threshold),
                '--recursive',
                '--output_relative_filenames',
                '--quiet',
                '--loader_workers', str(num_workers),
               ]


        '''
        results = load_and_run_detector_batch(model_file=args.detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=args.threshold,
                                          checkpoint_frequency=args.checkpoint_frequency,
                                          results=results,
                                          n_cores=args.ncores,
                                          use_image_queue=args.use_image_queue,
                                          quiet=args.quiet,
                                          image_size=args.image_size,
                                          class_mapping_filename=args.class_mapping_filename,
                                          include_image_size=args.include_image_size,
                                          include_image_timestamp=args.include_image_timestamp,
                                          include_exif_data=args.include_exif_data,
                                          augment=args.augment,
                                          # Don't download the model *again*
                                          force_model_download=False,
                                          detector_options=detector_options,
                                          loader_workers=args.loader_workers,
                                          preprocess_on_image_queue=args.preprocess_on_image_queue)
        '''

        run_megadetector(args)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", type=str, default=None, help="Filepath to the root directory of imagery to be processed")
    parser.add_argument("--modelName", type=str, default=0, help="MD_V5, MD_V6 etc.")
    parser.add_argument("--numWorkers", type=int, default=0, help="Parallel processes to run (Currently not implemented)")
    parser.add_argument("--detectThreshold", type=int, default=0, help="min confidence for recording a bounding box")
    args = parser.parse_args()
    if (args.dataPath is not None) and (args.settingsPath is not None): 
        print(f'Running Inference.py on {args.dataPath}, with the settings file {args.settingsPath}')

    #temp_images = '/media/olly/Red_SSD/Alita/Data/Demos'
    #temp_images = '/media/olly/Red_SSD/Alita/Data/Independent_Images'
    #temp_images = '/media/olly/Red_SSD/Alita/Data/Empty_Images'
    #detect(num_workers = 0, external_image_dir=temp_images)

    detect(external_image_dir=args.dataPath, 
           model_name=args.modelName, 
           num_workers=args.numWorkers,
           threshold=args.detectThreshold)