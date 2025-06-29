'''
Performs inference on series of camera trap images
- Seperates vidio into temprorary images
- Searches for an existing MegaDetctor json output
- If none found, runs MegaDetector to predict bounding boxes
- Runs classification over each image by cropping to 480x480 pixels
- Extracts time-stamps in the image exif data, 
- Collates the predictions into a single most likely 'encounter' class for each burst of images
'''

import time
import os
import sys
from pathlib import Path
import shutil
import random
import ast
import gc
import json
import argparse
import warnings
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from typing import Optional, Tuple, List, Literal, Dict
from dataclasses import dataclass
import zlib
import base64
import re

#ML & Pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Computer vision
import cv2
import piexif
from detection import detect

#Local Modules
import process_video as process_video
from interpret_json import process_all_jsons
from check_weights import check_weights
import predictions_2_json as predictions_2_json
from test_cuda import test_cuda


@dataclass
class InferConfig:
    '''Default configuration class'''
    def __init__(self, num_workers=None):
        cpu_cores = os.cpu_count() or 1

        if num_workers is None:
            self.num_workers = max(cpu_cores // 4, 1)
            if self.num_workers <= 2:
                print(Colour.S + f'Grrr, using only {self.num_workers} of the {cpu_cores} total threads to reduce the risk of crushing your puny earth operating system' + Colour.E)
            else:
                print(Colour.S + f'There are {cpu_cores} logical threads avalaible, using {self.num_workers}' + Colour.E)
        else:
            self.num_workers = num_workers

        self.EXPERIMENT_NAME = '_'
        self.RUN_ID = '_'
        self.CLASS_JOINS = {'lizard':['skink', 'lizard'], 
                            'finch':['greenfinch', 'goldfinch', 'chaffinch'], 
                            'quail':['quail_california', 'quail_brown'], 
                            'deer':['deer', 'white_tailed_deer']}
        self.CLASS_NAME_CHANGE = {'penguin':'little_blue_penguin', 
                                  'song thrush':'thrush', 
                                  'NZ_falcon':'nz_falcon'}
        
        self.HARD_CLASSES = ['sparrow', 'rosella', 'mallard', 'yellow_eyed_penguin', 'weasel', 'goat', 'redpoll', 'morepork', 'cow', 'dog']
        self.BATCH_SIZE = 4 if self.num_workers == 0 else 8 #Giving it the best chance of working on puny machines
        self.MODEL_NAME = 'tf_efficientnetv2_s.in21k'
        self.HEAD_NAME = 'ClassifierHead' # Alternative: BasicHead
        self.MD_DETECTION_THRESHOLD = 0.05 # Sets the threshold for the MegaDetector.  Below this no BBOX is produced
        self.ENCOUNTER_WINDOW = 30 # Seconds to collate the scores for an encounter
        self.EXIF_DT_FORMATS = ['%Y:%m:%d %H:%M:%S', '%y:%m:%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']
        self.DETECTOR_NAME = 'md_v5'
        self.OUT_FIELDS = ['Target','Prediction', 'Probability', 'Second_Pred', 
                           'Second_Prob', 'Third_Pred', 'Third_Prob']
        self.ACTIVATION = 'Sigmoid'


@dataclass
class ImageConfig:
    '''Wrapper class for image processing parameters'''
    EDGE_FADE = False
    MIN_FADE_MARGIN = 0
    MAX_FADE_MARGIN = 0
    BUFFER = 0
    REMOVE_BACKGROUND = False
    IMAGE_SIZE = 480 #Final width and height in pixels for the cropped images
    RESIZE_METHOD = 'md_crop' #rescale' # alternatively 'md_crop', should match method used for training
    MD_RESAMPLE = True #If using md_crop, True downscales large md crop boxes to the image size
    EDGE_FADE = False
    MIN_FADE_MARGIN = 0.0
    MAX_FADE_MARGIN = 0.0
    IMAGE_SIZE = 480 #Should match the size the transformed crops were during training, and be no larger than the stored crop size
    CROP_SIZE = 600 #This is the size that the images will be cropped to as part of the localisation step
    INPUT_MEAN = [ 0.485, 0.456, 0.406 ] # mean to be used for normalisation, using values from ImageNet.
    INPUT_STD = [ 0.229, 0.224, 0.225 ] # stddev to be used for normalisation, using values from ImageNet.


class Paths:
    '''Wrapper class to create and store filepaths'''
    RESULTS_FOLDER_NM = 'results'
    MODELS_FOLDER_NM = 'models'
    CLASS_NAMES = '_class_names.json'
    DEFAULT_IMAGE_FOLDER_NM = 'independent_images' # 'vids_images_testing' 'irish_images'
    WEIGHTS_FOLDER_SUFFIX = '_weights'
    WEIGHTS_FN_SUFFIX = '_best_weights.pt'
    PREDS_CSV_SUFFIX_OUT = '_predictions.csv'
    MD_WEIGHTS_NM = 'md_v5a.0.0.pt'
    DATA_FOLDER_NM = 'data'
    SETUP_FOLDER_NM = 'setup'
    EXPS_FOLDER_NM = 'experiments'
    RUNS_FOLDER_NM = 'runs'
    SETTINGS_FOLDER_NM = 'settings'
    RESOURCES_FOLDER_NM = 'resources'
    NAMING_SCHEMES_CSV = 'name_map_edited.csv'
    IMAGE_FOLDER_PTH = '' #The full file path to some external image directory

    def __init__(self,
                 project_dir: Path,
                 run_id: str,
                 exp_name: str,
                 image_dir: str,
                 preds_dir: str,
                 weights_pth: str,
                 detector_weights_pth:str):
        self.EXP_NAME = exp_name
        self.RUN_ID = run_id
        self.project_dir = project_dir
        self.experiment_dir = self.project_dir / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / self.EXP_NAME
        _default_preds_dir = self.experiment_dir / self.RUNS_FOLDER_NM / self.RUN_ID / self.RESULTS_FOLDER_NM
        self.models_dir = self.experiment_dir / self.RUNS_FOLDER_NM / self.RUN_ID / self.MODELS_FOLDER_NM
        self.weights_dir = self.models_dir / f'{self.RUN_ID}{self.WEIGHTS_FOLDER_SUFFIX}'
        self.naming_schemes_pth = self.project_dir / self.RESOURCES_FOLDER_NM / self.NAMING_SCHEMES_CSV
        
        _default_image_dir = self.project_dir / self.DATA_FOLDER_NM / self.DEFAULT_IMAGE_FOLDER_NM
        _default_weights_pth = self.weights_dir / f'{self.RUN_ID}{self.WEIGHTS_FN_SUFFIX}'

        #TODO: Tidy this up!
        if image_dir: #User-supplied filepath from external scripts
            self.image_dir = Path(image_dir)
        elif self.IMAGE_FOLDER_PTH:  #Get filepath from the settings file
            self.image_dir = Path(self.IMAGE_FOLDER_PTH)
        else:  #Run what ever the default is at the moment
            self.image_dir = _default_image_dir

        if preds_dir: #User-supplied filepath from external scripts
            self.preds_dir = Path(preds_dir) / 'Alita_Predictions'
        else:  #Run what ever the default is at the moment
            self.preds_dir = _default_preds_dir / 'Alita_Predictions'

        self.weights_pth = Path(weights_pth) if weights_pth else _default_weights_pth
        self.detector_weights = detector_weights_pth

        if not os.path.exists(self.preds_dir):
            os.makedirs(self.preds_dir)

        self.out_csv_path = self.preds_dir /  f'{self.EXP_NAME}_{self.RUN_ID}{self.PREDS_CSV_SUFFIX_OUT}'
        self.out_detailed_csv_path = self.preds_dir /  f'{self.EXP_NAME}_{self.RUN_ID}_full{self.PREDS_CSV_SUFFIX_OUT}'
        
        if weights_pth:
            self.class_names_pth = self.weights_pth.parent / f'{self.EXP_NAME}_{self.RUN_ID}_class_names.json'
        else:
            self.class_names_pth = self.project_dir / _default_preds_dir / f'{self.RUN_ID}{self.CLASS_NAMES}'

        check_weights('classifier', self.weights_pth, load=True)
        check_weights('MegaDetector', Path(self.detector_weights))


def resolve_path(path_string):
    '''Standardise filepaths to an absolute Path instance'''
    path = Path(path_string)
    if path.is_absolute():
        return path
    else:
        base_dir = Path(__file__).parent.parent.resolve()
        return base_dir / path


def get_settings(settings_pth=None, num_workers=None):
    """Gets an instance of the configuration classes, then looks for the settings file, 
    if it finds a matching key the value is updated, or evaluated to python expressions then updated"""
    
    evaluate_list = ['REMOVE_BACKGROUND', 'EDGE_FADE', 'CLASS_NAME_CHANGE',
                     'CLASS_JOINS', 'EXIF_DT_FORMATS']
    
    infer_settings = InferConfig(num_workers=num_workers)
    image_settings = ImageConfig()
    
    if settings_pth is not None:
        settings_pth = resolve_path(settings_pth)
        with open(settings_pth, 'r') as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        for key, value in yaml_data.items():
            for cfg in [infer_settings, image_settings]:
                if hasattr(cfg, key):
                    if (key in evaluate_list) and (isinstance(value, str)):
                        setattr(cfg, key, ast.literal_eval(value))
                    else:
                        setattr(cfg, key, value)

    return infer_settings, image_settings

# --------------------------- Functions & Classes-----------------------------------------
# ----------------------------------------------------------------------------------------
class Warn:  #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'

class Colour: 
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def check_for_empty(image_dir):
    extensions = {'.jpg', '.jpeg', '.mp4', '.avi', '.mov'}
    print("Checking for suitable media files (.jpg, .jpeg, .mp4, .avi & .mov)")
    print(f'looking in {image_dir}')
    media_files = {file_path for file_path in tqdm(image_dir.rglob('*')) 
                    if file_path.is_file() and file_path.suffix.lower() in extensions}
    if media_files:
        print(Colour.S + f'{len(media_files)} media files found for classification' + Colour.E)
        return False
    else:
        print(Warn.S + 'No media files found in your chosen folder, terminating the process')
        return True


def df_from_json(json_pths, species_list, image_folder):
    if json_pths:
        df = process_all_jsons(json_pths, species_list, image_folder=image_folder, best_only=True)
        df['Targets'] = df['Species']  #why did I do this?????
    else:
        print(Warn.S + "\nIt appears you are trying to use the MegaDetector for object localisation, but it has not prduced any bounding boxes." + Warn.E)
        print(Colour.S + 'Proceeding without localisation data, but this is expected to hurt performance' + Colour.E)
        df = make_dataframe(image_folder, species_list)
    return df


def make_dataframe(img_dir, class_list):
    def check_cls(class_nm, class_list):
        return 'unknown' if class_nm not in class_list + ['empty'] else class_nm
    file_names = [str(f) for f in Path(img_dir).rglob('*.jpg')]
    parent_names = [str(Path(fn).parent.name) for fn in file_names]
    class_names = [check_cls(nm, class_list) for nm in parent_names]
    xmins = ymins = [0] * len(class_names)
    widths = heights = [1] * len(class_names)
    confidences = [1] * len(class_names)
    df = pd.DataFrame({'File_Path': file_names, 'Targets': class_names, 'Confidence': confidences, 'x_min': xmins, 
                       'y_min': ymins, 'Width': widths, 'Height': heights, })
    return df


def update_target_names(df, name_changes, joins, verbose=True):
    """Updates the target names as per the settings file, so that the target names match the
    names used for training"""
    for key, value in name_changes.items():
        df['Targets'].replace(key, value, inplace=True)
    new_names = {item: key for key, items in joins.items() for item in items}
    for key, value in new_names.items():
        df['Targets'].replace(key, value, inplace=True)

    if len(list(df['Targets'].unique())) and verbose:
        print(Colour.S + 'The following list contains all the unique folder labels in your file structure that match class names' + Colour.E)
        print(df['Targets'].unique())
    return df


def run_megadetector(image_dir: Path, 
                     detector_weights_pth: str,
                     destn_fldr: Optional[Path] = None,
                     detector_name: str = 'md_v5',
                     detection_threshold: float = 0.05):
    
    #Store the MegaDetector output in the destination directory rather than with the images if possible
    root = destn_fldr if destn_fldr else image_dir
    json_path = str(root / 'detections.json')

    json_pths = [f for f in Path(image_dir).glob('*.json') if f.name == 'detections.json']
    if len(json_pths) == 1 and destn_fldr:
        shutil.copy2(json_pths[0], json_path)
        print(Colour.S + f'{len(json_pths)} An existing MegaDetector predictions detections.json file found in {image_dir} will be used'+ Colour.E)
    
    if not json_pths:
        detect(image_dir,
               json_out_path = Path(json_path),
               detector_weights_pth= Path(detector_weights_pth),  # detector_weights_pth,
               model_name=detector_name,
               threshold = detection_threshold,
               )
    return json_path


def get_dataframe(image_dir: Path, 
                  json_pths: List[str],
                  species_list: List[str],
                  resize_method: Literal['md_crop', 'rescale'] = 'md_crop',
                  class_name_change: dict = {},
                  class_joins: dict = {},
                  verbose: bool = True):

    if resize_method == 'md_crop':
        df = df_from_json(json_pths, species_list, str(image_dir)) 
    else:  # make a dataframe without detected bounding boxes (set them to the image boundary)
        df = make_dataframe(image_dir, species_list)
    df = update_target_names(df, class_name_change, class_joins, verbose=verbose)
    return df


def data_from_json(data_pth):
    with open(data_pth, 'r') as f:
        data = json.load(f)
    return data


def get_exif_dt(jpeg_bin, potential_formats=['%Y:%m:%d %H:%M:%S']):
    """Takes an image binary file, and the potential date formats, returns a string object 
    with %d/%m/%Y %H:%M:%S"""
    try:
        exif_data = piexif.load(jpeg_bin)
        if 'Exif' in exif_data:
            exif_dict = exif_data['Exif']
            datetime_original = exif_dict.get(piexif.ExifIFD.DateTimeOriginal, None)
            if datetime_original:
                datetime_original_str = datetime_original.decode('utf-8')  # Decode from bytes to string
                for format_string in potential_formats:
                    try:
                        formatted_dt = datetime.strptime(datetime_original_str, format_string)
                        return formatted_dt.strftime("%d/%m/%Y %H:%M:%S")  # Return the formatted string
                    except ValueError:
                        pass  # Continue to the next format if parsing fails
    except piexif.InvalidImageDataError:
        print("Invalid EXIF data in the image.")
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None


class ImagePrep():
    def __init__(self, mean, std, image_size):
        self.transform = A.Compose([
                A.CenterCrop(height=image_size, width=image_size, p=1),
                A.Normalize(mean=mean, std=std), ToTensorV2()])


class PredatorDataset(Dataset):
    def __init__(self,
                 labels_df: pd.DataFrame,
                 transform,
                 img_cfg: ImageConfig,
                 dt_formats: List[str]):

        self.df = labels_df
        self.transform = transform
        self.crop_size = img_cfg.CROP_SIZE
        self.resize_method = img_cfg.RESIZE_METHOD
        self.remove_background = img_cfg.REMOVE_BACKGROUND
        self.fade_edges = img_cfg.EDGE_FADE
        self.min_margin = img_cfg.MIN_FADE_MARGIN
        self.downsample = img_cfg.MD_RESAMPLE # Downsample image if the MD crop box size is > crop_size
        self.buffer = img_cfg.BUFFER # The fraction of the image w/h that is kept around the md bounding box
        self.counter = 0 
        self.dt_formats = dt_formats

    def __len__(self):
        return len(self.df)

        
    def load_image(self, image_path, mode):
        try:
            image_path = Path(image_path)  # Ensure it's a Path object
            with image_path.open('rb') as in_file:
                jpeg_buf = in_file.read()
                date_time = get_exif_dt(jpeg_buf, self.dt_formats)

            # Use imdecode to support Unicode paths on Windows + PyInstaller
            file_bytes = np.frombuffer(jpeg_buf, dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is not None and mode == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image, date_time

        except Exception as e:
            print(f"Warning: Unable to load image at '{image_path}': {e}")
            return None, None
    
    def edge_fade(self,
                  image: np.ndarray,
                  min_margin: float = 0.05,
                  max_margin: float = 0.05):
        '''Accepts an image array and looks for any black space around it if a max_margin 
        is given that is larger than the min_margin then a random width fading buffer will
        be created.  Otherwise a fading buffer = min_margin will be created'''
        def get_lin_array(margin, length):
            start = np.linspace(0, 1, margin)
            end = np.linspace(1, 0, margin)
            middle = np.ones(length-2*margin)
            return np.concatenate((start, middle, end))
        
        height, width, channels = image.shape
        dtype = image.dtype
        new_image = np.zeros((height, width, channels), dtype=dtype) 
        relative_margin = min_margin + random.random() * (max_margin-min_margin)
        non_zero_rows, non_zero_cols, _ = np.nonzero(image)
        left = np.min(non_zero_cols)
        top = np.min(non_zero_rows)
        right = np.max(non_zero_cols)
        bottom = np.max(non_zero_rows)
        crop_width = right-left
        crop_height = bottom - top
        margin = int(relative_margin * min(crop_width, crop_height))
        horizontal = get_lin_array(margin, crop_width)
        vertical = get_lin_array(margin, crop_height)
        mask = np.outer(vertical, horizontal)
        crop = image[top:bottom, left:right]
        if crop.shape[-1] == 1:
            faded_crop = crop * mask
        else:
            faded_crop = crop * mask[:, :, np.newaxis]
        new_image[top:bottom, left:right] = faded_crop #broadcast on to the black background
        return new_image

    def subtract_background(self,
                            image: np.ndarray,
                            row: pd.Series,
                            buffer: float
                            ):
        height, width, channels = image.shape
        dtype = image.dtype
        new_image = np.zeros((height, width, channels), dtype=dtype)
        clamp = lambda n: max(min(1, n), 0)
        x_min = int(clamp(row['x_min'] - buffer)*width)
        y_min = int(clamp(row['y_min'] - buffer)*height)
        x_max = int(clamp(row['x_min'] + row['Width'] + buffer)*width)
        y_max = int(clamp(row['y_min'] + row['Height'] + buffer)*height)
        image = image[y_min:y_max, x_min:x_max] #crop the image
        new_image[y_min:y_max, x_min:x_max] = image #broadcast on to the black background
        return new_image

    def crop_to_square(self,
                       image: np.ndarray
                       ):
        height, width = image.shape[:2]
        min_side_length = min(height, width)
        top = (height - min_side_length) // 2
        bottom = top + min_side_length
        left = (width - min_side_length) // 2
        right = left + min_side_length
        return image[top:bottom, left:right]

    def pad_to_square(self,
                      image: np.ndarray
                      ):
        height, width, channels = image.shape
        dtype = image.dtype
        max_dim = max(height, width, self.crop_size)
        square_image = np.zeros((max_dim, max_dim, channels), dtype=dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_image[y_offset:y_offset+height, x_offset:x_offset+width]= image
        return square_image, x_offset, y_offset

    def rescale_image(self, 
                      image: np.ndarray
                      ):
        size=self.crop_size
        crop = self.crop_to_square(image)
        return cv2.resize(crop, (size, size))
    
    def get_mega_crop_values(self,
                             row: pd.Series,
                             img_w: int,
                             img_h: int,
                             final_size: int
                             ):
        x_min, y_min, width, height = row['x_min'], row['y_min'], row['Width'], row['Height']
        # Megadetector output is [x_min, y_min, width_of_box, height_of_box] top left (normalised COCO)
        # Want to output a square centred on the old box, with width & height = final_size
        x_centre = (x_min + width/2) * img_w
        y_centre = (y_min + height/2) * img_h
        left = int(x_centre - final_size/2)
        top =  int(y_centre - final_size/2)
        right = left + final_size
        bottom = top + final_size

        # Corrections for when the box is out of the original image dimensions. Shifts by that amount
        if (left < 0) and (right > img_w):
            new_left, new_right = 0, final_size
        else:
            new_left   = left  - (left < 0) * left - (right > img_w)*(right - img_w)
            new_right  = right - (left < 0) * left - (right > img_w)*(right - img_w)
        
        if (top < 0) and (bottom > img_h):
            new_top, new_bottom = 0, final_size
        else:
            new_top    = top    - (top < 0) * top - (bottom > img_h) * (bottom - img_h)
            new_bottom = bottom - (top < 0) * top - (bottom > img_h) * (bottom - img_h)
        
        return new_left, new_top, new_right, new_bottom

    #Check if the MD crop box is larger than the final image size.  If so, downscale the whole image
    def get_new_scale(self,
                      row: pd.Series,
                      buffer: float,
                      width: int,
                      height: int,
                      final_size: int
                      ):
        '''
        Calculates how much to scale down the new image to, 
        so the max(bounding-box) + buffer = the desired crop size.
        Only effects images where the crop box would be greater than the crop size.
        '''
        clamp = lambda n: max(min(1, n), 0)
        x_min = clamp(row['x_min'] - buffer)
        y_min = clamp(row['y_min'] - buffer)
        x_max = clamp(row['x_min'] + row['Width'] + buffer)
        y_max = clamp(row['y_min'] + row['Height'] + buffer)
        max_dimension = max([(x_max - x_min)*width, (y_max - y_min)*height]) 
        return final_size/max_dimension if max_dimension > final_size else None

    def md_crop_image(self,
                      row: pd.Series,
                      image_arr: np.ndarray
                      ):
        '''
           1. Select the highest confidence megadetector detection
           2. Downsize the image if the bounding box is > than the expected crop size
           3. Convert the megadetector bbox to absolute pixel locations [left, top, right, bottom]
           4. Crop the numpy array
        '''
        img_buffer = self.buffer
        resample =  self.downsample
        size=self.crop_size
        img_h, img_w = image_arr.shape[:2]

        if resample:
            scale = self.get_new_scale(row, img_buffer, img_w, img_h, size)
            if scale is not None:
                img_w, img_h = int(round(img_w * scale)), int(round(img_h * scale))
                image_arr = cv2.resize(image_arr, (img_w, img_h), cv2.INTER_LANCZOS4)

        left, top, right, bottom = self.get_mega_crop_values(row, img_w, img_h, size)
        cropped_arr = image_arr[top:bottom, left:right]
        
        crop_h, crop_w = cropped_arr.shape[:2] # both = size, unless one dimension was too small
        if (crop_h < size) or (crop_w < size):
            cropped_arr, _, _ = self.pad_to_square(cropped_arr)

        #normalise the ltrb values to the whole image, which may have been scaled down
        norm_crop = [round(x, 4) for x in [left/img_w, top/img_h, right/img_w, bottom/img_h]]
        return cropped_arr, norm_crop

    def __getitem__(self, index):
        self.counter += 1

        row = self.df.iloc[index]
        f_path = row['File_Path']
        target = row['Targets']
        image, date_time = self.load_image(f_path, 'RGB')
        if image is None:
            print(f"Warning: Unable to load the image at '{f_path}'. Skipping...")
            return None, None, f_path, None, None
        
        if self.remove_background:
            image = self.subtract_background(image, row, self.remove_background)
        if self.fade_edges:
            image = self.edge_fade(image, min_margin=self.min_margin, max_margin=self.min_margin)
        if self.resize_method == 'rescale':
            image = self.rescale_image(image)#Just crops and downsamples image to a square of required size
            ltrb_norm = [0,0,1,1]
        else:
            image, ltrb_norm = self.md_crop_image(row, image) #Uses MegaDetector bounding boxes to localise animal
            
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image, target, f_path, str(ltrb_norm), date_time


class ClassifierHead(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes:int,
                 dropout_rate: float = 0
                 ):
        super().__init__()
        self.linear = nn.Linear(num_features, num_features//2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(num_features//2, num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class BasicHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(BasicHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomModel(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 model_name='efficientnetv2_l_21k',
                 head_name='ClassifierHead'
                 ):
        super().__init__()
        self.custom_head = ClassifierHead if head_name =='ClassifierHead' else BasicHead
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.in_features = self.backbone.classifier.in_features
        print(f'There are {self.in_features} input features to the classifier head and {self.num_classes} outputs')
        
        self.backbone.classifier = self.custom_head(self.in_features, self.num_classes)

    def forward(self, images):
        logits = self.backbone(images)
        return logits


def get_model(weights, num_classes, model_name='efficientnetv2_l_21k', head_name='ClassifierHead'):
    print(f'The path to the weights file is {weights}')
    print(f'The model name is {model_name}')
    print(f'The classfier head name is {head_name}')
    saved_state_dict = torch.load(weights)
    model = CustomModel(num_classes, model_name, head_name=head_name)
    model.load_state_dict(saved_state_dict)
    model.eval()
    return model


def infer_dataset(loader, model, class_list, device, activation='Sigmoid', verbose=True):
    
    num_samples = len(loader.dataset)
    print(Colour.S + f'There are {num_samples} images in the dataloader for inference.' + Colour.E)
    
    model.eval()
    model.to(device)
    targets_list, predictions_list, paths_list, crops_list, dt_list = [], [], [], [], []
    start_time = time.time()
    
    try:
        for batch in tqdm(loader):
            if batch is None:
                if verbose:
                    print("Skipping batch due to None")
                continue

            images, targets, paths, ltrb_dims, d_times = batch

            if images is None or targets is None or paths is None or ltrb_dims is None or d_times is None:
                if verbose:
                    print(f"Skipping batch due to None elements: {batch}")
                continue

            images = images.to(device)
            with torch.no_grad():
                logits = model(images)

            if activation == 'Sigmoid':
                probs = F.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=1)
            cpu_probs = probs.detach().cpu().numpy()
            targets_list.extend(targets)
            predictions_list.extend(cpu_probs)
            paths_list.extend(list(paths))
            crops_list.extend(list(ltrb_dims))
            dt_list.extend(d_times)
            del images, logits, probs, cpu_probs
            torch.cuda.empty_cache()
    except TypeError as e:
        print(f"Error: {e}")
        print('Fatal error: No valid images were found for inference')
        print('This could occur because the filepaths in the MegaDetector output: mdPredictions.json are out of date. Try deleting this file and re-run')
        print('It could also be possible that all the images intended for inference are corrupted and cannot be opened')
        sys.exit()
        
    total_time = time.time()-start_time
    preds_array = np.vstack(predictions_list)
    speed = num_samples/total_time
    if verbose:
        print(f'The classifier processed {num_samples} samples in {total_time:.2f} seconds')
        print(f'That is a mean of {speed:.2f} images per second')

    pred_df = pd.DataFrame(preds_array, index=paths_list, columns=class_list)
    targ_df= pd.DataFrame(targets_list, index=paths_list, columns=['Targets'])

    del model
    torch.cuda.empty_cache()

    results = {'targ_df': targ_df,
               'pred_df': pred_df,
               'crops_list': crops_list,
               'dt_list': dt_list,
               'speed': speed}
    
    return results


def custom_collate(batch):
    """Determines how to collate the items from the dataset __getitem__, and handle None values"""
    batch = [sample for sample in batch if sample[0] is not None]
    if not batch:
        return None
    images, targets, paths, ltrb_dims, d_times = zip(*batch)
    images = torch.stack(images)
    return images, targets, paths, ltrb_dims, d_times


def separate_vid_img(df: pd.DataFrame):
    """Splits off the video images into a different dataframe for later collation"""
    vid_df = df[df.index.str.contains('video_')]
    img_df = df[~df.index.str.contains('video_')]
    return vid_df, img_df


def collate_encounters(df: pd.DataFrame,
                       time_window: int = 30,
                       ):
    """Groups the dataframe by the parent folder, then runs the collate_encounters_per_folder function
    on each seperately.  Speeds up the sorting, and also prevents confusion between cameras that happen
    to be set off in different locations at the same time."""
    df = df.copy()
    df['Parent_Folder'] = df.index.map(lambda x: str(Path(x).parent))
    grouped_dfs = []
    for _, group_df in df.groupby('Parent_Folder'):
        grouped_df_result = collate_encounters_per_folder(group_df, time_window)
        grouped_dfs.append(grouped_df_result)
    result_df = pd.concat(grouped_dfs)
    result_df.drop('Parent_Folder', axis=1, inplace=True)
    return result_df


def collate_encounters_per_folder(df: pd.DataFrame,
                                  time_window: int = 30
                                  ):
    """Looks at the previous image, and if it was within 30 seconds adds it to the same enc"""
    df=df.copy()
    df['dt_object'] = pd.to_datetime(df['Date_Time'], dayfirst=True,  errors='coerce', format="%d/%m/%Y %H:%M:%S")
    df.sort_values(by='dt_object', inplace=True, ascending=True)
    df.index.name = 'Image_File_Paths'  #Names the index
    df = df.reset_index(drop=False)  #Turns the index into a regular column
    df.loc[0, 'time_difference'] = timedelta(days=0, seconds=0, microseconds=0)
    df.loc[1:,'time_difference'] = df['dt_object'].diff()
    df.loc[0, 'Encounter_Start'] = df.loc[0, 'dt_object']
    df.loc[df['time_difference'] > pd.Timedelta(seconds=time_window), 'Encounter_Start'] = df['dt_object'] #Set the date-time for the encounter-starts
    df['Encounter_Start'] = df['Encounter_Start'].ffill() # Fill up the next empties.  Holey moley, what a useful method!

    max_prob_rows = df.loc[df.groupby('Encounter_Start')['Probability'].idxmax()]
    df['Encounter'] = df['Encounter_Start'].map(max_prob_rows.set_index('Encounter_Start')['Prediction'])
    df['Max_Prob'] = df['Encounter_Start'].map(max_prob_rows.set_index('Encounter_Start')['Probability'])

    # Compute max Confidence per Encounter_Start (regardless of max prob row)
    max_conf_per_encounter = df.groupby('Encounter_Start')['Confidence'].max()
    df['max_confidence'] = df['Encounter_Start'].map(max_conf_per_encounter)

    # Set 'Encounter' column equal to 'Prediction' for rows with empty 'Date_Time'
    df.loc[df['Date_Time'].isna(), 'Encounter'] = df.loc[df['Date_Time'].isna(), 'Prediction']
    df.loc[df['Date_Time'].isna(), 'Max_Prob'] = df.loc[df['Date_Time'].isna(), 'Probability']
    df.set_index('Image_File_Paths', inplace=True, drop=True)  #Puts the index back as it was before
    df = df.drop(['dt_object', 'time_difference', 'File_Path'], axis=1)
    df['Encounter_Start'] = df['Encounter_Start'].dt.time
    #first_cols = ['Date_Time', 'Encounter_Start', 'Encounter', 'Max_Prob', 'max_confidence']  #'Image_File_Paths', 
    first_cols = ['Date_Time', 'Encounter_Start', 'Encounter', 'Max_Prob', 'Targets', 'Prediction', 
                'Probability', 'Second_Pred', 'Second_Prob', 'Third_Pred', 'Third_Prob', 'Crop', 
                'x_min', 'y_min', 'Width', 'Height', 'Confidence', 'max_confidence']
    new_columns_order = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[new_columns_order]
    return df


def agg_rows_by_mean(df):
    """This is for video, to take the mean of the probabilities by parent directory 
    (which corresponds to a single video file).  Then reduce the group to a single
    line where the max score, and the predicted class come from those agregated probs
    There is duplication here, it would be better done earlier
    """
    possible_text_cols = ['parent_dir', 'File_Path', 'Date_Time', 'Probability', 'Second_Prob', 'vid_path',
                'Third_Prob', 'Prediction', 'Second_Pred', 'Third_Pred', 'Confidence',
                'Targets', 'x_min', 'y_min', 'Width', 'Height', 'Crop', 'max_confidence']
    
    text_cols = [col for col in possible_text_cols if col in df.columns]
    prob_scores = [col for col in df.columns if col not in text_cols]
    agg_dict = {col: 'mean' for col in prob_scores}
    agg_dict.update({col: 'first' for col in text_cols})
    agg_dict.update({'Confidence':'max' })
    agg_df = df.groupby('parent_dir').agg(agg_dict)
    agg_df.reset_index(drop=True, inplace=True)
    agg_df['Probability'] = agg_df[prob_scores].max(axis=1)
    agg_df['Prediction'] = agg_df[prob_scores].idxmax(axis=1)
    agg_df['Second_Prob'] = agg_df[prob_scores].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    agg_df['Second_Pred'] = agg_df[prob_scores].apply(lambda row: row.nlargest(2).index[-1], axis=1)
    agg_df['Third_Prob'] = agg_df[prob_scores].apply(lambda row: row.nlargest(3).iloc[-1], axis=1)
    agg_df['Third_Pred'] = agg_df[prob_scores].apply(lambda row: row.nlargest(3).index[-1], axis=1)
    return agg_df


def decompress_path(encoded: str) -> str:
    """Decode a filename-safe string back to the original path."""
    b64_bytes = encoded.encode("ascii")                   # 1. Convert to bytes
    compressed = base64.urlsafe_b64decode(b64_bytes)      # 2. Decode from base64
    utf8_bytes = zlib.decompress(compressed)              # 3. Decompress
    return utf8_bytes.decode("utf-8")                     # 4. Decode to original string


def collate_video(df: pd.DataFrame,
                  threshold: float,
                  use_mean_scores: bool = False):
    """
    Collects all the image results from the same video, and makes a prediction based on the highest probability score,
    This is then applied to the video file as a whole, and the filepath is put back to  the original video filepath
    """
    
    df = df.copy()
    df['vid_path'] = [decompress_path(re.sub(r"_\d+$", "", Path(index_value).stem)) for index_value in df.index]  #jkjofsuiwe2jkfdui_0008
    df['parent_dir'] = [Path(index_value).parent for index_value in df.index] #parent folder
    df['grandparent'] = [Path(index_value).parent.parent for index_value in df.index]  #grandparent folder
    grandparents = df['grandparent'].unique().tolist()
    df = df.drop(['grandparent'], axis=1)

    if use_mean_scores:
        df = agg_rows_by_mean(df)  #gets the average scores, then recalculates the first, second, third probability
    idxmax_rows = df.groupby('parent_dir')['Probability'].idxmax()  #finds the row that has the maximum first probability
    max_df = df.loc[idxmax_rows].copy()  #Simple max scoring row for each video
    max_conf_per_encounter = df.groupby('parent_dir')['Confidence'].max()
    df['max_confidence'] = df['parent_dir'].map(max_conf_per_encounter)

    #Now do the same but only with more confident images above some MegaDetector Threshold
    #Avoids having empty images messing up the probability aggregation
    animal_df = df[df['Confidence'] > threshold].copy() 
    if use_mean_scores:
        animal_df = agg_rows_by_mean(animal_df)
    max_animal_rows = animal_df.groupby('parent_dir')['Probability'].idxmax()
    max_animal_df =  animal_df.loc[max_animal_rows].copy()

    #now combine the two dataframes, so that any videos where there were no confident boxes still have some prediction.
    animals_only_list = max_animal_df['parent_dir'].tolist()
    less_confident_list = max_df['parent_dir'].tolist()
    missing_parents = list(set(less_confident_list) - set(animals_only_list))
    missing_rows = max_df[max_df['parent_dir'].isin(missing_parents)]
    df = pd.concat([max_animal_df, missing_rows], ignore_index=True)

    df = df.drop(['parent_dir', 'File_Path'], axis=1)
    df.set_index('vid_path', inplace=True, drop=True)
    df.index.name = None
    df['Encounter_Start']= pd.to_datetime(df['Date_Time'], dayfirst=True, format="%d/%m/%Y %H:%M:%S").dt.time
    df['Encounter'] = df['Prediction']
    df['Max_Prob'] = df['Probability']
    first_cols = ['Date_Time', 'Encounter_Start', 'Encounter', 'Max_Prob', 'Targets', 'Prediction', 
                'Probability', 'Second_Pred', 'Second_Prob', 'Third_Pred', 'Third_Prob', 'Crop', 
                'x_min', 'y_min', 'Width', 'Height', 'Confidence', 'max_confidence']

    new_columns_order = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[new_columns_order]
    
    #Delete the Temp_Frames directory
    for directory in grandparents:
        if Path(directory).name == 'Temp_Frames':  #Shouldn't be needed, just added safety
            try:
                shutil.rmtree(directory)
                print(f"Deleted: {directory}")
            except Exception as e:
                print(f"Error deleting {directory}: {e}")

    return df


def append_missing_files(missing_files, df):
    """Fill in the Dataframe rows for any files that didn't make it through the classification process"""
    missing_df = df.iloc[0:0, :].copy()
    missing_df['filename'] = missing_files
    missing_df['Encounter'] = 'unprocessed'
    missing_df['Targets'] = 'unprocessed'
    missing_df['Prediction'] = 'unprocessed'
    missing_df.set_index('filename', inplace=True)
    df = pd.concat([df, missing_df])
    return df


def relabel_empties(df: pd.DataFrame, 
                    md_empty_threshold: float = 0.15, 
                    preds_empty_threshold: float = 0.5,
                    hard_classes: list[str] = [],
                    empty_string: str = 'Empty'):
    """Decide with what to do with images where both MD and the classifier produce low scores"""
    
    df['File_Path'] = df.index
    # Set the individual image preds (not the 'enctounter' to zero if:
    # (1)  Both models are below their thresholds  OR
    # (2)  The classifier is below the threshold, and the max class score is not one that we have trouble with.
    rows_to_modify = ( (df['Confidence']  < 0.05)
                     | ((df['Confidence']  < md_empty_threshold)) & (df['Probability'] < preds_empty_threshold))  #Both below the threshold
                     #Removed the next line, since this is the situation we are going to call 'unknown'.
                     #| ((df['Probability'] < preds_empty_threshold) & ~df['Prediction'].isin(hard_classes))) 
                     #  (Only classifier is below it's threshold, and its not a hard class
    cols_to_zero = ['Probability', 'Second_Prob', 'Third_Prob']
    cols_to_empty = ['Prediction', 'Second_Pred', 'Third_Pred']
    df.loc[rows_to_modify, cols_to_zero] = 0
    df.loc[rows_to_modify, cols_to_empty] = empty_string
    return df


def relabel_unknowns(df: pd.DataFrame, 
                     md_empty_threshold: float = 0.15, 
                     preds_empty_threshold: float = 0.5,
                     unknown_string: str = 'Unknown'):
    """Decide with what to do with images where MD score is high, but classifier is low"""
    
    df['File_Path'] = df.index
    # Set the individual image preds (not the 'enctounter' to zero if:
    # (1)  Both models are below their thresholds  OR
    # (2)  The classifier is below the threshold, and the max class score is not one that we have trouble with.
    rows_to_modify = ((df['Probability'] < preds_empty_threshold) & (df['Confidence'] > md_empty_threshold))   #Only classifier is below it's threshold, and its not a 
    cols_to_unknown = ['Prediction', 'Second_Pred', 'Third_Pred']
    df.loc[rows_to_modify, cols_to_unknown] = unknown_string
    return df


def make_results_table(results_dict: dict, 
                       bbox_df: pd.DataFrame, 
                       class_name_map: List, 
                       time_window: int, 
                       md_empty_threshold: float = 0.15,
                       empty_classify_threshold: float = 0.5, #If nothing over this, we predict 'unknown if md_empty is met'
                       hard_classes: List = [],
                       empty_string: str = 'Empty',
                       unknown_string: str = 'Unknown',
                       use_mean_video_preds: bool = False,
                       verbose: bool = True):
    """Join all the predictions and targets, and assemble into the final table for saving and analysis
    This function could be improved by collating video near the start, prior to subsequent processing"""

    targs_df = results_dict['targ_df']
    preds_df = results_dict['pred_df']
    final_crops = results_dict['crops_list']
    dt_list = results_dict['dt_list']

    def get_nth_pred(arr, nth):
        sorted_indices = np.argsort(arr, axis=1)
        sorted_rows = np.sort(arr, axis=1)
        nth_vals = list(sorted_rows[:, -nth])
        nth_idxs = list(sorted_indices[:, -nth])
        return nth_idxs, nth_vals
    
    preds_df = preds_df.round(4)
    preds_arr = np.round(preds_df.to_numpy(),4)
    
    norm_crops = [str(crop) for crop in final_crops]
    max_idx, max_vals = get_nth_pred(preds_arr,1)
    sec_idx, sec_vals = get_nth_pred(preds_arr,2)
    third_idx, third_vals = get_nth_pred(preds_arr,3)
    max_names = [str(class_name_map[idx]) for idx in max_idx]
    sec_names = [str(class_name_map[idx]) for idx in sec_idx]
    third_names = [str(class_name_map[idx]) for idx in third_idx]
    date_times = [dt if dt is not None else None for dt in dt_list]
    results_headers = ['Date_Time', 'Prediction', 'Probability',  'Second_Pred', 'Second_Prob', 'Third_Pred',  'Third_Prob', 'Crop']
    results_lists = [date_times, max_names, max_vals, sec_names, sec_vals, third_names, third_vals, norm_crops]
    data = {header:value for header, value in zip(results_headers,results_lists)}
    results_df = pd.DataFrame(data).set_index(targs_df.index)

    #At this point we only have the things in the header list above

    bboxes = ['File_Path','x_min', 'y_min', 'Width', 'Height', 'Confidence']
    labels_df = bbox_df[bboxes].copy()

    labels_df = labels_df.set_index('File_Path').astype('float32')
    common_file_names = labels_df.index.intersection(targs_df.index)
    missing_files = labels_df.index.difference(targs_df.index).to_list()
    if missing_files:
        print('Missing_from_inference because file could not be opened:', missing_files)
    labels_df = labels_df.loc[common_file_names]
    labels_df = labels_df.set_index(targs_df.index)
    preds_df = preds_df.set_index(targs_df.index)

    combined_df = pd.concat([targs_df, results_df, labels_df, preds_df], axis=1)


    combined_df = relabel_empties(combined_df, 
                                  md_empty_threshold=md_empty_threshold, 
                                  preds_empty_threshold=empty_classify_threshold,
                                  hard_classes=hard_classes,
                                  empty_string=empty_string)

    combined_df = relabel_unknowns(combined_df, 
                                  md_empty_threshold=md_empty_threshold, 
                                  preds_empty_threshold=empty_classify_threshold,
                                  unknown_string=unknown_string)

    videos_df, results_df = separate_vid_img(combined_df)

    if not results_df.empty:
        if verbose:
            print('Collating predictions into encounters')
        results_df = collate_encounters(results_df, time_window=time_window)

    ## collate_encounters should be the place max_confidence gets added

    print('\n\n the results df after collate_encounters')
    print(list(results_df.columns)[:20])

    if not videos_df.empty:
        videos_df = collate_video(videos_df,
                                  threshold = md_empty_threshold,
                                  use_mean_scores=use_mean_video_preds) #Collates all the video frames from each video clip into a single 'Image' line

        print('\n\n the videos_df after collate_videos')
        print(list(videos_df.columns)[:20], '\n\n')

        if results_df.empty:
            results_df = pd.DataFrame(columns=videos_df.columns.tolist())
        results_df = pd.concat([videos_df, results_df])


    #No need for MD confidence here, as it already zeroed earlier probs
    empty_mask = (results_df['Max_Prob'] < empty_classify_threshold)
    results_df.loc[empty_mask, 'Encounter'] = empty_string

    #This isn't quite right.  We need the max confidence here, much like the max-prob
    unknown_mask = ((results_df['max_confidence']  > md_empty_threshold) 
                    & (results_df['Max_Prob'] < empty_classify_threshold))
    results_df.loc[unknown_mask, 'Encounter'] = 'Unknown'
    results_df = results_df.drop(columns='max_confidence')

    # Finally change the encounter names for situations where the class name got into it. 
    # Only needed for cases where all rows in the encounter were probability 0 because MD says nothing there.
    results_df['Encounter'] = results_df['Encounter'].replace(['below_md_threshold', 'no_md_box'], empty_string)

    if missing_files:
        results_df = append_missing_files(missing_files, results_df)

    cols_to_string = ['Date_Time', 'Targets', 'Encounter', 'Prediction', 'Second_Pred', 'Third_Pred', 'Crop']
    data_types = {column: 'string' for column in cols_to_string}
    results_df = results_df.round(3).astype(data_types)
    
    results_df.index.name = 'File_Path'  #Names the index
    results_df = results_df.reset_index(drop=False)  #Turns the index into a regular column

    return results_df


def rename_columns_by_convention(results_dict: Dict,
                                 mapping_csv_path: Path, 
                                 target_convention: str = 'lower_case_underscore'):
    """
    Rename columns in `df` using a mapping CSV and a target naming convention.
    If a column has no mapping, it is left unchanged.

    Parameters:
        df (pd.DataFrame): The DataFrame with class name columns.
        mapping_csv_path (str or Path): Path to the mapping CSV.
        target_convention (str): Desired naming scheme (e.g., 'scientific_name').

    Returns:
        results dict with new keys,  and a sorted list of the new names
    """
    df_preds = results_dict['pred_df']
    df_targs = results_dict['targ_df']

    if mapping_csv_path.exists():
        mapping_df = pd.read_csv(mapping_csv_path)
    else:
        print(Warn.S + 'Warning: No name-mapping csv file found at {str(mapping_csv_path)}')
        return results_dict, list(df_preds.columns)

    if (target_convention == 'lower_case_underscore') or (target_convention not in mapping_df.columns):
        return results_dict, list(df_preds.columns)

    original_name_col = mapping_df.columns[0]
    mapping_df = mapping_df.dropna(subset=[original_name_col, target_convention])
    mapping_dict = dict(zip(mapping_df[original_name_col], mapping_df[target_convention]))

    # Rename columns where a mapping exists; leave others unchanged
    def rename_columns(df, mapping):
        return df.rename(columns=lambda col: mapping.get(col, col))

    results_dict['pred_df'] = rename_columns(df_preds, mapping_dict)
    results_dict['targ_df'] = rename_columns(df_targs, mapping_dict)
    new_names = [mapping_dict.get(col, col) for col in df_preds.columns]

    return results_dict, new_names


def make_jsons(df: pd.DataFrame,
               paths: Paths,
               md_json_pth: str,
               class_list: List[str],
               special_classes: List[str],
               verbose: bool = False):
    '''
    Create json files in the format expected by TimeLapse software https://timelapse.ucalgary.ca/,
    based on the original detection file from MegaDetector
    '''

    for special in special_classes:
        if special in df.columns:
            print(f'Saving an individual .csv and .json file for the class {special}')
            special_cols = ['File_Path', 'Date_Time', special]
            df_special = df[special_cols].copy()

            try:
                df_for_json = df_special.reset_index()
                destn_path = paths.preds_dir / f'{special}_only.json'
                predictions_2_json.main(df_for_json,
                                        paths.image_dir,
                                        md_json_pth,
                                        classes=[special],
                                        destn_path=destn_path,
                                        verbose=verbose)
            except Exception as e:
                print(Warn.S +  f'An error occurred with Predictions_2_JSON:'  + Warn.E, e)
                print('The predictions csv should have been produced anyway, but if you need the json file output', 
                    ' as well, try checking for corrupted file or video and remove those first, then re-run inference')
                    
            


    try:
        df_for_json = df.reset_index()
        predictions_2_json.main(df_for_json,
                                paths.image_dir,
                                md_json_pth,
                                classes=class_list,
                                destn_path= paths.preds_dir / 'alita_predictions.json',
                                verbose=verbose)
    except Exception as e:
        print(Warn.S +  f'An error occurred with Predictions_2_JSON:'  + Warn.E, e)
        print('The predictions csv should have been produced anyway, but if you need the json file output', 
            ' as well, try checking for corrupted file or video and remove those first, then re-run inference')
    return

def make_special_csvs(df: pd.DataFrame,
                      paths: Paths,
                      special_classes: List[str],
                      threshold: float = 0.5):
    '''
    Create json files in the format expected by TimeLapse software https://timelapse.ucalgary.ca/,
    based on the original detection file from MegaDetector
    '''

    for special in special_classes:
        if special in df.columns:
            special_cols = ['File_Path', 'Date_Time', special]
            df_special = df[special_cols].copy()
            df_special = df_special[df_special[special] >= threshold]
            df_special.to_csv(paths.preds_dir / f'{special}.csv', encoding='utf-8')

    return



# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------

def predict_images(project_dir: Path,
                   image_dir: str,
                   predictions_dir: str,
                   settings_pth: str,
                   weights_pth: str,
                   detector_weights_pth: str,
                   classify_conf_threshold: float = 0.5,
                   md_empty_threshold: float = 0.15,
                   naming_scheme: str = 'lower_case_underscore',
                   cpu_only: bool = False,
                   num_workers: int = 0,
                   special_interest_classes: List=[str],
                   use_mean_video_preds: bool = False,
                   verbose=True):
    warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt JPEG data")
    
    infer_cfg, image_cfg = get_settings(settings_pth, num_workers=num_workers)
    
    paths = Paths(project_dir=project_dir,
                  exp_name=infer_cfg.EXPERIMENT_NAME,
                  run_id=infer_cfg.RUN_ID,
                  image_dir=image_dir,
                  preds_dir = predictions_dir,
                  weights_pth = weights_pth,
                  detector_weights_pth=detector_weights_pth,
                  )
    
    if check_for_empty(paths.image_dir):
        return pd.DataFrame(), 0

    device, gpu = 'cpu', False if cpu_only else test_cuda(mem_threshold_gb=1.8)
    
    #if __name__ == "__main__":
    process_video.main(root_dir_pth=paths.image_dir, time_interval=0.5, verbose=verbose)
      
    species_list = data_from_json(paths.class_names_pth)

    md_output_pth = run_megadetector(paths.image_dir, 
                                     detector_weights_pth=str(paths.detector_weights),
                                     destn_fldr=paths.preds_dir,
                                     detector_name = 'md_v5',
                                     detection_threshold = infer_cfg.MD_DETECTION_THRESHOLD)

    bbox_df = get_dataframe(paths.image_dir,
                            [md_output_pth],
                            species_list,
                            resize_method = 'md_crop',
                            class_name_change = infer_cfg.CLASS_NAME_CHANGE,
                            class_joins = infer_cfg.CLASS_JOINS,
                            verbose=verbose)

    infer_transforms = ImagePrep(image_cfg.INPUT_MEAN,
                                 image_cfg.INPUT_STD,
                                 image_size=image_cfg.IMAGE_SIZE).transform
    
    dataset = PredatorDataset(bbox_df, 
                              infer_transforms, 
                              image_cfg,
                              dt_formats=infer_cfg.EXIF_DT_FORMATS)
    
    test_loader = DataLoader(dataset,
                             batch_size=infer_cfg.BATCH_SIZE,
                             collate_fn=custom_collate,
                             shuffle=False,
                             num_workers=infer_cfg.num_workers)
      
    model = get_model(paths.weights_pth, len(species_list), infer_cfg.MODEL_NAME, head_name=infer_cfg.HEAD_NAME)
    
    if verbose:
        print('The infer settings')
        for key, value in infer_cfg.__dict__.items():
            print(f"{key}: {value}")
        print(Colour.S + 'The following list contains all the possible output classes from the classifier' + Colour.E)
        print(species_list)
        total_images = len([f.name for f in Path(paths.image_dir).rglob('*.jpg')])
        print('Image Folder: {}, ({} images)'.format(paths.image_dir, total_images))
        print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if gpu else 'CPU'))
        print(f'The classifier empty threshold is {classify_conf_threshold}')

    results = infer_dataset(test_loader, 
                            model, 
                            species_list, 
                            device, 
                            activation=infer_cfg.ACTIVATION,
                            verbose=verbose)

    results, new_names = rename_columns_by_convention(results,
                                                      paths.naming_schemes_pth,  #Hand-edited csv with naming conversions
                                                      naming_scheme) # 'Scientific'

    results_df = make_results_table(results,
                                    bbox_df,
                                    new_names,
                                    infer_cfg.ENCOUNTER_WINDOW,
                                    md_empty_threshold=md_empty_threshold,
                                    empty_classify_threshold=classify_conf_threshold,
                                    hard_classes=infer_cfg.HARD_CLASSES,
                                    use_mean_video_preds=use_mean_video_preds,
                                    verbose=verbose)

    results_df.to_csv(paths.out_detailed_csv_path,  encoding='utf-8')

    print(f'New names: {new_names}')
    print(f'Special interest classes: {special_interest_classes}')

    make_jsons(results_df,
               paths,
               md_output_pth,
               new_names,
               special_interest_classes,
               verbose=verbose)
    
    make_special_csvs(results_df,
                      paths,
                      special_interest_classes,
                      threshold=classify_conf_threshold)

    cols_to_keep = ['File_Path', 'Date_Time', 'Encounter', 'Max_Prob']
    results_df[cols_to_keep].to_csv(paths.out_csv_path,  encoding='utf-8')

    if verbose:
        print('The dataframe saved to CSV')
        print(results_df[cols_to_keep].head())
        print(Colour.S + 'Process Complete'+ Colour.E)
        print(f'The predictions csv file was saved to: {paths.out_csv_path}')

    if gpu:
        gc.collect()
        torch.cuda.empty_cache()

    results_df.set_index('File_Path', inplace=True)

    return results_df, results['speed']

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    print(f'the project directory is {project_dir}')
    #For Testing
    #images =   str(project_dir / 'data/Cats')
    #images = "C:/Users/ollyp/OneDrive/Desktop/empty_folder"
    images = str(project_dir / "data/vids_small")
    results_path = "C:/Users/ollyp/OneDrive/Desktop"
    #results_path =  "/home/olly/Desktop"  
    settings = str(project_dir / 'models/Exp_46/Exp_46_Run_21.yaml')
    classify_weights = str(project_dir / 'models/Exp_46/Exp_46_Run_21_best_weights.pt')
    detector_weights = str(project_dir / 'models/md_v5a.0.0.pt')
    md_empty_threshold = 0.15
    classify_threshold = 0.5
    special_animals = ['Kiwi', 'Cat']
    #The detection threshold for bounding box generation is 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", type=str, default=images, help="Filepath to the root directory of imagery to be processed")
    parser.add_argument("--resultsPath", type=str, default=results_path, help="Filepath to the folder that will hold the predictions")
    parser.add_argument("--emptyDetectThreshold", type=float, default=md_empty_threshold, help="Threshold for classifying as empty based on MD confidence")
    parser.add_argument("--classifyThreshold", type=str, default=classify_threshold, help="Filepath to the root directory of imagery to be processed")
    
    args = parser.parse_args()
    images =  args.dataPath
    results_path =args.resultsPath
    md_empty_threshold =args.emptyDetectThreshold
    classify_empty_threshold = args.classifyThreshold

    output, speed = predict_images(project_dir = project_dir,
                                   image_dir=images,
                                   predictions_dir=results_path,
                                   settings_pth=settings,
                                   weights_pth=classify_weights,
                                   detector_weights_pth=detector_weights,
                                   md_empty_threshold=md_empty_threshold,
                                   classify_conf_threshold=classify_threshold,
                                   naming_scheme='Common',
                                   cpu_only = False,
                                   num_workers = 2,
                                   special_interest_classes = special_animals,
                                   use_mean_video_preds=False,
                                   verbose=True,
                                   )