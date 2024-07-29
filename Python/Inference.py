import torch
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
import gc
import json
import subprocess
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
import Process_Video
import piexif
import yaml
import random
import argparse
import warnings
from datetime import datetime
from datetime import timedelta
from Interpret_JSON import process_all_jsons
from Check_Weights import check_weights
import Predictions_2_JSON

class DefaultConfig:
    def __init__(self):
        cpu_cores = os.cpu_count()
        self.NUM_WORKERS = max(cpu_cores // 2, 0)  #Only use lots of cores on grunty machines
        if self.NUM_WORKERS <= 4:
            print(Colour.S + f'Grrr, using only {self.NUM_WORKERS} of the {self.NUM_WORKERS} total threads to reduce the risk of crushing your puny earth operating system' + Colour.E)
        else:
            print(Colour.S + f'There are {cpu_cores} logical threads avalaible, using {self.NUM_WORKERS}' + Colour.E)
        self.EXPERIMENT_NAME = 'Exp_38'
        self.RUN_ID = 'Run_01'
        self.CLASS_JOINS = {'lizard':['skink', 'lizard'], 'finch':['greenfinch', 'goldfinch', 'chaffinch'], 'quail':['quail_california', 'quail_brown'], 'deer':['deer', 'white_tailed_deer']}
        self.CLASS_NAME_CHANGE = {'penguin':'little_blue_penguin', 'song thrush':'thrush', 'NZ_falcon':'nz_falcon'}
        self.IMAGE_FOLDER_PTH = '' #Full file path to point to a different location 
        self.BATCH_SIZE = 4 if self.NUM_WORKERS == 0 else 8 #Giving it the best chance of working on puny machines
        self.OUT_FOLDER_NM = 'Inference_Results'
        self.REMOVE_BACKGROUND = True
        self.MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'
        self.HEAD_NAME = 'ClassifierHead' # Alternative: BasicHead
        self.RESIZE_METHOD = 'md_crop' #rescale' # alternatively 'md_crop', should match method used for training
        self.MD_RESAMPLE = True #Should match the value used for training if using md_crop.  True downscales large md crop boxes to the image size
        self.IMAGE_SIZE = 480 #Should match the size the transformed crops were during training, and be no larger than the stored crop size
        self.CROP_SIZE = 600 #This is the size that the images will be cropped to as part of the localisation step
        self.INPUT_MEAN = [ 0.485, 0.456, 0.406 ] # mean to be used for normalisation, using values from ImageNet.
        self.INPUT_STD = [ 0.229, 0.224, 0.225 ] # stddev to be used for normalisation, using values from ImageNet.
        self.BEST_ONLY = True # Only use the best crop from MegaDetector
        self.BUFFER = 0
        self.EMPTY_THRESHOLD = .5 #This needs to be pretty low, otherwise if there are a bunch of similar animals, you will incorrectly get asigned 'empty'
        self.MD_EMPTY_THRESHOLD = 0.1 #If the md score is below this the image will automatically be classed as 'empty' regardless of classifier scores
        self.RECLASSIFY_MD_EMPTIES = True #If this is true, The classifier is run, even though the prediction will be empty, the probability scores may be of interest.
        self.ENCOUNTER_WINDOW = 30 # Seconds to collate the scores for an encounter
        self.EXIF_DT_FORMATS = ['%Y:%m:%d %H:%M:%S', '%y:%m:%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']
        self.SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
        self.DEBUG = False
        self.EDGE_FADE = False
        self.MIN_FADE_MARGIN = 0.0
        self.MAX_FADE_MARGIN = 0.0
        self.OUT_FIELDS = ['Target', 'Prediction', 'Probability', 'Second_Pred', 'Second_Prob', 'Third_Pred', 'Third_Prob']
        self.DATA_FOLDER_NM = 'Data'
        self.SETUP_FOLDER_NM = 'Setup'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.RUNS_FOLDER_NM = 'Runs'
        self.SETTINGS_FOLDER_NM = 'Settings'
        self.RUN_MD_PS_NM = 'MegaDetector_Run.ps1'
        self.RESULTS_FOLDER_NM = 'Results'
        self.MODELS_FOLDER_NM = 'Models'
        self.CLASS_NAMES = '_class_names.json'
        self.DEFAULT_IMAGE_FOLDER_NM = 'Independent_Images' # '' #'elaine_subset' #''  #'corrupted_copy' 'Independent_Images'  'vids_images_testing' 'irish_images'
        self.WEIGHTS_FOLDER_SUFFIX = '_weights'
        self.WEIGHTS_FN_SUFFIX = '_best_weights.pt'
        self.PREDS_CSV_SUFFIX_OUT = '_predictions.csv'
        self.MD_WEIGHTS_NM = 'md_v5a.0.0.pt'


def get_config(settings_pth=None):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list =  ['DEBUG', 'REMOVE_BACKGROUND', 'EDGE_FADE', 'CLASS_NAME_CHANGE', 'CLASS_JOINS', 'EXIF_DT_FORMATS', 'RECLASSIFY_MD_EMPTIES', 'RERUN_MD_ON_ALL']
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

# --------------------------- Functions & Classes-----------------------------------------
# ----------------------------------------------------------------------------------------
class Warn:  #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Colour: 
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def check_gpu_memory(threshold_gb=2):
    use_gpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = (reserved_memory - allocated_memory) / 1e9
        use_gpu = free_memory >= threshold_gb
        if use_gpu:
            print(Colour.S + f'Using GPU, with {free_memory}GB memory available' + Colour.E)
        else:
            print(Colour.S + f'Only {free_memory} GB GPU memory is avaialble for the classifier, this insufficient. ' 
                  'Will run (slowly) on CPU instead. I suggest upgrading to an NVIDIA GPU with min 4GB to run faster' + Colour.E)
    return  device, use_gpu


def test_cuda():
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    if gpu:
        gc.collect()
        torch.cuda.empty_cache()
        device, gpu = check_gpu_memory()
    return device, gpu


def check_for_empty(image_dir):
    extensions = {'.jpg', '.jpeg', '.mp4', '.avi', '.mov'}
    print("Checking for suitable media files (.jpg, .jpeg, .mp4, .avi & .mov)")
    media_files = {file_path for file_path in tqdm(image_dir.rglob('*')) if file_path.is_file() and file_path.suffix.lower() in extensions}
    if media_files:
        print(Colour.S + f'{len(media_files)} media files found for classification' + Colour.E)
    else:
        print(Warn.S + 'No media files found in your chosen folder, terminating the process')
        sys.exit()
    

def df_from_json(json_pths, species_list, image_folder):
    if json_pths:
        df = process_all_jsons(json_pths, species_list, image_folder=image_folder)
        df['Targets'] = df['Species']
    else:
        print(Warn.S + "\nIt appears you are trying to use the MegaDetector for object localisation, but it has not prduced any bounding boxes." + Warn.E)
        print(Colour.S + 'Proceeding without localisation data, but this is expected to hurt performance' + Colour.E)
        df = make_dataframe(image_folder, species_list)

    return df


def make_dataframe(img_dir, class_list):
    def check_cls(class_nm, class_list):
        return 'unknown' if class_nm not in class_list else class_nm
    file_names = [str(f) for f in Path(img_dir).rglob('*.jpg')]
    parent_names = [str(Path(fn).parent.name) for fn in file_names]
    class_names = [check_cls(nm, class_list) for nm in parent_names]
    xmins = ymins = [0] * len(class_names)
    widths = heights = [1] * len(class_names)
    confidences = [1] * len(class_names)
    df = pd.DataFrame({'File_Path': file_names, 'Targets': class_names, 'Confidence': confidences, 'x_min': xmins, 
                       'y_min': ymins, 'Width': widths, 'Height': heights, })
    return df


def split_md_empties(df, threshold):
    above_threshold_df = df[df['Confidence'] > threshold].copy()
    below_threshold_df = df[df['Confidence'] <=  threshold].copy()
    return above_threshold_df, below_threshold_df


def update_target_names(df, cfg):
    """Updates the target names as per the settings file, so that the target names match the
    names used for training"""
    for key, value in cfg.CLASS_NAME_CHANGE.items():
        df['Targets'].replace(key, value, inplace=True)
    new_names = {item: key for key, items in cfg.CLASS_JOINS.items() for item in items}
    for key, value in new_names.items():
        df['Targets'].replace(key, value, inplace=True)

    if len(list(df['Targets'].unique())):
        print(Colour.S + 'The following list contains all the unique folder labels in your file structure that match class names' + Colour.E)
        print(df['Targets'].unique())
    return df


def find_dataframe(image_dir, species_list, run_md_ps, cfg):
    json_pths = [f for f in Path(image_dir).glob('*.json') if f.name == 'mdPredictions.json']
    if json_pths:
        print(Colour.S + f'{len(json_pths)} MegaDetector output json files were found ')
    else:
        print('No existing .json files were found from the MegaDetector output, will run the MegaDetector to generate')
    setattr(cfg, 'SOURCE_IMAGES_PTH', str(image_dir)) # So that the path stem doesn't get replaced in df_from_json()
    if cfg.RESIZE_METHOD == 'md_crop' and not json_pths:
        #Call up MegaDetector_Run.ps1 with image_dir as filepath input
        print(f'Running the MegaDetector on {image_dir}')
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(run_md_ps), str(image_dir)]  #this threw an error
        #cmd = ["powershell.exe", "-ExecutionPolicy", "Unrestricted", "-File", str(run_md_ps), str(image_dir)]  #not tried but should work the same
        cmd = ["powershell.exe", "-File", str(run_md_ps), str(image_dir)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # try adding "python", "-Xfrozen_modules=off"
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
        process.wait()
        
        json_pths = [f for f in Path(image_dir).rglob('*.json') if f.name != 'alita_predictions.json']
        #If something goes wrong with the MD here, json paths will be an empty list
        df = df_from_json(json_pths, species_list, str(image_dir)) #need to make sure the right source directory is in the cfg
    elif cfg.RESIZE_METHOD == 'md_crop' and json_pths:
        df = df_from_json(json_pths, species_list, str(image_dir)) 
        
    else:  # make a dataframe without detected bounding boxes (set them to the image boundary)
        df = make_dataframe(image_dir, species_list)
    if cfg.DEBUG:
        df = df.iloc[::20]
        print("This is the imput dataframe")
    
    df = update_target_names(df, cfg)
    
    empties_df = None
    if not cfg.RECLASSIFY_MD_EMPTIES:
        df, empties_df = split_md_empties(df, cfg.MD_EMPTY_THRESHOLD)
    
    return df, empties_df, json_pths[0]


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


class PredatorDataset(Dataset):
    def __init__(self, 
                 labels_df, 
                 transform,  
                 crop_size=600, 
                 buffer=0,
                 resize_method='rescale', 
                 remove_background=False, 
                 fade_edges=False,
                 min_margin=0.05,
                 downsample=False,
                 dt_formats=['%Y:%m:%d %H:%M:%S']):
        self.df = labels_df
        self.transform = transform
        self.crop_size = crop_size
        self.resize_method = resize_method
        self.remove_background = remove_background
        self.fade_edges = fade_edges
        self.min_margin = min_margin
        self.downsample = downsample # Downsample image if the MD crop box size is > crop_size
        self.buffer = buffer # The fraction of the image w/h that is kept around the md bounding box
        self.counter = 0 
        self.dt_formats = dt_formats

    def __len__(self):
        return len(self.df)

    def load_image(self, image_path, mode):
        try:
            with open(image_path, 'rb') as in_file:
                jpeg_buf = in_file.read()    
                date_time = get_exif_dt(jpeg_buf, self.dt_formats)
                image = cv2.imread(image_path)
                if image is not None:
                    if mode == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, date_time
        except:
            print(f"Warning: Unable to load the image at '{image_path}'. Skipping...")
            return None, None
    
    def edge_fade(self, image, min_margin=0.05, max_margin=0.05):
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

    def subtract_background(self, image, row, buffer):
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

    def crop_to_square(self, image):
        height, width = image.shape[:2]
        min_side_length = min(height, width)
        top = (height - min_side_length) // 2
        bottom = top + min_side_length
        left = (width - min_side_length) // 2
        right = left + min_side_length
        return image[top:bottom, left:right]

    def pad_to_square(self, image):
        height, width, channels = image.shape
        dtype = image.dtype
        max_dim = max(height, width, self.crop_size)
        square_image = np.zeros((max_dim, max_dim, channels), dtype=dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_image[y_offset:y_offset+height, x_offset:x_offset+width]= image
        return square_image, x_offset, y_offset

    def rescale_image(self, image_arr):
        size=self.crop_size
        crop = self.crop_to_square(image_arr)
        return cv2.resize(crop, (size, size))
    
    def get_mega_crop_values(self, row, img_w, img_h, final_size):
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
    def get_new_scale(self, row, buffer, width, height, final_size):
        #figures out how much to scale down the new image to, so the max(bounding-box) + buffer = the desired crop size
        #only effects images where the crop box would be greater than the crop size
        clamp = lambda n: max(min(1, n), 0)
        x_min = clamp(row['x_min'] - buffer)
        y_min = clamp(row['y_min'] - buffer)
        x_max = clamp(row['x_min'] + row['Width'] + buffer)
        y_max = clamp(row['y_min'] + row['Height'] + buffer)
        max_dimension = max([(x_max - x_min)*width, (y_max - y_min)*height]) 
        return final_size/max_dimension if max_dimension > final_size else None

    def md_crop_image(self, row, image_arr):
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
        self.counter +=1

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
            augmented = self.transform(image=image)
            image = augmented['image']

        """not very happy about the ltrb_norm datatype.  
        if it is a list of floats, Pytorch turns it into a tensor of length batch_size with one value per item, not sure why
        With hind-sight I've done this dataset class rather unsually.  I'm abusing the dataloader a bit here, it's not really
        intended for such complex calculations,  and passing so many variables for non-pytorch purposes.  But this way I only 
        need to open each image once, so it's the most efficient method.  Revisit this question later."""  
        
        return image, target, f_path, str(ltrb_norm), date_time


class ClassifierHead(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.2):
        super().__init__()
        self.Linear = nn.Linear(num_features, num_features//2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(num_features//2, num_classes)
        
    def forward(self, x):
        x = self.Linear(x)
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

def get_custom_head(head_name):
    if head_name == 'ClassifierHead':
        return ClassifierHead
    else:
        return BasicHead


class CustomModel(pl.LightningModule):
    def __init__(self, 
                 class_list,
                 model_name='efficientnetv2_l_21k',
                 custom_head=ClassifierHead
                 ):
        super().__init__()
        
        self.num_classes = len(class_list)
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.in_features = self.backbone.classifier.in_features
        print(f'There are {self.in_features} input features to the classifier head and {self.num_classes} outputs')
        self.backbone.classifier = custom_head(self.in_features, self.num_classes)

    def forward(self,images):
        logits = self.backbone(images)
        return logits


def get_model(weights, species, model_name='efficientnetv2_l_21k', head_name='ClassifierHead'):
    model_state_dict = torch.load(weights)
    custom_head=get_custom_head(head_name)
    model = CustomModel(species, model_name, custom_head)
    model.load_state_dict(model_state_dict)
    return model


def get_transforms(cfg):
    transforms = A.Compose([
                #A.CenterCrop(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE, p=1),
                A.Normalize(mean=cfg.INPUT_MEAN, std=cfg.INPUT_STD), ToTensorV2()])
    return transforms


def infer_dataset(loader, model, class_list, device):
    num_samples = len(loader.dataset)
    print(Colour.S + f'There are {num_samples} images in the dataloader for inference.' + Colour.E)
    
    if __name__ == '__main__':
        pbar = tqdm(total=num_samples)
    model.eval()
    model.to(device)
    targets_list, predictions_list, paths_list, crops_list, dt_list = [], [], [], [], []
    start_time = time.time()
    try:
        for batch in loader:
            if batch is None:
                print(f"Skipping batch due to None")
                continue

            images, targets, paths, ltrb_dims, d_times = batch

            if images is None or targets is None or paths is None or ltrb_dims is None or d_times is None:
                print(f"Skipping batch due to None elements: {batch}")
                continue

            images = images.to(device)
            with torch.no_grad():
                logits = model(images)
            if __name__ == '__main__':
                pbar.update(len(images))
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
        
    if __name__ == '__main__':
        pbar.close()
    total_time = time.time()-start_time
    preds_array = np.vstack(predictions_list)
    speed = num_samples/total_time
    print(f'The classifier processed {num_samples} samples in {total_time:.2f} seconds')
    print(f'That is a mean of {speed:.2f} images per second')

    pred_df = pd.DataFrame(preds_array, index=paths_list, columns=class_list)
    targ_df= pd.DataFrame(targets_list, index=paths_list, columns=['Targets'])
    del model
    torch.cuda.empty_cache()
    return targ_df, pred_df, crops_list, dt_list, speed


def custom_collate(batch):
    """Determines how to collate the items from the dataset __getitem__, and handle None values"""
    batch = [sample for sample in batch if sample[0] is not None]
    if not batch:
        return None
    images, targets, paths, ltrb_dims, d_times = zip(*batch)
    images = torch.stack(images)
    return images, targets, paths, ltrb_dims, d_times


def separate_vid_img(df):
    """Splits off the video images into a different dataframe for later collation"""
    vid_df = df[df.index.str.contains('video_')]
    img_df = df[~df.index.str.contains('video_')]
    return vid_df, img_df


def collate_encounters(df, time_window=30):
    """Groups the dataframe by the parent folder, then runs the collate_encounters_per_folder function
    on each seperately.  Speeds up the sorting, and also prevents confusion between cameras that happen
    to be set off in different locations at the same time."""
    df = df.copy()
    df['Parent_Folder'] = df.index.map(lambda x: str(Path(x).parent))
    grouped_dfs = []
    for _, group_df in tqdm(df.groupby('Parent_Folder')):
        grouped_df_result = collate_encounters_per_folder(group_df, time_window)
        grouped_dfs.append(grouped_df_result)
    result_df = pd.concat(grouped_dfs)
    result_df.drop('Parent_Folder', axis=1, inplace=True)
    return result_df


def collate_encounters_per_folder(df, time_window=30):
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
    # Set 'Encounter' column equal to 'Prediction' for rows with empty 'Date_Time'
    df.loc[df['Date_Time'].isna(), 'Encounter'] = df.loc[df['Date_Time'].isna(), 'Prediction']
    df.loc[df['Date_Time'].isna(), 'Max_Prob'] = df.loc[df['Date_Time'].isna(), 'Probability']
    df.set_index('Image_File_Paths', inplace=True, drop=True)  #Puts the index back as it was before
    df = df.drop(['dt_object', 'time_difference', 'File_Path'], axis=1)
    df['Encounter_Start'] = df['Encounter_Start'].dt.time
    first_cols = ['Date_Time', 'Encounter_Start', 'Encounter', 'Max_Prob']  #'Image_File_Paths', 
    new_columns_order = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[new_columns_order]

    
    return df


def agg_rows_by_mean(df):
    """This is for video, to take the mean of the probabilities by parent directory 
    (which corresponds to a single video file).  Then reduce the group to a single
    line where the max score, and the predicted class come from those agregated probs
    There is duplication here, it would be better done earlier
    """
    other_cols = ['parent_dir', 'File_Path', 'Date_Time', 'Probability', 'Second_Prob', 'vid_path',
                'Third_Prob', 'Prediction', 'Second_Pred', 'Third_Pred', 'Confidence',
                'Targets', 'x_min', 'y_min', 'Width', 'Height', 'Crop']
    prob_scores = [col for col in df.columns if col not in other_cols]
    agg_dict = {col: 'mean' for col in prob_scores}
    agg_dict.update({col: 'first' for col in other_cols})
    agg_dict.update({'Confidence':'max' })
    agg_df = df.groupby('parent_dir').agg(agg_dict)
    agg_df.reset_index(drop=True, inplace=True)
    agg_df['Probability'] = agg_df[prob_scores].max(axis=1)
    agg_df['Prediction'] = agg_df[prob_scores].idxmax(axis=1)
    agg_df['Second_Prob'] =  agg_df[prob_scores].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    agg_df['Second_Pred'] =  agg_df[prob_scores].apply(lambda row: row.nlargest(2).index[-1], axis=1)
    agg_df['Third_Prob'] =  agg_df[prob_scores].apply(lambda row: row.nlargest(3).iloc[-1], axis=1)
    agg_df['Third_Pred'] =  agg_df[prob_scores].apply(lambda row: row.nlargest(3).index[-1], axis=1)
    return agg_df


def collate_video(df, threshold, use_mean_scores=True):
    """Collects all the image results from the same video, and makes a prediction based on the highest probability score,
    This is then applied to the video file as a whole, and the filepath is put back to  the original video filepath"""
    
    df = df.copy()
    df['path_string'] = [Path(index_value).stem for index_value in df.index]  #a string like E-_dir1-_dir2-_dir3-_filename_MP4_0008
    df['parent_dir'] = [Path(index_value).parent for index_value in df.index] #parent folder
    df['grandparent'] = [Path(index_value).parent.parent for index_value in df.index]  #grandparent folder
    grandparents = df['grandparent'].unique().tolist()
    df['path_string'] = df['path_string'].apply(lambda x: ':\\'.join(x.split('-_', 1)))  # put back the drive colon
    df['path_string'] = df['path_string'].apply(lambda x: x.rsplit('_', 1)[0])  # remove the last integer block
    df['path_string'] = df['path_string'].apply(lambda x: x.replace('-_', "\\")) # put back the path seperators
    df['vid_path'] = df['path_string'].apply(lambda x: '.'.join(x.rsplit('_', 1))) #put back the final extension
    df = df.drop(['path_string', 'grandparent'], axis=1)
    print(df.head(10))
    if use_mean_scores:
        df = agg_rows_by_mean(df)  #gets the average scores, then recalculates the first, second, third probability
    idxmax_rows = df.groupby('parent_dir')['Probability'].idxmax()  #simly finds the row that has the maximum first probability
    max_df = df.loc[idxmax_rows].copy()  #Simple max scoring row for each video

    #now do the same but only with more confident images above some MegaDetector Threshold
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
                'x_min', 'y_min', 'Width', 'Height', 'Confidence']

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


def relabel_empties(df, empties_df, threshold):
    """This function deals with what to do with images that the MegaDetector gives a low confidence score, suggesting
    there are probably no animals present.  Either they can be classified anyway, then use the classifer threshold (a 
    little risky, since the classfier probabilities always sum to 1, but OK if there are a lot of classes)
    3 kinds of empty:   
    1. If all probabilities in the encounter are < the EMPTY_THRESHOLD & MD Confidence > MD_EMPTY_THRESHOLD  
        -->   Encounter value is changed to 'empty' (outside this function)
    2. The MD confidence score is < MD_EMPTY_THRESHOLD   --> MD Classifier is run, encounter-'empty', Prediction-is as scored
                                                    OR (RECLASIFY_EMPTIES=False)   encounter-'empty', Prediction-'below_md_threshold',
    If MD confidence > MD_EMPTY_THRESHOLD & Probability < EMPTY_THRESHOLD   --> Prediction is set to 'empty'
    """
    
    df['File_Path'] = df.index
    
    def get_timestamps(path_list):
        stamps_list = []
        for image_path in tqdm(path_list):
            try:
                with open(image_path, 'rb') as in_file:
                    jpeg_buf = in_file.read()    
                    date_time = get_exif_dt(jpeg_buf)
                stamps_list.append(date_time)
            except Exception as e:
                stamps_list.append(None)
        return stamps_list

    if empties_df is None:
        #This is the case when empties were processed anyway, so we still need to reset the predictions
        rows_to_modify = (df['Confidence'] < threshold)  # Tried adding & (df['Probability'] < threshold) but it doesn't help 
        cols_to_zero = ['Probability', 'Second_Prob', 'Third_Prob']
        cols_to_empty = ['Prediction', 'Second_Pred', 'Third_Pred']
        df.loc[rows_to_modify, cols_to_zero] = 0
        df.loc[rows_to_modify, cols_to_empty] = 'empty'
    else:
        #This is the case when empties were split into another dataframe of their own and not processed by the classifier.
        cols_to_copy = ['File_Path','Targets','Confidence', 'x_min', 'y_min', 'Width', 'Height']
        new_cols = list(set(df.columns) - set(cols_to_copy))
        empties_df = empties_df[cols_to_copy].copy()
        print(empties_df[empties_df.index.duplicated()])
        empty_df = pd.DataFrame(index=empties_df.index, columns=new_cols).fillna(0)
        empty_df['Crop'] = '[0, 0, 1, 1]'
        empties_df = pd.concat([empties_df, empty_df], axis=1) #Concat the new columns
        empty_files = empties_df['File_Path'].tolist()
        empties_df['Date_Time'] = get_timestamps(empty_files)
        empties_df['Prediction'] = 'below_md_threshold'
        empties_df.loc[empties_df['Confidence'] == -1, 'Prediction'] = 'no_md_box'
        df = pd.concat([df, empties_df], ignore_index=True)
        df['DT_Object'] = pd.to_datetime(df['Date_Time'],  errors='coerce', dayfirst=True)  # This is throwing an error sometimes
        df = df.sort_values(by='DT_Object')
        df = df.drop('DT_Object', axis=1, inplace=False)
        df = df.set_index('File_Path', inplace=False, drop=True)
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


def make_output_table(targs_df, 
                      preds_df, 
                      labels_df, 
                      final_crops, 
                      dt_list, 
                      class_name_map, 
                      empties_df, 
                      time_window, 
                      empty_threshold=0, 
                      md_empty_threshold=0.5):
    """Join all the predictions and targets, and assemble into the final table for saving and analysis
    This function could be improved by collating video near the start, prior to subsequent processing"""
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
    labels_df = labels_df.set_index('File_Path').astype('float32')
    common_file_names = labels_df.index.intersection(targs_df.index)
    #print('Targ_filenames', targs_df.index)
    #print('label_filenames',labels_df.index)   # The problem is here, the troublemaker file isn't in this
    missing_files = labels_df.index.difference(targs_df.index).to_list()
    if missing_files:
        print('Missing_from_inference because file could not be opened:', missing_files)
    labels_df = labels_df.loc[common_file_names]
    labels_df = labels_df.set_index(targs_df.index)
    preds_df = preds_df.set_index(targs_df.index)
    combined_df = pd.concat([targs_df, results_df, labels_df, preds_df], axis=1)
    print('Relabeling  images with MegaDetector confidence below the threshold to empty')
    combined_df = relabel_empties(combined_df, empties_df, md_empty_threshold)
    videos_df, results_df = separate_vid_img(combined_df)
    
    #print(Colour.S + 'The Videos_df head' + Colour.E)
    #print(videos_df.head())
    
    #print(Colour.S + 'Images df head' + Colour.E)
    #print(results_df.head())
    
    #At this point neither have 'image_file_paths' column header

    if not results_df.empty:
        print('Collating predictions into encounters')
        results_df = collate_encounters(results_df, time_window=time_window)

    if not videos_df.empty:
        videos_df = collate_video(videos_df, 0.7) #Collates all the video frames from each video clip into a single 'Image' line
        if results_df.empty:
            results_df = pd.DataFrame(columns=videos_df.columns.tolist())
        results_df = pd.concat([videos_df, results_df])

    #If all the classifier scores are really low, or MD less than the threshold change the encounter to 'empty'
    #So if the MD score is lower than the threshold, it will over-ride anything from the other images in the encounter
    mask = (results_df['Confidence']  < md_empty_threshold) & (results_df['Max_Prob'] < empty_threshold)  #This doesn't help. Instead replace with an empty class & use an OR > .
    results_df.loc[mask, 'Encounter'] = 'empty'
    #finally change the encounter names for situations where the class name got into it. 
    # Only needed for cases where all rows in the encounter were probability 0 because MD says nothing there.
    results_df['Encounter'] = results_df['Encounter'].replace(['below_md_threshold', 'no_md_box'], 'empty')

    if missing_files:
        results_df = append_missing_files(missing_files, results_df)

    cols_to_string = ['Date_Time', 'Targets', 'Encounter', 'Prediction', 'Second_Pred', 'Third_Pred', 'Crop']
    data_types = {column: 'string' for column in cols_to_string}
    results_df = results_df.round(3).astype(data_types)
    
    results_df.index.name = 'File_Path'  #Names the index
    results_df = results_df.reset_index(drop=False)  #Turns the index into a regular column
    return results_df

# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_pth=None, external_image_dir=None):
    warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt JPEG data")
    cfg = get_config(settings_pth)
    project_dir = Path(os.path.abspath('')).parent
    experiment_dir = project_dir / cfg.DATA_FOLDER_NM / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME
    results_dir = experiment_dir / cfg.RUNS_FOLDER_NM / cfg.RUN_ID / cfg.RESULTS_FOLDER_NM
    models_dir = experiment_dir / cfg.RUNS_FOLDER_NM / cfg.RUN_ID / cfg.MODELS_FOLDER_NM
    weights_dir = models_dir / f'{cfg.RUN_ID}{cfg.WEIGHTS_FOLDER_SUFFIX}'
    weights_pth = weights_dir / f'{cfg.RUN_ID}{cfg.WEIGHTS_FN_SUFFIX}'
    check_weights('classifier', [weights_pth], load=True)
    check_weights('MegaDetector', [path for path in project_dir.rglob(cfg.MD_WEIGHTS_NM) if path.is_file()])
    class_names_pth = results_dir / f'{cfg.RUN_ID}{cfg.CLASS_NAMES}'
    default_image_dir = project_dir / cfg.DATA_FOLDER_NM / cfg.DEFAULT_IMAGE_FOLDER_NM
    if external_image_dir: #User-supplied filepath from external scripts
        image_dir = Path(external_image_dir)
    elif cfg.IMAGE_FOLDER_PTH:  #Get user-supplied filepath, written into the settings file
        image_dir = Path(cfg.IMAGE_FOLDER_PTH)
    else:  #Run what ever the default is at the moment
        image_dir = default_image_dir
    
    out_csv_path = image_dir /  f'{cfg.EXPERIMENT_NAME}_{cfg.RUN_ID}{cfg.PREDS_CSV_SUFFIX_OUT}'
    out_detailed_csv_path = image_dir /  f'{cfg.EXPERIMENT_NAME}_{cfg.RUN_ID}_full{cfg.PREDS_CSV_SUFFIX_OUT}'
    run_md_ps_pth = Path(os.path.abspath('')) / cfg.SETUP_FOLDER_NM / cfg.RUN_MD_PS_NM

    check_for_empty(image_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    Process_Video.main(root_dir_pth=image_dir, time_interval=0.5)
    
    device, gpu = test_cuda()
    species_list = data_from_json(class_names_pth)
    print(Colour.S + 'The following list contains all the possible output classes from the classifier'
          'Note that the modelled probabilities over all classes will always sum to 1' + Colour.E)
    print(species_list)

    labels_df, empties_df, md_output_pth = find_dataframe(image_dir, species_list, run_md_ps_pth, cfg)    
    print(Colour.S + f'Length of the labels dataframe: {len(labels_df)}' + Colour.E)
    print(labels_df.head(3))
    infer_transforms = get_transforms(cfg)
    dataset = PredatorDataset(labels_df, 
                            infer_transforms, 
                            crop_size=cfg.CROP_SIZE, 
                            buffer= cfg.BUFFER,
                            resize_method=cfg.RESIZE_METHOD, 
                            remove_background=cfg.REMOVE_BACKGROUND,
                            fade_edges=cfg.EDGE_FADE,
                            min_margin=cfg.MIN_FADE_MARGIN,
                            downsample=cfg.MD_RESAMPLE,
                            dt_formats=cfg.EXIF_DT_FORMATS)

    test_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, collate_fn=custom_collate, shuffle=False, num_workers=cfg.NUM_WORKERS)
    model = get_model(weights_pth, species_list, cfg.MODEL_NAME, cfg.HEAD_NAME)

    total_images = len([f.name for f in Path(image_dir).rglob('*.jpg')])
    print('Image Folder: {}, ({} images)'.format(image_dir, total_images))
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if gpu else 'CPU'))
    targets_df, predictions_df, final_crops_list, dt_list, speed = infer_dataset(test_loader, model, species_list, device)
    warnings.resetwarnings()
    bboxes = ['File_Path','x_min', 'y_min', 'Width', 'Height', 'Confidence']
    inferred_df = make_output_table(targets_df, 
                                    predictions_df, 
                                    labels_df[bboxes], 
                                    final_crops_list, 
                                    dt_list, 
                                    species_list, 
                                    empties_df,
                                    cfg.ENCOUNTER_WINDOW, 
                                    cfg.EMPTY_THRESHOLD,
                                    cfg.MD_EMPTY_THRESHOLD)

    
    inferred_df.to_csv(out_detailed_csv_path)
    print('The dataframe saved to CSV')
    
    cols_to_keep = ['File_Path', 'Date_Time', 'Encounter', 'Max_Prob']
    print(inferred_df[cols_to_keep].head())
    inferred_df[cols_to_keep].to_csv(out_csv_path)
    
    try:
        Predictions_2_JSON.main(inferred_df.reset_index(), image_dir, md_output_pth, species_list)
    except Exception as e:
        print(Warn.S +  f'An error occurred with Predictions_2_JSON:'  + Warn.E, e)
        print('The predictions csv should have been produced anyway, but if you need the json file output', 
              ' as well, try checking for corrupted file or video and remove those first, then re-run inference')
    
    print(Colour.S + 'Process Complete'+ Colour.E)
    print(f'The predictions csv file was saved to: {out_csv_path}')
    if gpu:
        gc.collect()
        torch.cuda.empty_cache()
    return inferred_df, speed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", type=str, default=None, help="Filepath to the root directory of imagery to be processed")
    parser.add_argument("--settingsPath", type=str, default=None, help="Filepath to the settings YAML file for this model")
    args = parser.parse_args()
    if (args.dataPath is not None) and (args.settingsPath is not None): 
        print(f'Running Inference.py on {args.dataPath}, with the settings file {args.settingsPath}')
    output, speed = main(settings_pth=args.settingsPath, external_image_dir=args.dataPath)