'''
Trains the model from an existing clean crops and labels file.
Saves out the test set predictions for futher evaluation.
'''
#Standard Python
import os # sys
import time
import gc
import json
import random
import ast
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Machine Learning
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import yaml
#import cv2
from PIL import Image

#PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# ---------------- Functions & Classes for basic setup-----------------------------------------
# ---------------------------------------------------------------------------------------------
class TrainConfig:
    '''Namespace for training hyperparameters'''
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_40'
        self.RUN_ID = 'Run_03'
        self.DESCRIPTION = ''
        self.EXTRA_CORES = 1 #None will result in the max cpu count being used.
        self.FOCAL_GAMMA = 1
        self.MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'
        self.HEAD_NAME = 'ClassifierHead' # Alternative: 'BasicHead'
        self.FOCAL_ALPHA = False
        self.FOCAL_ALPHA_OFFSET = 0
        self.FOCAL_WEIGHTS_OFFSET = 100
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.5
        self.RANDOM_SEED = 2023
        self.MAX_EPOCHS = 25
        self.EPOCH_LENGTH = 100000
        self.EPOCHS_BACKBONE_FROZEN = 10 # Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS = 2 # If unfreezing the backbone, ufreeze this many layers
        self.LOSS_FUNCTION = 'BinaryFocalLoss' # Could also consider 'CrossEntropy' or 'FocalLoss'
        self.PATIENCE = 3 # Stop training if no improvement
        self.BATCH_SIZE = 16 #For eval only
        self.TRAIN_BATCH_SIZE = 64
        self.LEARNING_RATE = 1e-3
        self.INITIAL_LR = 1e-4
        self.WARMUP_EPOCHS = 2
        self.LR_CYCLE_LENGTH = 16
        self.MIN_LR = 1e-5
        self.LR_DECAY = 0.01
        self.WEIGHT_DECAY = 1e-5
        self.WEIGHTED_SAMPLING = False # 1/sqrt(N)
        self.CLASS_JOINS = {}


class ImageConfig:
    '''Wrapper class for image processing parameters'''
    EDGE_FADE = False
    MIN_FADE_MARGIN = 0
    MAX_FADE_MARGIN = 0
    IMAGE_SIZE = 480 #Final width and height in pixels for the cropped images
    INPUT_MEAN = [ 0.485, 0.456, 0.406 ] # mean to be used for normalisation, using values from ImageNet.
    INPUT_STD = [ 0.229, 0.224, 0.225 ] # stddev to be used for normalisation, using values from ImageNet.


class DataConfig:
    '''Wrapper class data related parameters'''
    MIN_SAMPLES = 10
    #TEST_FRACTION = 0.1
    VAL_FRACTION = 0.1
    REDUCE_DATA = None # None #None #40  Reduces the input dataframe size by this factor for debugging


class Paths:
    '''Wrapper class for filepaths'''
    DATA_FOLDER_NM = 'Data'
    INPUT_FOLDER_NM = 'Inputs'
    EXPS_FOLDER_NM = 'Experiments'
    SETTINGS_FOLDER_NM = 'Settings'
    RUNS_FOLDER_NM = 'Runs'
    IMAGE_FOLDER_NM = 'Cropped_Images'
    CROP_LABELS_NM = 'crop_labels.parquet'
    #TEST_DF_FN = '_test_split.parquet'  #The final label df in the right form for the dataloader
    VAL_DF_FN = '_val_split.parquet'
    WEIGHTS_FOLDER_SUFFIX = '_weights'
    METRICS_FN_SUFFIX = '_monitor_train.png'
    BEST_WEIGHTS_FN_SUFFIX = '_best_weights.pt'
    RESULTS_DF_SUFFIX = '_df.pkl'
    CLASS_NAMES_OUT = '_class_names.json'
    RESULTS_FOLDER_NM = 'Results'  # Increment or name this to name the results folder
    MODELS_FOLDER_NM = 'Models'

    def __init__(self, experiment_name, run_id=None, ):
        _project_dir = Path(__file__).resolve().parent.parent
        _experiment_dir = _project_dir / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / experiment_name

        self.image_dir = _experiment_dir / self.INPUT_FOLDER_NM / self.IMAGE_FOLDER_NM
        self.run_id = run_id if run_id is not None else datetime.now().strftime('%y_%m_%d_%H')
        self.labels_path = _experiment_dir / self.INPUT_FOLDER_NM / self.CROP_LABELS_NM
        self.labels_path_extra_cols = _experiment_dir / self.INPUT_FOLDER_NM / 'cleaned_labels.parquet'
        self.results_dir = _experiment_dir / self.RUNS_FOLDER_NM / run_id / self.RESULTS_FOLDER_NM
        self.models_dir = _experiment_dir / self.RUNS_FOLDER_NM / run_id / self.MODELS_FOLDER_NM
        self.weights_pth = self.models_dir / f'{run_id}{self.WEIGHTS_FOLDER_SUFFIX}'
        self.final_weights_pth = self.weights_pth / f'{run_id}{self.BEST_WEIGHTS_FN_SUFFIX}'
        self.class_names_pth = self.results_dir / f'{run_id}{self.CLASS_NAMES_OUT}'
        #self.test_parquet_pth = self.results_dir / f'{run_id}{self.TEST_DF_FN}'
        self.val_parquet_pth = self.results_dir / f'{run_id}{self.VAL_DF_FN}'
        self.train_metrics_pth = self.results_dir / f'{run_id}{self.METRICS_FN_SUFFIX}'
        self.pickle_paths = {
            'val_target_df': self.results_dir / f'{run_id}_val_target{self.RESULTS_DF_SUFFIX}',
            'val_pred_df': self.results_dir / f'{run_id}_val_pred{self.RESULTS_DF_SUFFIX}',
            'train_target_df': self.results_dir / f'{run_id}_train_target{self.RESULTS_DF_SUFFIX}',
            'train_pred_df': self.results_dir / f'{run_id}_train_pred{self.RESULTS_DF_SUFFIX}'}    
        
        for fldr in [self.results_dir, self.models_dir]:
            if not os.path.exists(fldr):
                os.makedirs(fldr)

def get_settings(settings_pth=None):
    """Gets an instance of the configuration classes, then looks for the settings file, 
    if it finds a matching key the value is updated, or evaluated to python expressions then updated"""

    evaluate_list = ['USE_CUTMIX', 'USE_MIXUP', 'FOCAL_ALPHA', 'WEIGHTED_SAMPLING', 'FOCAL_ALPHA', 'USE_FOCAL_LOSS',
                    'EPOCHS_BACKBONE_FROZEN', 'WEIGHT_DECAY', 'LEARNING_RATE', 'INITIAL_LR', 'WARMUP_EPOCHS','LR_CYCLE_LENGTH',
                     'MIN_LR', 'DEBUG', 'EDGE_FADE', 'CLASS_JOINS', 'REDUCE_DATA']

    train_settings = TrainConfig()
    image_settings = ImageConfig()
    data_settings = DataConfig()

    if settings_pth:
        with open(settings_pth, 'r') as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        for key, value in yaml_data.items():
            for cfg in [train_settings, image_settings, data_settings]:
                if hasattr(cfg, key):
                    if (key in evaluate_list) and (isinstance(value, str)):
                        setattr(cfg, key, ast.literal_eval(value))
                    else:
                        setattr(cfg, key, value)

    print(Colour.S + 
          'Training with settings from settings file: '
          + f'{train_settings.EXPERIMENT_NAME}, run: {train_settings.RUN_ID}'
          + Colour.E)

    return train_settings, image_settings, data_settings


def set_hardware(cfg):
    if cfg.EXTRA_CORES is None:
        num_workers = cpu_count()-1
    else:
        num_workers = cfg.EXTRA_CORES
    gpu = torch.cuda.is_available()
    accelerator = 'gpu' if gpu else 'cpu'
    torch.set_float32_matmul_precision('medium') #could try setting to 'high' at expense of speed

    print(f'Loading data with {num_workers + 1} CPU cores')
    print(f"Using torch {torch.__version__} "
        f"{torch.cuda.get_device_properties(0) if accelerator == 'gpu' else 'CPU'}")

    if accelerator =='gpu':
        gc.collect()
        torch.cuda.empty_cache()

    return num_workers, accelerator

# -----------------------------------Functions & Classes-----------------------------------------
# -----------------------------------------------------------------------------------------------
class Colour:
    '''Turn shell print statements bold blue'''
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_training_data(df, cfg):
    """
    Combine classes as per the settings for this training run, into a new 'Target' Column, keeping
    the original 'species' column unchanged as per the original cleaning notebook
    """

    #df = pd.read_parquet(df_path)
    df['Targets'] = df['Species'].copy()

    print('The original dataframe')
    print(df.columns)

    # Get a dict to handle all name changes by merger
    flattened_list = [(key, value) for key, values in cfg.CLASS_JOINS.items() for value in values]
    inv_merge = {val:key for (key, val) in flattened_list}
    # Add the merger dict, to the name change dict, then do the name change
    df['Targets'] = df['Targets'].replace(to_replace=inv_merge)
    n_reduced = df['Species'].nunique()  - df['Targets'].nunique()
    targets = df['Targets'].unique()
    print(f'Total classes reduced by {n_reduced} due to class mergers')
    return df, targets

'''
def remove_missing_images(dir_pth, dataframe):
    """Checks if any images have gone missing from the crops directory, and if so removes from the dataframe"""
    print(dataframe.head())
    species = set(list(dataframe['Species'].unique()))
    names_in_dir = [str(f.name) for f in Path(dir_pth).rglob('*.*')]
    names_in_df = list(dataframe['Cropped_File_Name'].unique())
    missing_names = list(set(names_in_df) - set(names_in_dir))
    print(f'There are {len(missing_names)} files in the labels dataframe, not found in the image folder.')
    old_length = len(dataframe)
    mask = dataframe['Cropped_File_Name'].isin(missing_names)
    dataframe = dataframe[~mask]
    print(f'{old_length - len(dataframe)} rows were removed from the labels dataframe')
    new_species_list = sorted(list(dataframe['Species'].unique()))
    if set(species) != new_species_list:
        removed_species = species.difference(set(new_species_list))
        print(f'These species were removed from the model as their images were all missing: " {removed_species}')
        print(f"There are {dataframe['Targets'].nunique()}, unique targets after removing missing images")
    return dataframe.copy()
'''

def remove_missing_images(dir_pth, dataframe):
    """Removes missing images from the dataframe if they are not found in the directory"""
    #print(dataframe.head())

    species = set(dataframe['Species'].unique())
    names_in_dir = {f.name for f in Path(dir_pth).iterdir() if f.is_file()}  # Faster than rglob('*.*')
    names_in_df = set(dataframe['Cropped_File_Name'].unique())
    missing_names = names_in_df - names_in_dir

    print(f'There are {len(missing_names)} files in the labels dataframe, not found in the image folder.')
    old_length = len(dataframe)
    dataframe = dataframe[~dataframe['Cropped_File_Name'].isin(missing_names)]
    print(f'{old_length - len(dataframe)} rows were removed from the labels dataframe')

    new_species_list = set(dataframe['Species'].unique())
    removed_species = species - new_species_list

    if removed_species:
        print(f'These species were removed from the model as their images were all missing: {removed_species}')
        print(f"There are {dataframe['Targets'].nunique()} unique targets after removing missing images.")

    return dataframe.copy()



def remove_rare_classes_from_training(df, min_targets=10):
    class_counts = df['Targets'].value_counts()
    valid_classes = class_counts[class_counts >= min_targets].index
    filtered_df = df[df['Targets'].isin(valid_classes)].copy()
    return filtered_df


def split_data(df, val_fraction=0.1, debug_speed=None):
    df=df.copy().reset_index(drop=True)
    targets_list = list(map(str, list(df['Targets'].unique())))
    targets_list.sort()
    num_classes = len(targets_list)
    print(f'There are {len(df)}, total images in labels dataframe')
    print(f'There are a total of {num_classes} classes')

    camera_list = df["Camera"].value_counts().index.tolist()
    np.random.shuffle(camera_list)
    n = max(1, int(len(camera_list) * val_fraction))
    val_cameras = camera_list[:n]
    print('Selected Cameras for Validation')
    print(val_cameras)

    total_images = len(df)

    if debug_speed:
        df = df.groupby("Species").sample(frac=0.1, random_state=42).reset_index(drop=True)

    val_idx = df.index[df["Camera"].isin(val_cameras)].tolist()
    train_idx = df.index[~df["Camera"].isin(val_cameras)].tolist()

    all_idx = [train_idx, val_idx]
    

    print(' Training set size: \t', len(train_idx))
    print(' Validation set size: \t', len(val_idx))
    print(' Total dataset: \t', total_images)
    print(f'The target list is: {targets_list}')
    print(f'The length of the target list is: {len(targets_list)}')

    df.drop(columns=['Camera'], inplace=True)
    splits_list =  [df.iloc[idx].copy() for idx in all_idx]

    return splits_list, targets_list


def get_new_filepath(df, fldr):
    df.drop('File_Path', axis=1, inplace=True) #remove the original one
    apply_fn = df.apply(lambda row: str(fldr / row['Cropped_File_Name']), axis=1)
    df['File_Path'] = apply_fn
    return df[['Targets', 'File_Path']].copy()


def save_as_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f)
    return


def encode_df(df, class_list, image_fldr=None):
    df = df.copy()
    if image_fldr is not None:
        df = get_new_filepath(df, image_fldr)
    df[['Targets','File_Path']] = df[['Targets','File_Path']].astype(str)
    df = pd.concat([df, pd.get_dummies(df['Targets'], dtype=int)], axis=1)
    missing_classes = list(set(class_list).difference(list(df.Targets.unique())))
    df[missing_classes] = 0 #Ensures all dataframes have the same columns
    df = df[['Targets','File_Path'] + class_list] #Ensure all dfs have cols in the same order
    return df


def get_class_weights(df):
    '''This function is for calculating weighted sampling.'''
    df = df.iloc[:, 2:] # removing the 'filepath' and 'targets' columns
    col_sums = df.sum()
    counts_array = col_sums.values
    counts_array = np.sqrt(counts_array)
    class_weights = counts_array.tolist()
    sample_idxs = np.argmax(df.values, axis=1).tolist()
    return [1 / class_weights[idx] for idx in sample_idxs]


def remove_rare_classes(target_df, pred_df, rare_threshold):
    col_sums = target_df.sum()
    mask = col_sums >= rare_threshold
    target_df = target_df.loc[:,mask]
    pred_df = pred_df.loc[:,mask]
    return target_df, pred_df


def get_map_score(target_df, pred_df, average='macro'):
    target_df, pred_df = remove_rare_classes(target_df, pred_df, 1)

    col_sums = target_df.sum()
    mask = col_sums >= 1 #keeping this in to avoid division by 0
    targs_arr = target_df.loc[:,mask].copy().values
    preds_arr = pred_df.loc[:,mask].copy().values
    if average is None:
        scores_vals = average_precision_score(targs_arr,preds_arr, average=None)
        scores_keys = target_df.columns[mask].tolist()
        scores_dict = {k:v for (k,v) in zip(scores_keys, scores_vals)}
    else:
        scores_dict = {'mean': average_precision_score(targs_arr,preds_arr, average=average)}
    return scores_dict['mean']


class PredatorDataset(Dataset):
    def __init__(self,
                 labels_df,
                 transform=None,
                 edge_fade=False,
                 min_margin=0.05,
                 max_margin=0.05):
        self.df = labels_df  # pd.read_csv(metadata_csv_path)
        self.transform = transform
        self.fade_edges = edge_fade
        self.min_margin = min_margin
        self.max_margin = max_margin

    def __len__(self):
        return len(self.df)

    def load_image(self, image_path, mode):
        try:
            image = Image.open(image_path)
            if mode == 'RGB':
                image = image.convert('RGB')
            if image.size != (600, 600):
                print(f"Image at {image_path} has shape {image.size}, not (600, 600).")
            image_array = np.array(image)
            return image_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

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

    def __getitem__(self, index):
        while True:
            row = self.df.iloc[index]
            f_pth = row['File_Path']
            image = self.load_image(f_pth, 'RGB')

            if image is not None:
                break
            print(f"Warning: Unable to load the image at '{f_pth}'. Skipping...")
            index = torch.randint(0, len(self.df), (1,)).item()  # Get a random index

        if self.fade_edges:
            image = self.edge_fade(image, min_margin=self.min_margin, max_margin=self.max_margin)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        ohe_vals = row.iloc[2:].values.astype(int)
        #targets = torch.tensor(ohe_vals).float().to(torch.float32)
        targets = torch.from_numpy(ohe_vals).float()
        return image, targets, f_pth


class ImageAugmentation():
    def __init__(self, mean, std, height, width):
        self.train = A.Compose([
            A.OneOf([A.RandomFog(p=1), 
                     A.RandomRain(rain_type='torrential', p=1)], p=0.2),
            A.Sequential([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-.5, .1), rotate_limit=30, p=0.8),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
                A.RandomCrop(height=height, width=width, p=1)]),
            A.OneOf([A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1), 
                     A.HueSaturationValue(p=1),
                     A.RandomBrightnessContrast(p=1),
                     #A.ChannelShuffle(p=1),  perfomance on hidden set jumped 1.7% when this was removed.
                    ], p=0.5),
            A.ToGray(p=0.2), 
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(src_radius=200, p=.2),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower = 70, p=.2),
            A.GaussianBlur(sigma_limit=9, p=0.1),  #Before exp 36 run 06 was: sigma_limit=9, p=0.1
            A.Normalize(mean=mean, std=std),
            ToTensorV2()])

        self.val = A.Compose([
            A.CenterCrop(height=height, width=width, p=1),
            A.Normalize(mean=mean, std=std), ToTensorV2()])


def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(lr, params, weight_decay):
    model_optimizer = Adam(
            filter(lambda p: p.requires_grad, params), 
            lr=lr,
            weight_decay=weight_decay)
    interval = "epoch"
    lr_scheduler = CosineAnnealingWarmRestarts(
                            model_optimizer,
                            T_0=20,  #16
                            T_mult=1,
                            eta_min=1e-5,
                            last_epoch=-1)

    return { "optimizer": model_optimizer,
             "lr_scheduler": {"scheduler": lr_scheduler,
                        "interval": interval,
                        "monitor": "val_loss",
                        "frequency": 1}}


class FocalLoss(nn.Module):
    '''Multi-class Focal loss with pre-computed values for alpha 
       assumes one-hot encoded targets'''

    def __init__(self, alphas, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alphas = torch.FloatTensor(alphas)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = torch.sum(targets * self.alphas, dim=1)
        loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
    

# https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
class WBCEFocalLoss(nn.Module):
    '''
    Binary Cross Entropy Focal Loss weighted by 1/(sqrt(offset + N)) 
    '''
    def __init__(self, class_weights, alpha=0.7, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.class_weights = torch.FloatTensor(class_weights).to('cuda')
        print(f'hello, inside bce, self.gamma is {self.gamma}  self.class_weights is {self.class_weights}')

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)
        focal_weight = targets * (1 - probas) ** self.gamma + (1 - targets) * probas ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = self.class_weights * focal_weight * alpha_weight * bce_loss
        return loss.mean()


# https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
class AlphaBCEFocalLoss(nn.Module):
    '''
    f_bce_alphas: My experiment with offsetting the alpha linearly by
     -alpha_offset (for most common) to + alpha_offset (least common)
     I don't think this idea is much good, because we are treating different classes
     differently so they can no longer be compared with each other unbiased.
    '''
    def __init__(self, alphas, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alphas = torch.FloatTensor(alphas)

    def forward(self, logits, targets):
        _alphas = self.alphas.to(logits.device)  
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)
        focal_weight = targets * (1 - probas) ** self.gamma + (1 - targets) * probas ** self.gamma
        alpha_weight = targets * _alphas + (1 - targets) * (1 - _alphas)
        loss = focal_weight * alpha_weight * bce_loss
        return loss.mean()


class BCEFocalLoss(nn.Module):
    '''
    Binary Cross Entropy Focal Loss
    '''
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)
        focal_weight = targets * (1 - probas) ** self.gamma + (1 - targets) * probas ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = focal_weight * alpha_weight * bce_loss
        return loss.mean()


def get_loss_function(train_df,
                      name,
                      base_alpha=0.25,
                      alpha_offset=0,
                      weights_offset=10000,
                      gamma=1):
    '''Select and instantiate a choice of loss function'''
    train_df = train_df.iloc[:, 2:] # removing the 'filepath' and 'targets' columns
    col_sums = train_df.sum()
    counts_array = col_sums.values
    inv_counts = counts_array.mean() / counts_array
    f_bce_alphas = base_alpha + (0.5 - counts_array / counts_array.max()) * 2 * alpha_offset

    weights = 1 / (np.sqrt(weights_offset + counts_array))
    weights /= weights.max() # Normalize to a range from 0 to 1


    if name == 'FocalLoss':
        print(f'Using focal loss with gamma of {gamma}')
        loss = FocalLoss(inv_counts, gamma)
        activation = 'SoftMax'
    elif name == 'BinaryFocalLoss':
        print(f'Using binary focal loss with gamma:  {gamma}, alpha: {base_alpha}')
        loss = BCEFocalLoss(base_alpha, gamma)
        activation = 'Sigmoid'
    elif name == 'AlphaBinaryFocalLoss':
        print(f'Using binary focal loss with gamma of {gamma}, base_alpha of {base_alpha}, offset {alpha_offset}')
        loss = AlphaBCEFocalLoss(f_bce_alphas, gamma)
        activation = 'Sigmoid'
    elif name == 'WeightedBinaryFocalLoss':
        print(f'Using weighted binary focal loss with gamma of {gamma}, alpha of {base_alpha}')
        loss = WBCEFocalLoss(weights, alpha=base_alpha, gamma=gamma)
        activation = 'Sigmoid'
    else:
        loss = nn.CrossEntropyLoss()
        print(f'Using cross entropy loss')
        activation = 'SoftMax'
    return activation, loss


class ClassifierHead(nn.Module):
    '''A slightly larger classifier head with two linear layers'''
    def __init__(self, num_features, num_classes, dropout_rate=0.2):
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
    '''Bare bones fc classifier head'''
    def __init__(self, in_features, num_classes):
        super(BasicHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomModel(pl.LightningModule):
    '''Pytorch Lightning Model with customised callbacks, logging and loss functions'''
    def __init__(self,
                class_list,
                loss = nn.CrossEntropyLoss(),
                lr = 1e-3,
                weight_decay = 1e-5,
                unfreeze=None,
                unfreeze_lyrs=2,
                pickle_path_dict = None,
                mixup_alpha = 0.5,
                use_mixup = False,
                model_name = 'efficientnetv2_l_21k',
                custom_head = ClassifierHead,
                initial_lr=  1e-5,
                warmup_epochs= 2,
                cycle_length= 6,
                min_lr= 1e-5,
                lr_decay= .5,
                batch_size=64
                ):
        super().__init__()

        self.unfreeze = unfreeze
        self.unfreeze_layers = unfreeze_lyrs
        self.lr = lr
        self.decay = weight_decay
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.pickle_paths = pickle_path_dict
        self.backbone = timm.create_model(model_name, pretrained=True)
        #self.backbone.set_grad_checkpointing(True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.in_features = self.backbone.classifier.in_features
        print(f'There are {self.in_features} input features to the classifier head {self.num_classes} outputs')
        self.backbone.classifier = custom_head(self.in_features, self.num_classes)
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        self.mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.loss_function = loss
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.lr_decay = lr_decay
        self.batch_size = batch_size

    def forward(self, images):
        logits = self.backbone(images)
        return logits


    def configure_optimizers(self):
        def custom_lr_scheduler(epoch):
            '''CosineAnealingWarmRestarts but with a decay and a warmup'''
            initial = self.initial_lr / self.lr
            rel_min = self.min_lr / self.lr
            step_size = (1-initial) / self.warmup_epochs
            warmup = initial + step_size * epoch if epoch <= self.warmup_epochs else 1
            cycle = epoch-self.warmup_epochs
            decay = 1 if epoch <= self.warmup_epochs else self.lr_decay ** (cycle // self.cycle_length)
            phase = np.pi * (cycle % self.cycle_length) / self.cycle_length
            cos_anneal = 1 if epoch <= self.warmup_epochs else  rel_min + (1 - rel_min) * (1 + np.cos(phase)) / 2
            return warmup * decay * cos_anneal #this value gets multipleid by the initial lr (self.lr)

        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)
        return [optimizer], [scheduler]


    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        alpha=self.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_with_mixup(self, x, y):
        x, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
        y_pred = self(x)
        loss_mixup = mixup_criterion(self.loss_function, y_pred, y_a, y_b, lam)
        return loss_mixup, y_pred

    def training_step(self, batch, batch_idx):
        image, target, _ = batch
        if self.mixup:
            loss, y_pred = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred,target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        train_output = {"train_loss": loss, "logits": y_pred, "targets": target}
        self.train_outputs.append(train_output)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target, _ = batch
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, batch_size=self.batch_size)
        output = {"val_loss": val_loss, "logits": y_pred, "targets": target}
        self.val_outputs.append(output)
        return {"val_loss": val_loss, "logits": y_pred, "targets": target}

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def on_validation_epoch_end(self):
        val_outputs = self.val_outputs
        avg_val_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean().cpu().detach().numpy()
        output_val_logits = torch.cat([x['logits'] for x in val_outputs],dim=0)
        val_targets = torch.cat([x['targets'] for x in val_outputs],dim=0).cpu().detach().numpy()

        train_outputs = self.train_outputs
        if train_outputs:
            train_losses = [x['train_loss'].cpu().detach().numpy() for x in train_outputs]
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            output_train_logits = torch.cat([x['logits'] for x in train_outputs],dim=0)
            train_targets = torch.cat([x['targets'] for x in train_outputs],dim=0).cpu().detach().numpy()
        else:
            avg_train_loss = avg_val_loss #we need this because the first time it's an empty list
            output_train_logits = torch.ones(1,output_val_logits.shape[1])
            train_targets = torch.zeros(1, output_val_logits.shape[1])

        val_probs = F.softmax(output_val_logits, dim=1).cpu().detach().numpy()
        train_probs = F.softmax(output_train_logits, dim=1).cpu().detach().numpy()

        val_pred_df = pd.DataFrame(val_probs, columns = self.class_list)
        val_target_df = pd.DataFrame(val_targets, columns = self.class_list)
        train_pred_df = pd.DataFrame(train_probs, columns = self.class_list)
        train_target_df = pd.DataFrame(train_targets, columns = self.class_list)

        train_cmap = get_map_score(train_target_df, train_pred_df) if len(train_target_df) > 16 else 1
        val_cmap = get_map_score(val_target_df, val_pred_df) if len(train_target_df) > 16 else 1

        self.metrics_list.append({'train_loss':avg_train_loss,
                                  'val_loss': avg_val_loss, 
                                  'train_cmap': train_cmap,
                                  'val_cmap': val_cmap, 
                                  })

        print(f'epoch {self.current_epoch} train loss {avg_train_loss}')
        print(Colour.S + f'epoch {self.current_epoch} validation loss: ' + Colour.E, avg_val_loss)
        print(Colour.S +f'epoch {self.current_epoch} validation mAP score: ' + Colour.E, val_cmap)
        optimizer_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f'Learning rate from optimiser at epoch {self.current_epoch}: {optimizer_lr}')

        val_target_df.to_pickle(self.pickle_paths['val_target_df'])
        val_pred_df.to_pickle(self.pickle_paths['val_pred_df'])
        train_target_df.to_pickle(self.pickle_paths['train_target_df'])
        train_pred_df.to_pickle(self.pickle_paths['train_pred_df'])
        self.val_outputs = []
        self.train_outputs = []
        self.val_epoch +=1

    def on_train_epoch_end(self, *args, **kwargs):
        if (self.unfreeze is not None) and (self.current_epoch == self.unfreeze):
            unfrozen_layers = list(self.backbone.children())[-self.unfreeze_layers:]

            for layer in unfrozen_layers:
                if not isinstance(layer, nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = True
            print(Colour.S + f'Unfreezing the top {self.unfreeze_layers} '
            f'layers of the backbone after {self.current_epoch} epochs' + Colour.E)

    def get_my_metrics_list(self):
        return self.metrics_list


def get_dataloaders(df_train,
                    df_valid,
                    img_cfg,
                    img_transforms,
                    sample_weights=None,
                    num_workers=0,
                    batch_size=64,
                    epoch_length=1000000):
    edge_fade = img_cfg.EDGE_FADE
    min_margin = img_cfg.MIN_FADE_MARGIN
    max_margin = img_cfg.MAX_FADE_MARGIN

    ds_train = PredatorDataset(df_train,
                               img_transforms.train,
                               edge_fade=edge_fade,
                               min_margin=min_margin,
                               max_margin=max_margin)
    ds_val = PredatorDataset(df_valid,
                             img_transforms.val,
                             edge_fade=edge_fade,
                             max_margin=min_margin) #Fix the max margin to the minimum for val/test

    p_workers = True if num_workers > 0 else False
    if sample_weights is not None:
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=epoch_length)
        dl_train = DataLoader(ds_train,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=num_workers)
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, persistent_workers=p_workers,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers = num_workers)
    return dl_train, dl_val, ds_train, ds_val


class EarlyStoppingMinEpochs(EarlyStopping):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            print('Not checking for early stopping check because start_epoch not yet reached')
            return
        # Call the parent method to retain the usual early stopping logic
        print('Checking for early stopping')
        super().on_validation_end(trainer, pl_module)


def run_training(weights_dir,
                 dl_train,
                 dl_val,
                 logger=None,
                 epochs=16,
                 patience=4,
                 loss_function = nn.CrossEntropyLoss(),
                 model=None
                 ):
    '''Function to instantiate trainer, run training
    returns training metrics and path to the last weights'''

    print("Running training...")
    early_stop_callback = EarlyStoppingMinEpochs(monitor="val_loss",
                                                 start_epoch=5,
                                                 min_delta=0,
                                                 patience=patience,
                                                 verbose= True,
                                                 mode="min")

    # saves top- checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=8,
                                          monitor="val_loss",
                                          mode="min",
                                          dirpath=weights_dir,
                                          save_last= True,
                                          save_weights_only=True,
                                          verbose= True,
                                         )

    callbacks_to_use = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks_to_use,
        precision='16-mixed',
        accelerator='gpu')

    print("Running trainer.fit")
    trainer.fit(model, train_dataloaders = dl_train, val_dataloaders = dl_val)
    best_model_pth = trainer.checkpoint_callback.best_model_path
    metrics = model.get_my_metrics_list()
    del model, trainer, loss_function, dl_train, dl_val
    return metrics, best_model_pth


def get_best_model(class_list, model_pth, model_name, custom_head):
    '''Loads a model and sets up for evaluation'''
    print(f'using model {model_name} for evaluation')
    best_model_state_dict = torch.load(model_pth)['state_dict']
    best_model = CustomModel(class_list, model_name=model_name, custom_head=custom_head)
    best_model.load_state_dict(best_model_state_dict)
    best_model.eval()
    return best_model


def check_best_model(best_model,
                     df,
                     use_gpu,
                     img_transforms,
                     result_csv_path=None,
                     activation='Sigmoid',
                     batch_size=16,
                     num_batches=8):
    '''Checks the model, and runs a small sample of images as a quick sanity check'''

    if use_gpu:
        best_model = best_model.cuda()
    test_ds = PredatorDataset(df, img_transforms)
    loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    print(f'Evaluating the validation images with the best model')
    correct = 0
    total_preds= 0
    classes = [name for name in df.columns if name not in {'File_Path', 'Targets'}]
    #pbar = tqdm(total=batch_size*num_batches)

    start_time = time.time()
    targets_list, predictions_list, paths_list = [], [], []

    for images, targets, paths in loader:
        if use_gpu:
            images, targets = images.cuda(), targets.cuda()
        logits = best_model(images)
        if activation == 'Softmax':
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.sigmoid(logits)
        for _probs, _target in zip(probs, targets):
            target_idx = torch.argmax(_target)
            prediction_idx = torch.argmax(_probs)
            if target_idx == prediction_idx:
                correct += 1
            total_preds += 1
            targets_list.append(classes[target_idx])

        #targets_list.extend(targets.detach().cpu())
        predictions_list.extend(probs.detach().cpu().numpy())
        paths_list.extend(list(paths))

        #pbar.update(len(images))
        probs.det
    #pbar.close()
    preds_array = np.vstack(predictions_list)
    results_df = pd.DataFrame(preds_array, columns=classes)
    results_df['Targets'] = targets_list
    results_df['File_Path'] = paths_list
    results_df = results_df.reindex(columns=['Targets', 'File_Path'] + classes)
    print(results_df.head())
    if result_csv_path is not None:
        results_df = results_df
        results_df.to_parquet(result_csv_path)

    total_time = time.time()-start_time
    accuracy = correct / total_preds
    print(f'There were {correct} correct predictions from {total_preds} samples.'
          f'A mean accuracy of {accuracy:.2f}')
    print(f'Processed {total_preds} test samples in {total_time:.2f} seconds')
    print(f'That is a mean of {total_preds/total_time:.2f} images per second')
    del best_model, images, targets, logits, probs
    return


def plot_train_metrics(metrics, save_path):
    '''Saves a plot of the training metrics for later analysis'''
    #The first check is at 0, second at 0.5.
    train_losses = [x['train_loss'] for x in metrics][1:]
    val_losses = [x['val_loss'] for x in metrics][1:]
    train_precision = [x['train_cmap'] for x in metrics][1:]
    val_precision = [x['val_cmap'] for x in metrics][1:]
    num_checks = len(val_losses) + 1  #+1 because the list was sliced
    print(f'There were {num_checks} checkpoints recorded')
    time_axis = [0.5*x - 0.5 for x in range(2, num_checks+1)]

    _, ax = plt.subplots()
    plt.plot(time_axis, train_losses, 'r', label='Train Loss')
    plt.plot(time_axis, val_losses, '--k', label='Val Loss')
    plt.legend()
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.tick_params('both', colors='r')

    # Get second axis
    ax2 = ax.twinx()
    plt.plot(time_axis, train_precision, 'b', label='Train mAP')
    plt.plot(time_axis, val_precision, '--g', label='Val mAP')
    ax2.set_ylabel('Accuracy')
    plt.legend()
    plt.legend(loc='lower left')
    ax.tick_params('both', colors='b')
    plt.savefig(save_path)

# ----------------------------------- Main  --------------------------------------
# --------------------------------------------------------------------------------
def train(settings_path):   #settings_path="/media/olly/Red_SSD/Alita/Settings/Exp_500_Run_01.yaml"
    '''Main function.  Can be run by directly by setting the default settings path an argument
    or calling from elsewhere with a new settings file, or from a shell terminal with flags'''
    train_cfg, image_cfg, data_cfg = get_settings(settings_path)
    paths = Paths(train_cfg.EXPERIMENT_NAME, train_cfg.RUN_ID)
    num_workers, accelerator  = set_hardware(train_cfg)

    print('The crop labels')
    labels_df = pd.read_parquet(paths.labels_path)
    print(labels_df.head())

    if 'Camera' not in labels_df.columns:
        labels_df_extra_cols = pd.read_parquet(paths.labels_path_extra_cols)
        print(labels_df_extra_cols.head())
        labels_df = labels_df.merge(labels_df_extra_cols[['File_Path', 'Camera']], on='File_Path', how='inner')
 
    start_time = time.time()
    #num_files = sum(1 for entry in tqdm(os.scandir(paths.image_dir), desc='Scanning for training images') if entry.is_file())
    print(f'Label data location: {str(paths.labels_path)}')
    pl.seed_everything(train_cfg.RANDOM_SEED, workers=True)
    random.seed(2025)

    in_df, _ = get_training_data(labels_df, train_cfg)
    in_df = remove_missing_images(paths.image_dir, in_df)
    in_df = remove_rare_classes_from_training(in_df, min_targets=10)
    splits, target_list  = split_data(in_df, val_fraction=0.1, debug_speed=data_cfg.REDUCE_DATA)
    save_as_json(target_list, paths.class_names_pth)
    train_df, val_df = [encode_df(_df, target_list, paths.image_dir) for _df in splits]

    print(f"Number of unique targets in the training dataframe is {train_df['Targets'].nunique()}")
    print(f"Number of unique targets in the validation dataframe is {val_df['Targets'].nunique()}")

    weights = get_class_weights(train_df) if train_cfg.WEIGHTED_SAMPLING else None

    augmentations = ImageAugmentation(mean=image_cfg.INPUT_MEAN,
                                      std=image_cfg.INPUT_STD,
                                      height=image_cfg.IMAGE_SIZE,
                                      width=image_cfg.IMAGE_SIZE)
    print('df_train')
    print(train_df.head())

    dl_train, dl_val, _, _ = get_dataloaders(train_df,
                                             val_df,
                                             image_cfg,
                                             augmentations,
                                             weights,
                                             num_workers,
                                             batch_size=train_cfg.TRAIN_BATCH_SIZE,
                                             epoch_length=train_cfg.EPOCH_LENGTH)

    logger = CSVLogger(save_dir=paths.results_dir, name=train_cfg.RUN_ID)

    activation, loss = get_loss_function(train_df,
                                         name=train_cfg.LOSS_FUNCTION,
                                         base_alpha = train_cfg.FOCAL_ALPHA,
                                         alpha_offset = train_cfg.FOCAL_ALPHA_OFFSET,
                                         weights_offset=train_cfg.FOCAL_WEIGHTS_OFFSET,
                                         gamma = train_cfg.FOCAL_GAMMA)

    head = ClassifierHead if train_cfg.HEAD_NAME == 'ClassifierHead' else BasicHead

    training_model = CustomModel(target_list,
                        loss=loss,
                        lr = train_cfg.LEARNING_RATE,
                        weight_decay=train_cfg.WEIGHT_DECAY,
                        unfreeze=train_cfg.EPOCHS_BACKBONE_FROZEN,
                        unfreeze_lyrs=train_cfg.UNFREEZE_LAYERS,
                        pickle_path_dict=paths.pickle_paths,
                        mixup_alpha = train_cfg.MIXUP_ALPHA,
                        use_mixup=train_cfg.USE_MIXUP,
                        model_name=train_cfg.MODEL_NAME,
                        custom_head=head,
                        initial_lr=train_cfg.INITIAL_LR,
                        warmup_epochs=train_cfg.WARMUP_EPOCHS,
                        cycle_length=train_cfg.LR_CYCLE_LENGTH,
                        min_lr=train_cfg.MIN_LR,
                        lr_decay=train_cfg.LR_DECAY,
                        batch_size=train_cfg.BATCH_SIZE)
    

    metrics, best_model_path = run_training(paths.weights_pth,
                                    dl_train,
                                    dl_val,
                                    logger=logger,
                                    epochs = train_cfg.MAX_EPOCHS,
                                    patience=train_cfg.PATIENCE,
                                    loss_function= loss,
                                    model=training_model)

    end_time = time.time()
    print(f'Processing completed in {end_time-start_time:.2f} seconds')
    plot_train_metrics(metrics, paths.train_metrics_pth)

    best_model = get_best_model(target_list,
                                best_model_path,
                                train_cfg.MODEL_NAME,
                                custom_head=head)
    torch.save(best_model.state_dict(), paths.final_weights_pth)
    print(f'Final model saved to {paths.final_weights_pth}')

    check_best_model(best_model,
                     val_df,
                     (accelerator =='gpu'),
                     augmentations.val,
                     result_csv_path=paths.val_parquet_pth,
                     batch_size=train_cfg.BATCH_SIZE,
                     activation=activation)
    del best_model

    gc.collect()
    torch.cuda.empty_cache()

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    train(settings_path = '/home/olly/Desktop/Alita/Data/Experiments/Exp_46/Inputs/Exp_46_Run_09.yaml')
