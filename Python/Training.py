#Trains the model from existing clean crops and labels file.  
#Saves out the test set predictions for futher evaluation.

import torch
from torch import nn
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms as transforms
from torchvision import models as models
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import timm
import torch.nn.functional as F
import time
import gc
import json
from datetime import datetime
from multiprocessing import cpu_count
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts, LambdaLR
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import yaml
import cv2

# ---------------- Functions & Classes for basic setup-----------------------------------------
# ---------------------------------------------------------------------------------------------
class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_26'
        self.RUN_ID = 'Run_01'
        self.DESCRIPTION = ''
        self.IMAGE_SIZE = 480 #Final width and height in pixels for the cropped images
        self.EXTRA_CORES = 3 #None will result in the max cpu count being used. 0 Means just the original processor
        self.TEST_FRACTION = 0.1
        self.VAL_FRACTION = 0.1
        self.FOCAL_GAMMA = 1
        self.MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'
        self.HEAD_NAME = 'ClassifierHead' # Alternative: BasicHead
        self.FOCAL_ALPHA = False 
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.5  
        self.RANDOM_SEED = 2023
        self.MAX_EPOCHS = 20
        self.EPOCH_LENGTH = 100000
        self.EPOCHS_BACKBONE_FROZEN = 10 # Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS = 2 # If unfreezing the backbone, ufreeze this many layers
        self.LOSS_FUNCTION = 'FocalLoss' # Could also consider 'CrossEntropy'  
        self.PATIENCE = 3 # Stop training if the last 2 epochs made no improvement
        self.BATCH_SIZE = 16 #For eval only
        self.TRAIN_BATCH_SIZE = 64
        self.LEARNING_RATE = 1e-3
        self.INITIAL_LR = 1e-4
        self.WARMUP_EPOCHS=2
        self.LR_CYCLE_LENGTH=16
        self.MIN_LR=1e-5
        self.LR_DECAY=0.01
        self.WEIGHT_DECAY = 1e-5
        self.WEIGHTED_SAMPLING = False # 1/sqrt(N)
        self.EDGE_FADE = False
        self.MIN_FADE_MARGIN = 0
        self.MAX_FADE_MARGIN = 0
        self.DEBUG = False
        self.DEBUG_SPEED = None #40 #None #20 #40
        self.CLASS_JOINS = {}
        
        # Parameters below shouldn't need changing
        self.DATA_FOLDER_NM = 'Data'
        self.INPUT_FOLDER_NM = 'Inputs'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.SETTINGS_FOLDER_NM = 'Settings'
        self.RUNS_FOLDER_NM = 'Runs'
        self.IMAGE_FOLDER_NM = 'Cropped_Images'
        self.CROP_LABELS_NM = 'crop_labels.parquet'
        self.TEST_DF_FN = '_test_split.parquet'  #The final label df in the right form for the dataloader
        self.WEIGHTS_FOLDER_SUFFIX = '_weights'
        self.METRICS_FN_SUFFIX = '_monitor_train.png'
        self.BEST_WEIGHTS_FN_SUFFIX = '_best_weights.pt'
        self.RESULTS_DF_SUFFIX = '_df.pkl'
        self.CLASS_NAMES_OUT = '_class_names.json'
        self.RESULTS_FOLDER_NM = 'Results'  # Increment or name this to name the results folder
        self.MODELS_FOLDER_NM = 'Models'


def get_config(settings_pth=None):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = ['USE_CUTMIX', 'USE_MIXUP', 'FOCAL_ALPHA', 'WEIGHTED_SAMPLING', 'FOCAL_ALPHA', 'USE_FOCAL_LOSS',
                    'EPOCHS_BACKBONE_FROZEN', 'WEIGHT_DECAY', 'LEARNING_RATE', 'INITIAL_LR', 'WARMUP_EPOCHS','LR_CYCLE_LENGTH',
                     'MIN_LR', 'DEBUG', 'EDGE_FADE', 'CLASS_JOINS']
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


def get_paths(cfg):
    project_dir = Path(__file__).resolve().parent.parent
    run_id = cfg.RUN_ID if cfg.RUN_ID else datetime.now().strftime('%y_%m_%d_%H')
    paths = {}
    experiment_dir = project_dir / cfg.DATA_FOLDER_NM / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME
    paths['image_dir'] = experiment_dir / cfg.INPUT_FOLDER_NM / cfg.IMAGE_FOLDER_NM
    paths['labels_path'] = experiment_dir / cfg.INPUT_FOLDER_NM / cfg.CROP_LABELS_NM
    results_dir = experiment_dir / cfg.RUNS_FOLDER_NM / run_id / cfg.RESULTS_FOLDER_NM 
    models_dir = experiment_dir / cfg.RUNS_FOLDER_NM / run_id / cfg.MODELS_FOLDER_NM
    weights_pth = models_dir / f'{run_id}{cfg.WEIGHTS_FOLDER_SUFFIX}'
    paths['final_weights_pth'] =  weights_pth / f'{run_id}{cfg.BEST_WEIGHTS_FN_SUFFIX}'
    paths['class_names_pth'] = results_dir / f'{run_id}{cfg.CLASS_NAMES_OUT}'
    paths['test_parquet_pth'] = results_dir / f'{run_id}{cfg.TEST_DF_FN}'
    paths['train_metrics_pth'] = results_dir / f'{run_id}{cfg.METRICS_FN_SUFFIX}'
    paths['pickle_paths'] = {'val_target_df': results_dir / f'{run_id}_val_target{cfg.RESULTS_DF_SUFFIX}',
                    'val_pred_df': results_dir / f'{run_id}_val_pred{cfg.RESULTS_DF_SUFFIX}',
                    'train_target_df': results_dir / f'{run_id}_train_target{cfg.RESULTS_DF_SUFFIX}',
                    'train_pred_df': results_dir / f'{run_id}_train_pred{cfg.RESULTS_DF_SUFFIX}'}
    paths['weights_pth']=weights_pth
    paths['results_dir']=results_dir
    paths['models_dir']=models_dir
    return paths
    
    
def set_hardware(cfg):
    if cfg.EXTRA_CORES is None:
        num_workers = cpu_count()-1
    else:
        num_workers = cfg.EXTRA_CORES
    gpu = torch.cuda.is_available()
    accelerator = 'gpu' if gpu else 'cpu' 
    torch.set_float32_matmul_precision('medium') #could try setting to 'high' at expense of speed
    return num_workers, accelerator 

# -----------------------------------Functions & Classes-----------------------------------------
# -----------------------------------------------------------------------------------------------
class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'
    
def get_training_data(df_path, cfg):
    """Combine classes as per the settings for this training run, into a new 'Target' Column, keeping
    the original 'species' column unchanged as per the original cleaning notebook"""
    df = pd.read_parquet(df_path)
    df['Targets'] = df['Species'].copy()
    
    # Get a dict to handle all name changes by merger
    flattened_list = [(key, value) for key, values in cfg.CLASS_JOINS.items() for value in values]
    inv_merge = {val:key for (key, val) in flattened_list}
    # Add the merger dict, to the name change dict, then do the name change
    df['Targets'] = df['Targets'].replace(to_replace=inv_merge)
    n_reduced = df['Species'].nunique()  - df['Targets'].nunique()
    targets = df['Targets'].unique()
    print(f'Total classes reduced by {n_reduced} due to class mergers')
    return df, targets


def remove_missing_images(dir_pth, dataframe):
    """Checks if any images have gone missing from the crops directory, and if so removes from the dataframe"""
    print(dataframe.head())
    species = set(list(dataframe['Species'].unique()))
    names_in_dir = [str(f.name) for f in Path(dir_pth).rglob('*.*')]
    names_in_df = list(dataframe['Cropped_File_Name'].unique())
    missing_names = list(set(names_in_df) - set(names_in_dir))
    print(f'There are {len(missing_names)} files in the labels dataframe, not found in the image folder.') 
    #print(f'For example: \n {missing_names[:5]}')
    old_length = len(dataframe)
    mask = dataframe['Cropped_File_Name'].isin(missing_names)
    dataframe = dataframe[~mask]
    print(f'{old_length - len(dataframe)} rows were removed from the labels dataframe')
    new_species_list = sorted(list(dataframe['Species'].unique()))
    if set(species) != new_species_list:
        removed_species = species.difference(set(new_species_list))
        print(f'These species were removed from the model as their images were all missing: " {removed_species}')
    return dataframe


def split_data(df, image_dir, test_fraction=0.1, val_fraction=0.1, debug_speed=None):
    df = remove_missing_images(image_dir, df)
    targets_list = list(map(str, list(df['Targets'].unique())))
    targets_list.sort()
    num_classes = len(targets_list)
    print(f'There are {len(df)}, total images in labels dataframe')
    print(f'There are a total of {num_classes} classes')

    total_images = len(df)
    indices = list(range(total_images))

    random.shuffle(indices)
    if debug_speed:
        indices = indices[:int(len(indices)/debug_speed)]

    train_fraction = 1 - test_fraction - val_fraction
    train_sp = int(np.floor(train_fraction * len(indices))) # The training-validation split
    valid_sp = int(np.floor(val_fraction * len(indices))) + train_sp # The validation-test split
    train_idx, val_idx, test_idx = indices[:train_sp], indices[train_sp:valid_sp], indices[valid_sp:]
    
    print(' Training set size: \t', len(train_idx))
    print(' Validation set size: \t', len(val_idx))
    print(' Test set size: \t', len(test_idx))
    print(' Total dataset: \t', total_images)

    all_idx = [train_idx, val_idx, test_idx]
    return [df.iloc[idx] for idx in all_idx], targets_list


def get_new_filepath(df, fldr):
    df.drop('File_Path', axis=1, inplace=True) #remove the original one
    apply_fn = df.apply(lambda row: str(fldr / row['Cropped_File_Name']), axis=1)
    df['File_Path'] = apply_fn
    return df[['Targets', 'File_Path']].copy()


def save_as_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
    return


def encode_df(df, image_fldr, class_list):
    df = get_new_filepath(df, image_fldr)
    df = pd.concat([df, pd.get_dummies(df['Targets'], dtype=int)], axis=1)
    missing_classes = list(set(class_list).difference(list(df.Targets.unique())))
    df[missing_classes] = 0 #Ensures all dataframes have the same columns
    df = df[['Targets','File_Path'] + class_list] ## Ensure all dataframes have columns in the same order
    return df


def get_class_weights(df, weighted_sampling):
    num_unique = df['Targets'].nunique()
    df = df.iloc[:, 2:] # removing the 'filepath' and 'targets' columns
    col_sums = df.sum()
    counts_array = col_sums.values
    
    if weighted_sampling:
        counts_array = np.sqrt(counts_array) 
        focal_weights = counts_array.mean() / counts_array  #uses the sqrt value, to account for the changed distribution
        class_weights = counts_array.tolist()
        sample_idxs = np.argmax(df.values, axis=1).tolist()
        sampling_weights = [1 / class_weights[idx] for idx in sample_idxs] 
    else:
        sampling_weights = None
        focal_weights = counts_array.mean() / counts_array
    print(f'There are {num_unique} unique classes, and the dimensions of the weights arrays are {focal_weights.shape}')
    return focal_weights, sampling_weights


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
    def __init__(self, labels_df, transform=None, edge_fade=False, min_margin=0.05, max_margin=0.05):
        self.df = labels_df  # pd.read_csv(metadata_csv_path)
        self.transform = transform
        self.fade_edges = edge_fade
        self.min_margin = min_margin
        self.max_margin = max_margin
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, image_path, mode):
        try:
            image = cv2.imread(image_path)
            if image is not None:
                if mode == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != (600, 600):
                    print(image_path, image.shape)
                    pass
                return image
            else:
                #print(f"Warning: Unable to load the image at '{image_path}'. Skipping...")
                return None
        except Exception as e:
            pass
            #print(f"Warning: Unable to load the image at '{image_path}'. Error: {str(e)}. Skipping...")
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
        targets = torch.tensor(ohe_vals).float().to(torch.float32)
        return image, targets


def get_transforms():
    INPUT_MEAN = [ 0.485, 0.456, 0.406 ] # mean to be used for normalisation, using values from ImageNet.
    INPUT_STD = [ 0.229, 0.224, 0.225 ] # stddev to be used for normalisation, using values from ImageNet.

    train_transforms = A.Compose([
            A.OneOf([A.RandomFog(p=1), 
                     A.RandomRain(rain_type='torrential', p=1)], p=0.2),
            A.Sequential([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-.75, .1), rotate_limit=30, p=0.8),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
                A.RandomCrop(height=480, width=480, p=1)]),
            A.OneOf([A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1), #got worse when a second one of these (with def vals) was removed 
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
            A.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
            ToTensorV2()])

    val_transforms = A.Compose([
        A.CenterCrop(height=480, width=480, p=1),
        A.Normalize(mean=INPUT_MEAN, std=INPUT_STD), ToTensorV2()])

    image_transforms = {'train':train_transforms,'val':val_transforms}
    return image_transforms


def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

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
    def __init__(self, class_weights=None, alpha=False, gamma=2):  #add something to handle non-cuda situation
        super(FocalLoss, self).__init__() 
        self.weights = torch.FloatTensor(class_weights).cuda() if (class_weights is not None and alpha) else None
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        class_indices = torch.argmax(targets, dim=1).long()
        if self.alpha:
            loss = (self.weights[class_indices] * (1 - pt) ** self.gamma * ce_loss).mean()
        else:
            loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return loss


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
                 loss = nn.CrossEntropyLoss(),
                 lr = 1e-3,
                 weight_decay = 1e-5,
                 checkpoint_path=None,
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

    def forward(self,images):
        logits = self.backbone(images)
        return logits
        
    #def configure_optimizers(self):  #Could be modified to include interesting warmup conditions
        #return get_optimizer(self.lr, params=self.parameters(), weight_decay=self.decay)
    
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

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=self.mixup_alpha)
        y_pred = self(X)
        loss_mixup = mixup_criterion(self.loss_function, y_pred, y_a, y_b, lam)
        return loss_mixup, y_pred

    def training_step(self, batch, batch_idx):
        image, target = batch        
        if self.mixup:
            loss, y_pred = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred,target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        train_output = {"train_loss": loss, "logits": y_pred, "targets": target}
        self.train_outputs.append(train_output) 
        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch     
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
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
        return

    def on_train_epoch_end(self, *args, **kwargs):
            if (self.unfreeze is not None) and (self.current_epoch == self.unfreeze):
                unfrozen_layers = list(self.backbone.children())[-self.unfreeze_layers:]
                
                for layer in unfrozen_layers:
                    if not isinstance(layer, nn.BatchNorm2d):
                        for param in layer.parameters():
                            param.requires_grad = True
                print(Colour.S + f'Unfreezing the top {self.unfreeze_layers} layers of the backbone after {self.current_epoch} epochs' + Colour.E)
    def get_my_metrics_list(self):
        return self.metrics_list


def get_dataloaders(df_train, df_valid, cfg, transforms, sample_weights=None, num_workers=0):
    edge_fade = cfg.EDGE_FADE
    min_margin = cfg.MIN_FADE_MARGIN
    max_margin = cfg.MAX_FADE_MARGIN
    batch_size = cfg.TRAIN_BATCH_SIZE
    epoch_length = cfg.EPOCH_LENGTH
    
    ds_train = PredatorDataset(df_train, transforms['train'], edge_fade=edge_fade, min_margin=min_margin, max_margin=max_margin) 
    ds_val = PredatorDataset(df_valid, transforms['val'], edge_fade=edge_fade, max_margin=min_margin) #Fix the max margin to the minimum for val/test
    p_workers = True if num_workers > 0 else False
    if sample_weights is not None:
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=epoch_length)
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers) 
        pass  
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, persistent_workers=p_workers,
                              shuffle=True, num_workers=num_workers, pin_memory=True) 
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers = num_workers)   
    return dl_train, dl_val, ds_train, ds_val


def run_training(weights_dir, 
                 dl_train,
                 dl_val,
                 logger=None,   
                 epochs=16, 
                 patience=2,
                 loss_function = nn.CrossEntropyLoss(),
                 model=None
                 ):
    print(f"Running training...")
    logger = None
    
    early_stop_callback = EarlyStopping(monitor="val_loss", 
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

    callbacks_to_use = [checkpoint_callback, early_stop_callback] #MonitoringCallback   # work more on this later.
    
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
    print(f'using model {model_name} for evaluation')
    best_model_state_dict = torch.load(model_pth)['state_dict']
    best_model = CustomModel(class_list, model_name=model_name, custom_head=custom_head)
    best_model.load_state_dict(best_model_state_dict)
    best_model.eval()
    return best_model


def check_best_model(best_model, test_df, use_gpu, transforms, batch_size=16, num_batches=8):
    test_ds = PredatorDataset(test_df, transforms['val'])
    loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    print(f'Evaluating {batch_size*num_batches} example test images')
    correct = 0
    total_preds= 0
    pbar = tqdm(total=batch_size*num_batches)

    start_time = time.time()
    for _ in range(num_batches):
        images, targets = next(iter(loader))
        if use_gpu:
            images, targets, best_model = images.cuda(), targets.cuda(), best_model.cuda()
        logits = best_model(images)
        probs = F.softmax(logits, dim=1)
        for probs, target in zip(probs, targets):
            if torch.argmax(target) == torch.argmax(probs):
                correct += 1
            total_preds += 1
        pbar.update(len(images))
    pbar.close()
    total_time = time.time()-start_time
    accuracy = correct / total_preds
    print(f'There were {correct} correct predictions from {total_preds} samples.  A mean accuracy of {accuracy:.2f}')
    print(f'Processed {total_preds} test samples in {total_time:.2f} seconds')
    print(f'That is a mean of {total_preds/total_time:.2f} images per second')
    del best_model, images, targets, logits, probs
    return


def plot_train_metrics(metrics, save_path):
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


# ----------------------------------- Main Function---------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_path=None):  # "E:\Project\Settings\Exp_05_Run_13.yaml"
    cfg = get_config(settings_path)
    paths = get_paths(cfg)
    num_workers, accelerator  = set_hardware(cfg)
    image_dir = paths['image_dir']
    print(Colour.S + f'Training with settings from settings file: {cfg.EXPERIMENT_NAME}, run: {cfg.RUN_ID}' + Colour.E)

    #setup steps that only need doing once
    start_time = time.time()
    num_files = sum(1 for entry in os.scandir(image_dir) if entry.is_file())
    print(f'Loading data with {num_workers + 1} CPU cores')
    gpu = (accelerator =='gpu')
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if gpu else 'CPU'))
    print(f'Label data location: {str(paths["labels_path"])}')
    print(f'{num_files} files found in the image folder {str(image_dir)}')
    for fldr in [paths['results_dir'], paths['models_dir']]:
        if not os.path.exists(fldr):
            os.makedirs(fldr)
    if gpu:
        gc.collect()
        torch.cuda.empty_cache()
    pl.seed_everything(cfg.RANDOM_SEED, workers=True)

    #Training 
    in_df, targets = get_training_data(paths['labels_path'], cfg)  #maybe this is a better place for the targets list?  check the format first
    debug_speed = cfg.DEBUG_SPEED
    splits, target_list  = split_data(in_df, image_dir, debug_speed=debug_speed) 
    print(target_list)
    save_as_json(target_list, paths['class_names_pth'])
    train_df, val_df, test_df = [encode_df(_df, image_dir, target_list) for _df in splits]
    test_df.to_parquet(paths['test_parquet_pth'])
    focal_weights, sampling_weights = get_class_weights(train_df, cfg.WEIGHTED_SAMPLING)
    image_transforms = get_transforms()
    dl_train, dl_val, _, _ = get_dataloaders(train_df, val_df, cfg, image_transforms, sampling_weights, num_workers)
    logger = CSVLogger(save_dir=paths['results_dir'], name=cfg.RUN_ID)
    if cfg.LOSS_FUNCTION == 'FocalLoss':
        loss_instance = FocalLoss(focal_weights, cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA)
    else:
        loss_instance = nn.CrossEntropyLoss()
    
    custom_head = get_custom_head(cfg.HEAD_NAME)
    training_model = CustomModel(target_list,
                        loss=loss_instance,
                        lr = cfg.LEARNING_RATE,
                        weight_decay=cfg.WEIGHT_DECAY,
                        checkpoint_path=paths['weights_pth'],
                        unfreeze=cfg.EPOCHS_BACKBONE_FROZEN,
                        unfreeze_lyrs=cfg.UNFREEZE_LAYERS,
                        pickle_path_dict=paths['pickle_paths'],
                        mixup_alpha = cfg.MIXUP_ALPHA,
                        use_mixup=cfg.USE_MIXUP,
                        model_name=cfg.MODEL_NAME,
                        custom_head=custom_head,
                        initial_lr=cfg.INITIAL_LR,
                        warmup_epochs=cfg.WARMUP_EPOCHS,
                        cycle_length=cfg.LR_CYCLE_LENGTH,
                        min_lr=cfg.MIN_LR,
                        lr_decay=cfg.LR_DECAY,
                        )

    metrics, best_model_path = run_training(paths['weights_pth'], 
                                    dl_train,
                                    dl_val,
                                    logger=logger,
                                    epochs = cfg.MAX_EPOCHS,
                                    patience=cfg.PATIENCE,
                                    loss_function= loss_instance,
                                    model=training_model                               
                                    )

    end_time = time.time()
    print(f'Processing completed in {end_time-start_time:.2f} seconds')
    plot_train_metrics(metrics, paths['train_metrics_pth'])

    #Check and save the final model
    best_model = get_best_model(target_list, best_model_path, cfg.MODEL_NAME, custom_head=custom_head)
    torch.save(best_model.state_dict(), paths['final_weights_pth'])
    print(f'Final model saved to {paths["final_weights_pth"]}')
    
    check_best_model(best_model, test_df, gpu, image_transforms, cfg.BATCH_SIZE)
    del best_model
    
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()