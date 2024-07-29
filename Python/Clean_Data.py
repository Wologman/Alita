from pathlib import Path
import pandas as pd
import random
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np


class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_01'
        self.SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
        self.CLASSES = [] #leave empty, will populate from folder names
        self.CLASSES_TO_EXCLUDE = ['shag', 'moth', 'swallow', 'grey_faced_petrol', 'fluttering_shearwater', 'whitehead', 'skink', 'lizard', 
                                   'fernberd', 'skylark', 'grey_duck', 'long_tailed_cuckoo', 'spotted_dove', 'nz_falcon', 'mohua', 'kingfisher',
                                   'fiordland_crested_penguin', 'stilt', 'rosella', 'swan', 'grey_warbler ']
        self.CLASS_JOINS = {'lizard':['skink', 'lizard'], 'finch':['greenfinch', 'goldfinnch', 'chaffinch'], 'quail':['quail_california', 'quail_brown']}
        self.CLASS_NAME_CHANGE = {'penguin':'little_blue_penguin', 'song thrush':'thrush', 'NZ_falcon':'nz_falcon'}
        self.LOCATIONS_TO_EXCLUDE = []
        self.LOCATIONS_FOR_TEST_ONLY =  ['N01', 'BWS', 'EBF', 'EM1', 'ES1']
        self.MD_THRESHOLD_TO_TRAIN_WITH = 0.5
        self.LOW_CONF = []
        self.IMAGE_LIMIT = 20000 # Maximum number of images of a given class at a particular location code
        self.MAX_CLASS_PER_CAMERA = 250 #Maximum images from a particular animal category, from a given camera
        self.LOW_CLASS_PER_CAMERA = 50 #Groups over this will throw away half the extra images above this value up to MAX_CLASS_PER_CAMERA
        self.FILTER_METHOD = 'random'
        
        #Parameters that shouldn't need changing below
        self.DATA_FOLDER_NM = 'Data'
        self.SETTINGS_FOLDER_NM = 'Settings'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.LABELS_FROM_JSON_NM = 'all_labels.parquet'
        self.CLEANED_LABELS_NM = 'cleaned_labels.parquet'
        self.DATA_FOLDER_NM = 'Data'
        self.INPUT_FOLDER_NM = 'Inputs'

def get_config(settings_pth):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = ['CLASSES', 'CLASSES_TO_EXCLUDE', 'CLASS_JOINS', 'CLASS_NAME_CHANGE', 
                     'LOCATIONS_TO_EXCLUDE', 'LOCATIONS_FOR_TEST_ONLY', 'LOW_CONF']
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

def format_df(df):
    if 'Description' in df.columns:
        if not df['Description'].isnull().all():
            df['Camera'] = df['Description'].str.split('__', expand=True).apply(lambda x: '-'.join(x.dropna()[:2]), axis=1)
    else:
        df['Camera'] = df['Location']

    if 'Date_Time' in df.columns:  
        df['Date_Time_Object'] = pd.to_datetime(df['Date_Time'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
        mean_datetime = df['Date_Time_Object'].mean()
        df['Date_Time_Object'].fillna(mean_datetime, inplace=True)
    return df


def get_distinct_images(df, limit):
    """Uses k-means clustering together with features from the exif data and megadetector
    in order to subset only the N most distinctive images from a given class+camera location"""
    df = df.copy()
    scaler = MinMaxScaler()
    kmeans = KMeans(n_clusters=limit, n_init=10)
    reference_time = pd.to_datetime('2023-01-01 00:00:00')
    df['Sec_Since_Ref'] = (df['Date_Time_Object'] - reference_time).dt.total_seconds()
    features = ['Sec_Since_Ref', 'Confidence', 'x_min', 'y_min', 'Width', 'Height']
    data = df[features].values 
    data = scaler.fit_transform(data)
    _ = kmeans.fit_predict(data)
    indices = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)[0]
    #print(len(indices))
    return indices


def remove_extras(df2,  limiting_col, feature_name, method='random', lower_limit=50, upper_limit=250):
    """
    Args:
        limit (int): The maximum to be allowed for this grouping
        limiting_col (str): The column header
        feature_name (str): The more descriptive name the column represents
        method: 'random' to remove the extras randomly, or 'difference' keep the most distinctive images
    Returns:
        dataframe : Dataframe with no more than the allowe limit for each grouping
        groups with excess lines removed randomly
    """
  
    def remove_overs(row):
        limiter = getattr(row, limiting_col)
        num_img = row.count
        obs_cls = row.Species
        all_idxs = df2[(df2[limiting_col] == limiter) & (df2['Species'] == obs_cls)].index.to_list()
        if method == 'difference':
            group_df = df2.iloc[all_idxs]
            limit = lower_limit + np.clip((num_img-lower_limit)//2, 0,upper_limit-lower_limit)
            keep_idxs = get_distinct_images(group_df, limit)
        else:
            keep_idxs = random.sample(all_idxs, lower_limit)
        return keep_idxs
    
    grouped_counts = df2.groupby([limiting_col, 'Species']).size().reset_index(name='count')
    under_lim = grouped_counts[grouped_counts['count'] < lower_limit]
    over_lim = grouped_counts[grouped_counts['count'] >= lower_limit]

    description = f'Processing labels under max {feature_name}-class limit'
    combined_groups = under_lim['Species'] + under_lim[limiting_col]
    
    df2['combined'] = df2['Species'] + df2[limiting_col]
    under_max_lim_df = df2[(df2['combined'].isin(combined_groups))]
    print(f'there are {len(under_max_lim_df)} images that will be kept from under camera-limit combinations')

    description = f'Processing labels over max {feature_name}-class limit'
    iterate_list = list(over_lim.itertuples())
    nested_list = Parallel(n_jobs=-1)(delayed(remove_overs)(row) for row in tqdm(iterate_list, desc=description))
    indices_to_keep = [item for sublist in nested_list for item in sublist]
    print(f'there are {len(indices_to_keep)} indices that will be kept from over-limit combinations')
    max_limit_df = df2.iloc[indices_to_keep]
    new_df = pd.concat([under_max_lim_df, max_limit_df])
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# ----------------------------------- Main Script-----------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_pth = None):
    random.seed(2023)
    cfg = get_config(settings_pth)
    project_folder = Path(__file__).resolve().parent.parent
    experiment_folder = project_folder / cfg.DATA_FOLDER_NM / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME
    in_pth =  experiment_folder / cfg.INPUT_FOLDER_NM / cfg.LABELS_FROM_JSON_NM
    out_pth = experiment_folder / cfg.INPUT_FOLDER_NM / cfg.CLEANED_LABELS_NM

    in_df = pd.read_parquet(in_pth)
    print(f'Reading {len(in_df)} image labels from the original parquet file')
    n_unknown = len(in_df[in_df['Species'] == 'unknown'])
    n_classes = in_df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes} original classes')
    print(f'{n_unknown} original rows with the [unknown] class')
    #in_df = remove_old_filepaths(image_root_pth, in_df)
    df = format_df(in_df)
    #Remove unwanted datasets or classes
    df = df[~(df['Location'].isin(cfg.LOCATIONS_TO_EXCLUDE))]
    n_classes_2 = df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes-n_classes_2} classes were deliberatly excluded')
    df = df[~(df['Species'].isin(cfg.CLASSES_TO_EXCLUDE))]
    print(f'{len(df)} lines left after removing unwanted locations and classes')

    #Fix class names
    for key, value in cfg.CLASS_NAME_CHANGE.items():
        df.replace(key, value, inplace=True)
    n_classes_3 = df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes_2- n_classes_3} unique classes removed by name changes')

    #Remove low scoring MD predictions, but not the ones from the LOW_CONF list
    df = df[(df['Confidence'] >= cfg.MD_THRESHOLD_TO_TRAIN_WITH) | (df['Species'].isin(cfg.LOW_CONF))]
    print(f'{len(df)} lines left after removing low scoring MegaDetector predictions')

    #Limit any one class to a maximum of MAX_CLASS_PER_CAMERA images for each camera trap.
    print(df.head())
    df = df.copy().reset_index(drop=True)
    df = remove_extras(df, 'Camera', 'Camera', method=cfg.FILTER_METHOD, lower_limit=cfg.LOW_CLASS_PER_CAMERA,  upper_limit=cfg.MAX_CLASS_PER_CAMERA,)
    print(f'{len(df)} lines left after limiting to {cfg.IMAGE_LIMIT} images per class-camera combination')

    #Limit any one class to a maximum of IMAGE_LIMIT images for each dataset.
    df = remove_extras(df, 'Location', 'Dataset', method='random', lower_limit=cfg.IMAGE_LIMIT, upper_limit=cfg.IMAGE_LIMIT)
    print(f'{len(df)} lines left after limiting to {cfg.IMAGE_LIMIT} images per class-location combination')

    df = df.drop('Date_Time_Object', axis=1)
    df.to_parquet(out_pth)
    print(f'{len(df)} rows written to the cleaned parquet file')
    n_classes = df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes} final unique species were left (not counting [unknown] class)')
    print(f'The crop annotation file for training saved to {out_pth}')

    num_rows = len(df)
    random_indices = random.sample(range(num_rows), 500)
    random_selection = df.iloc[random_indices]
    random_selection.to_csv(experiment_folder / cfg.INPUT_FOLDER_NM / 'random_sample.csv')
    print(random_selection.head())
    print(df.describe())

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()