from pathlib import Path
import pandas as pd
import random
import yaml
import h5py
from prioritise_data import (check_duplicate_paths,
                             remove_duplicates_by_hash,
                             limit_with_embeddings,
                             limit_with_bbox_vals,
                             limit_randomly,
                             )

class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'

class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_40'
        #self.SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
        self.CLASSES = [] #leave empty, will populate from folder names
        self.CLASSES_TO_EXCLUDE = ['shag', 'bat', 'moth', 'skink', 'lizard',  'skylark', 'grey_duck', 'empty', 'campbell_island_teal'] 
        self.CLASS_JOINS = {'lizard':['skink', 'lizard'], 'finch':['greenfinch', 'goldfinnch', 'chaffinch'], 'quail':['quail_california', 'quail_brown']}
        self.CLASS_NAME_CHANGE = {'penguin':'little_blue_penguin', 'song thrush':'thrush', 'NZ_falcon':'nz_falcon'}
        self.LOCATIONS_TO_EXCLUDE = []
        self.LOCATIONS_FOR_TEST_ONLY =  ['N01', 'BWS', 'EBF', 'EM1', 'ES1'] + ['EL1', 'ES1', 'N02', 'N04', 'N06', 'N08', 'OTR', 'WRO']
        self.MD_THRESHOLD_TO_TRAIN_WITH = 0.5
        self.LOW_CONF = []

        #Parameters for different methods of selecting images for training
        self.MIN_HASH_DIFFERENCE = 50  #Below this will be considered a duplicate and rejected
        self.MAX_PER_CLASS_PER_CAMERA = 200 #Maximum images from a particular animal category, from a given camera
        self.MAX_PER_CLASS_PER_LOCATION = 3000
        self.prioritising_methods = ['remove_duplicates_by_hash',  'embedding_limit_cam', 'embedding_limit_location'] 
        self.rehash_if_new_images = True

    #[
    #'remove_duplicates_by_hash',
    #'bbox_limit_location',
    #'rand_limit_cam'
    #'rand_limit_location'
    #'embedding_limit_cam'
    #'embedding_limit_location'
    #]

class Paths:
    SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
    DATA_FOLDER_NM = 'data'
    SETTINGS_FOLDER_NM = 'settings'
    EXPS_FOLDER_NM = 'experiments'
    LABELS_FROM_JSON_NM = 'all_labels.parquet'
    CLEANED_LABELS_NM = 'cleaned_labels.parquet'
    INPUT_FOLDER_NM = 'inputs'
    MD_PREVIOUS = 'MD_Last_Run'
    IMG_HASH_PTH = 'image_hashes.h5'
    IMG_EMBED_PTH = 'image_embeddings.h5'
    HASH_PAIR_DIST_PTH = 'hash_pair_distances.h5'

    def __init__(self, experiment_name):
        _project_folder = Path(__file__).resolve().parent.parent
        self.experiment_folder = _project_folder / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / experiment_name
        self.in_pth =  self.experiment_folder / self.INPUT_FOLDER_NM / self.LABELS_FROM_JSON_NM
        self.out_pth = self.experiment_folder / self.INPUT_FOLDER_NM / self.CLEANED_LABELS_NM
        self.hash_path = _project_folder / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / self.MD_PREVIOUS / self.IMG_HASH_PTH
        self.embedding_path = _project_folder / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / self.MD_PREVIOUS / self.IMG_EMBED_PTH
        self.hash_pair_dist = _project_folder / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / self.MD_PREVIOUS / self.HASH_PAIR_DIST_PTH


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


def get_config(settings_pth: str):
    #This seems unreliable,  it wasn't updating the locations_to_exclude
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = ['CLASSES', 'CLASSES_TO_EXCLUDE', 'CLASS_JOINS', 'CLASS_NAME_CHANGE', 
                     'LOCATIONS_TO_EXCLUDE', 'LOCATIONS_FOR_TEST_ONLY', 'LOW_CONF','MIN_HASH_DIFFERENCE',
                     'MAX_PER_CLASS_PER_CAMERA', 'MAX_PER_CLASS_PER_LOCATION', 'prioritising_methods']
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

def format_df(df: pd.DataFrame):
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


# ----------------------------------- Main Script-----------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_pth = None):
    random.seed(2023)
    cfg = get_config(settings_pth)
    img_cfg = ImageConfig()
    paths = Paths(cfg.EXPERIMENT_NAME)

    in_df = pd.read_parquet(paths.in_pth)
    print(f'Reading {len(in_df)} image labels from the original parquet file')
    n_unknown = len(in_df[in_df['Species'] == 'unknown'])
    n_classes = in_df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes} original classes')
    print(f'{n_unknown} original rows with the [unknown] class')
    df = format_df(in_df)

    print(df['Camera'].value_counts())
    print(df['Camera'].nunique())
    #Remove unwanted datasets or classes
    df = df[~(df['Location'].isin(cfg.LOCATIONS_TO_EXCLUDE))]
    n_classes_2 = df['Species'].nunique() - (n_unknown!=0)
    print(f"The unique classes after removing unwanted locations are {df['Species'].unique()}")
    df = df[~(df['Species'].isin(cfg.CLASSES_TO_EXCLUDE))]
    print(f'{len(df)} lines left after removing unwanted locations and classes')
    print(f"The unique classes after removing unwanted classes are {df['Species'].unique()}")

    #Fix class names
    for key, value in cfg.CLASS_NAME_CHANGE.items():
        df.replace(key, value, inplace=True)
    n_classes_3 = df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes_2- n_classes_3} unique classes removed by name changes')
    print(f"The unique classes after name changes are {df['Species'].unique()}")

    #Remove low scoring MD predictions, but not the ones from the LOW_CONF list
    df = df[(df['Confidence'] >= cfg.MD_THRESHOLD_TO_TRAIN_WITH) | (df['Species'].isin(cfg.LOW_CONF))]
    print(f'{len(df)} lines left after removing low scoring MegaDetector predictions')

    df = df.copy().reset_index(drop=True)
    df = check_duplicate_paths(df)

    #so we want to recalculate the hash distances only if this has changed since last time.


    recalculate = cfg.rehash_if_new_images  #This is a workaround, should be =True, but my comparison logic isn't working as expected.
    if paths.hash_path.exists() and paths.hash_pair_dist.exists():
        original_length = len(df)
        print(f'The original length of the image dataset before removing duplicates is {original_length}')
        with h5py.File(paths.hash_path, "r") as f:
            length = f["keys"].shape[0]  # Get the length without loading data
        if length == original_length:
            recalculate = False
            print("Length of hash dataset is:", length)
            print('Not recalculating hash pairs as the revious hashing included all images')

    

    if 'remove_duplicates_by_hash' in cfg.prioritising_methods:
        print(f"The unique classes before removal by hash are {df['Species'].unique()}")
        df, duplicate_pairs = remove_duplicates_by_hash(df,
                                                h5_path=paths.hash_path,
                                                hash_dist_h5_path = paths.hash_pair_dist,
                                                cols=['Species','Camera'],
                                                min_distance=cfg.MIN_HASH_DIFFERENCE,
                                                recalculate=recalculate,
                                                verbose=True,
                                                )
        print(f"The unique classes after removal by hash are {df['Species'].unique()}")

    if 'bbox_limit_location' in cfg.prioritising_methods:
        df = limit_with_bbox_vals(df,
                                  cols = ['Species', 'Location'],
                                  limit=cfg.MAX_PER_CLASS_PER_LOCATION)
    if 'rand_limit_cam' in cfg.prioritising_methods:
        df=limit_randomly(df,
                          cols=['Species','Camera'],
                          limit=cfg.MAX_PER_CLASS_PER_CAMERA)
        print(f'{len(df)} lines left after limiting to {cfg.MAX_PER_CLASS_PER_CAMERA} images per class-camera')
    if 'rand_limit_location' in cfg.prioritising_methods:
        df=limit_randomly(df,
                          cols=['Species','Location'],
                          limit=cfg.MAX_PER_CLASS_PER_LOCATION)
        print(f'{len(df)} lines left after limiting to {cfg.MAX_PER_CLASS_PER_LOCATION} images per class-location')
    if 'embedding_limit_cam' in cfg.prioritising_methods:
        df = limit_with_embeddings(df,
                                   h5_path=paths.embedding_path,
                                   cols=['Species','Camera'],
                                   img_cfg=img_cfg,
                                   limit=cfg.MAX_PER_CLASS_PER_CAMERA)
        print(f'{len(df)} lines left after limiting to {cfg.MAX_PER_CLASS_PER_CAMERA} images per class-camera')
    
    print(f"The unique classes after removal by camera-embedding are {df['Species'].unique()}")
    if 'embedding_limit_location' in cfg.prioritising_methods:
        df = limit_with_embeddings(df,
                                   h5_path=paths.embedding_path,
                                   cols=['Species','Location'],
                                   img_cfg=img_cfg,
                                   limit=cfg.MAX_PER_CLASS_PER_LOCATION)
        print(f'{len(df)} lines left after limiting to {cfg.MAX_PER_CLASS_PER_LOCATION} images per class-location')
        print(f"The unique classes after removal by location-embedding are {df['Species'].unique()}")
    df = df.drop('Date_Time_Object', axis=1)
    df.to_parquet(paths.out_pth)
    print(f'{len(df)} rows written to the cleaned parquet file')
    n_classes = df['Species'].nunique() - (n_unknown!=0)
    print(f'{n_classes} final unique species were left (not counting [unknown] class)')
    print(f"The final classes immediately after cleaning steps are: {df['Species'].unique()}")
    print(f'The crop annotation file for training saved to {paths.out_pth}')

    num_rows = len(df)
    random_indices = random.sample(range(num_rows), min(num_rows, 200))
    random_selection = df.iloc[random_indices]
    print(random_selection.head())
    random_selection.to_csv(paths.experiment_folder / paths.INPUT_FOLDER_NM / 'random_sample.csv', escapechar='\\')
    print(Colour.S + '\nA random selection of the dataframe after data cleaning steps:\n' + Colour.E)
    print(random_selection.head())

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(settings_pth='/media/olly/Red_SSD/Alita/Settings/Exp_500_Run_01.yaml')