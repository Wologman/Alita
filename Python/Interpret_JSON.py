'''If running directly, 

This script is intended to go through any json files produced by the MegaDetector, and also get the class name from the filename, 
and produce a single parquet file with all the info needed for later cropping

It could be improved by re-using this parquet the next time, and only processing new .json

The output columns will be: ['File_Path', 'Mega_Class', 'Confidence', 'x_min', 'y_min', 'Width', 'Height', 'Species', 'Location'] 
The coordinates are relative to the image on [0,1] (normalised COCO).

Also works when there is no class name in the file structure, so that it can be used for inferance.  The 'Species' will then be 'unknown'
'''
import json
from pathlib import Path
import operator
import pandas as pd
import re
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import concurrent.futures
from PIL import Image
from PIL.ExifTags import TAGS

class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'MD_Last_Run'  #Must match the folder name under the Data folder
        self.BEST_ONLY = True #If true, only the most probable prediction from each image is kept for training
        self.CLASSES_FROM_DIR_NMS = True
        self.CLASSES = ['mouse','robin','possum','stoat','cat','rat','thrush','kea','blackbird','wallaby','tomtit','cow',
                        'sheep','human','rifleman','kiwi','rabbit','deer','weka','parakeet','ferret','hare','pukeko','harrier',
                        'bellbird','hedgehog','chaffinch','dunnock','sealion','weasel','pipit','yellow_eyed_penguin','magpie',
                        'myna','quail','greenfinch','yellowhammer','pig','kereru','tui','starling','sparrow','silvereye','fantail',
                        'dog','moth','goat','pateke','banded_rail','oystercatcher','black_fronted_tern','paradise_duck','mallard',
                        'morepork','goldfinch','chamois','redpoll','takahe','kaka','shore_plover','canada_goose','spurwing_plover',
                        'tieke','white_faced_heron','lizard','shag','black_backed_gull','little_blue_penguin','brown_creeper',
                        'black_billed_gull','crake','skylark','pheasant','skink','grey_warbler','swan','fernbird','banded_dotterel',
                        'rosella','fiordland_crested_penguin','pied_stilt','mohua','long_tailed_cuckoo','kingfisher','nz_falcon',
                        'grey_duck','spotted_dove','swallow'] + ['penguin', 'song thrush', 'bell', 'browncreeper', 'kakariki', 
                        'mice', 'tahr','waxeye', 'whio']
        self.SOURCE_IMAGES_PTH = 'Z:\\alternative_footage\\CLEANED'
        self.INDEPENDENT_TEST_ONLY = ['N01', 'BWS', 'EBF', 'EM1', 'ES1']
        self.UPDATE_EXIF = True
        self.UPDATE_META_DATA = True
        
        #Atributes that should not need changing below
        self.LAST_MD_FOLDER_NM = 'MD_Last_Run'
        self.LABELS_FROM_JSON_NM = 'all_labels.parquet' #Output label file.
        self.EXIF_DATA_NM = 'last_exif_data.parquet' #Output exif data file
        self.DATA_FOLDER_NM = 'Data'  #Directory for all data folders
        self.EXPS_FOLDER_NM = 'Experiments'
        self.INPUT_FOLDER_NM = 'Inputs'

def get_config(settings_pth):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = ['CLASSES', 'CLASSES_FROM_DIR_NMS', 'UPDATE_EXIF', 'UPDATE_META_DATA'] #any attributes with values that need evaluating from strings
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
    """Sets up filepaths for this script using relative paths from the parent dir"""
    project_dir = Path(__file__).resolve().parent.parent
    data_folder = project_dir / cfg.DATA_FOLDER_NM 
    output_path = data_folder / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME / cfg.INPUT_FOLDER_NM / cfg.LABELS_FROM_JSON_NM
    output_path_last_run = data_folder / cfg.EXPS_FOLDER_NM / cfg.LAST_MD_FOLDER_NM / cfg.LABELS_FROM_JSON_NM
    exif_pth =data_folder / cfg.EXPS_FOLDER_NM / cfg.LAST_MD_FOLDER_NM  / cfg.EXIF_DATA_NM
    json_file_folder = data_folder / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME / cfg.INPUT_FOLDER_NM
    json_paths = [f for f in json_file_folder.glob('*.json')]
    print(f'json files found in {json_file_folder}',json_paths)
    dataset_pth = Path(cfg.SOURCE_IMAGES_PTH)
    return output_path, exif_pth, output_path_last_run,   json_paths, dataset_pth


def iterate_json(jsons, best_only, data_pth_root):
    """Iterates through a single json file, extracts the useful values 
    and returns a list of lists, one list per image file"""
    print(f'The root path of the image folder is {data_pth_root}')
    def add_root_if_missing(fpath):
        fpath = Path(fpath)
        if not fpath.is_absolute():
            fpath = Path(data_pth_root) / fpath
        return str(fpath)
    
    for image in jsons:
        detection_list = []
        confidence_list = []
        file_path = image.get('file')
        detections = image.get('detections')
        file_path = add_root_if_missing(file_path)    

        if detections is not None:
            for detect in detections:
                predicted_class = int(detect['category'])
                confidence = detect['conf']
                confidence_list.append(confidence)
                left = detect['bbox'][0]  #x_min (normalised to image width)
                top = detect['bbox'][1]  #y_min (normalised to image height)
                width = detect['bbox'][2]  #box width  (normalised to image width)
                height = detect['bbox'][3]  #box height (normalised to image height) 
                detection_list.append([file_path, predicted_class, confidence, left, top, width, height])
            if detection_list:
                if best_only:
                    best_index, _ = max(enumerate(confidence_list), key=operator.itemgetter(1), default=(0, 0))
                    detection_list = [detection_list[best_index]]
            else: detection_list = [[file_path, -1, -1, 0, 0, 1, 1]] #This is for when detections is an empty list, it happens.
        else:
            detection_list = [[file_path, -1, -1, 0, 0, 1, 1]]
        yield detection_list  # detection_list: A list of lists, with only one list if BEST_ONLY=True


def get_class_list(grandparent_dir):
    """Takes the dataset root returns the foldernames of all the grandchildren, which should be the class names"""
    parents = [folder for folder in grandparent_dir.iterdir() if folder.is_dir()]

    master_list=[]
    for parent in parents:
        one_list = [folder.name for folder in parent.iterdir() if folder.is_dir()]
        master_list = master_list + one_list
    sorted_list = sorted(list(set(master_list)))
    print(f'{len(sorted_list)} unique species found from folders: {sorted_list}')
    return sorted_list


def get_class_location(filepath, classes):
    """Parses the file path and returns any strings matching the class list, and 3 letter upper case strings
    that are the unique identifier for each camera location"""
    folder_names = re.split(r'[\\/]', filepath)
    num_locations = 0
    for name in folder_names:
        if len(name) == 3 and any(char.isupper() for char in name):
            num_locations +=1
            location = name
    if num_locations != 1:
        location = 'unknown'

    lowered_names = [f.lower() for f in folder_names] # In case a folder has accidental capitals
    if classes:
        class_name = next(iter(set(classes).intersection(set(lowered_names))), 'unknown')
    else:
        class_name = lowered_names[folder_names.index(location) + 1]
    return class_name, location


def load_json(json_path):
    """Opens a single json file and loads into an array of dictionaries, returns that array
    each dict has the keys 'file', 'max_detection_conf', 'detections' """
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
        json_array = json_dict['images']
        if __name__ == '__main__':
            print("Number of images in json array", len(json_array))
    return json_array


def process_json_array(json_path, classes, image_folder, best_only=True):
    """Iterates through each item of image data, extract the bits of interest, returns a dataframe
    #each image has potentially several crops in a list if BEST_ONLY=False"""
    OUT_COLUMNS = ['File_Path', 'Mega_Class', 'Confidence', 'x_min', 'y_min', 'Width', 'Height', 'Species', 'Location']
    j_array = load_json(json_path)
    data_rows = []
    if __name__ == '__main__': 
        pbar = tqdm(total=len(j_array), desc="Processing JSON file items")
    for detects in iterate_json(j_array, best_only, image_folder):
        if detects:
            for one_thing in detects:
                fpath = one_thing[0]
                observed_class, cam_location = get_class_location(fpath, classes)
                new_row = one_thing + [observed_class, cam_location]
                data_rows.append(new_row)

        if __name__ == '__main__':
            pbar.update(1)
    if __name__ == '__main__':
        pbar.close()
    df = pd.DataFrame(data_rows, columns=OUT_COLUMNS)
    return df

def process_all_jsons(json_path_list, classes, image_folder=None, best_only=True):
    """Iterates throguh a list of json files and appends the result from each to a single dataframe which is returned"""
    #OUT_COLUMNS = ['File_Path', 'Mega_Class', 'Confidence', 'x_min', 'y_min', 'Width', 'Height', 'Species', 'Location']
    #dataframe = pd.DataFrame(columns=OUT_COLUMNS)
    first_iteration = True
    for json_path in json_path_list: 
        df = process_json_array(json_path, classes, image_folder, best_only)
        unknown_rows = df[df['Species'] == 'unknown']
        if __name__ == '__main__':
            print('Unknown species:')
            print(unknown_rows.head(3))
        if first_iteration:
            dataframe= df.copy()
            first_iteration = False
        else:
            dataframe = pd.concat([dataframe,df])

    if __name__ == '__main__': 
        unknown_class = dataframe['Species'].value_counts().get('unknown', 0)
        unknown_place = dataframe['Location'].value_counts().get('unknown', 0)
        species = dataframe['Species'].nunique() - (unknown_class!=0)
        places = dataframe['Location'].nunique() - (unknown_place!=0)
        print('The first three rows of the dataframe extracted from the MegaDetector predictions:')
        print(dataframe.head())
        print(f'{species} Unique species found')
        print(f'{places} Unique dataset location folders found')
        print(f'{unknown_class} Entries that had an unknown class')
        print(f'{unknown_place} Entries that had an unknown location')
        print(f'{len(dataframe)} rows from the MegaDetector Runs')
    
    return dataframe


def remove_missing_dirs(root, df):
    """This compares a list of location + class, from the source directory, with the same from
    the final dataframe (generated by the MegaDetector), and removes any rows where those folders 
    are missing.  Necessary if folders have been removed since the MegaDetector was run
    Saves time for the next step, which looks at individual files."""
    loc_spec_lst = df.groupby(['Location', 'Species']).size().reset_index()[['Location', 'Species']].agg(tuple, axis=1).tolist()
    print('Checking by folder name if any instances from the MegaDetector have been removed')
    folders_to_keep = []
    for item in tqdm(loc_spec_lst):
        location = item[0]
        species = item[1]
        dir_path = root / location / species
        if dir_path.exists() and dir_path.is_dir():
            folders_to_keep.append(item)

    tuples_df = pd.DataFrame(folders_to_keep, columns=['Location', 'Species'])
    filtered_df = df.merge(tuples_df, on=['Location', 'Species'], how='inner')
    print(f'{len(df) - len(filtered_df)} rows were removed from the dataframe')
    return filtered_df


def remove_old_filepaths(root_dir, df):
    """Remove from the dataframe any old filepaths that have been deleted or removed 
       from the source directory
    Args:
        root_dir (path object): path to the root directory to be searched
        df (dataframe): dataframe with filepaths as path objects
    Returns:
        dataframe: dataframe with the lines matching missing filepaths removed
    """
    tqdm.pandas()
    def string_to_path(x):
        return Path(x)
    df['File_Path_Objects'] = df['File_Path'].progress_apply(string_to_path)
    print('Getting a unique list from the new file path object column')
    df_fnames = set(df['File_Path_Objects'].unique().tolist())
    subdirs = [entry for entry in root_dir.iterdir() if entry.is_dir()]

    def search_for_jpgs(folder):
        jpgs = {path for path in folder.rglob('*.[jJ][pP][gG]')}
        return jpgs
    
    print('Searching the MD output to make a set of all the existing jpg or JPG files')
    found_filenames = set() 
    overall_progress = tqdm(total=len(subdirs), desc="Folders processed")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(search_for_jpgs, subdirs):
            found_filenames.update(result)
            overall_progress.update(1)
    overall_progress.close()

    missing_files = list(df_fnames - found_filenames)
    print(f'Removing {len(missing_files)} files from the DataFrame as they cannot be found')
    print(missing_files[:10])
    df = df[~df['File_Path_Objects'].isin(missing_files)]
    del df['File_Path_Objects']

    return df


def extract_exif_data(image_path):   #https://exiv2.org/tags.html    #306=DateTime, #36867=DateTimeOriginal 270=ImageDescription
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return {'File_Path': str(image_path), 'Date_Time': 'exif_not_found', 'Description': 'exif_not_found'}
            tag_list = [306, 36867, 270]
            dt, description = 'dt_not_found', 'description_not_found'

            for tag in tag_list:
                try:
                    value = exif_data.get(tag)
                    if value is not None and not value.isspace():
                        if tag == 306 or tag == 36867:
                            dt = value
                        elif tag == 270:
                            description = value
                except KeyError:
                    continue
            return {'File_Path': str(image_path), 'Date_Time': dt, 'Description': description}
    except (FileNotFoundError, OSError):
        return {'File_Path': str(image_path), 'Date_Time': 'file_not_valid', 'Description': 'file_not_valid'}


def get_last_exif_data(file_path):
    """Looks for an existing parquet file, and returns a dataframe.  If no file found, 
    a dataframe is returned with the right column names, no rows"""
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        column_names = ['File_Path', 'Date_Time', 'Description']
        df = pd.DataFrame(columns=column_names).astype(str)
    return df

# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------

def main(settings_pth=None):
    cfg = get_config(settings_pth)
    if not cfg.UPDATE_META_DATA:
        return
    output_path, exif_path, last_run_path, json_paths, data_root = get_paths(cfg)
    if cfg.CLASSES_FROM_DIR_NMS:
        class_list = get_class_list(data_root)
    else: class_list = cfg.CLASSES
    all_processed = process_all_jsons(json_paths, class_list,  data_root, cfg.BEST_ONLY)
    df_from_json = remove_missing_dirs(data_root, all_processed)
    df_from_json = remove_old_filepaths(data_root, all_processed)
    
    #Print some stats about rats, just as a sanity check
    num_rats = (df_from_json['Species'] == 'rat').sum()
    print(f'There are {num_rats} from the MD output json files')
    rat_folders = [grandchild for grandchild in data_root.glob('*/*') if grandchild.is_dir() and grandchild.name == 'rat']
    rat_count = 0
    for rat_folder in tqdm(rat_folders):
        rat_count += sum(1 for _ in rat_folder.glob('*.[jJ][pP][gG]'))
    print(f'There are {rat_count} rats in the dataset folders (including the independent test set folders)')
    independent_rats = [rat_folder for rat_folder in rat_folders if rat_folder.parent.name in cfg.INDEPENDENT_TEST_ONLY]
    independent_rat_count = 0
    for rat_folder in tqdm(independent_rats):
        independent_rat_count += sum(1 for _ in rat_folder.glob('*.[jJ][pP][gG]'))
    print(f'There are {independent_rat_count} rats in the independent test folders, so {rat_count - independent_rat_count} for training')
    print(f'There are {len(df_from_json)} rows in the final dataframe to be written to parquet')

    file_paths = df_from_json['File_Path'].tolist()
    exif_df = get_last_exif_data(exif_path)
    print(f'There are {len(exif_df)} rows found in the previous exif data file')
    if cfg.UPDATE_EXIF:
        exif_df = exif_df[exif_df['File_Path'].isin(file_paths)]
        print(f'After filtering based on File_Path, there are {len(exif_df)} rows left')
        exif_df = exif_df[exif_df['File_Path'] != 'file_not_valid']
        print(f'After filtering based on Date_Time, there are {len(exif_df)} rows left')
        exif_df = exif_df[exif_df['Date_Time'] != 'dt_not_found']
        print(f'After filtering based on Date_Time (again), there are {len(exif_df)} rows left')
        exif_df = exif_df[exif_df['Description'] != 'description_not_found']
        print(f'After filtering based on Description, there are {len(exif_df)} rows of exif data left')
        already_have_exif = exif_df['File_Path'].to_list()
        file_paths = list(set(file_paths).difference(set(already_have_exif)))
        print(f'Extracting EXIF data from files without complete EXIF data')
        new_exif_data = Parallel(n_jobs=8, prefer='threads')(delayed(extract_exif_data)(fp) for fp in tqdm(file_paths))
        new_exif_df = pd.DataFrame(new_exif_data)
        if len(new_exif_data)>=1:
            new_exif_df = new_exif_df[new_exif_df['Date_Time'] != 'file_not_valid']
            exif_df = pd.concat([exif_df, new_exif_df], ignore_index=True)

    exif_df['File_Path'] = exif_df['File_Path'].astype(str) #can probably remove this
    data_out = df_from_json.merge(exif_df, on='File_Path', how='inner', suffixes=('', '')) #merges vertically, removing any rows not present in both
    print('Final dataframe from interpreting jsons and exif data')
    print(data_out.head())

    exif_df.to_parquet(exif_path) # Updates the 'MD_Last_Run' folder
    data_out.to_parquet(output_path) # Updates the labels file before cleaning step.
    data_out.to_parquet(last_run_path) # Updates the 'MD_Last_Run' folder
    return data_out

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    output = main()