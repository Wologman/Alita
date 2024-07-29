import os
from pathlib import Path
import pandas as pd
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm
import time
import yaml
import cv2 
import numpy as np


class DefaultConfig:
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_01'
        self.DATA_FOLDER_NM = 'Data'
        self.INPUT_FOLDER_NM = 'Inputs'
        self.EXPS_FOLDER_NM = 'Experiments'
        self.IMAGE_FOLDER_NM = 'Cropped_Images'
        self.CLEANED_LABELS_NM = 'cleaned_labels.parquet'
        self.CROP_LABELS_NM = 'crop_labels.parquet'
        self.REMOVE_BACKGROUND = True
        self.BACKGROUND_MARGIN = 0.2
        self.PROCESS_IMAGES = True   #False is for debugging, won't do the image crops, just make the labels file
        self.BUFFER = 0.1
        self.CROP_SIZE = 600
        self.MD_RESAMPLE = True #The MD bbox will be used to rescale largest dimension + buffer down to CROP_SIZE, if > CROP_SIZE
        self.DEBUG = False  #If True, max of 50 images to be processed
        self.DEBUG_SPEED = 40
        self.EXTRA_CROP_CORES = 5  #If None, the number of available cores will be used, if 0 then will process in serial

def get_config(settings_pth):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = ['MD_RESAMPLE', 'DEBUG', 'REMOVE_BACKGROUND']
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


def find_already_finished(img_fldr):
    """Checks if any images have already been added to the directory, to allow the process to be re-started if interrupted"""
    if not os.path.isdir(img_fldr):
        os.makedirs(img_fldr)
    finished_list = list(img_fldr.rglob("*.jpg"))
    return finished_list


def make_final_labels(in_labels, debug=False, reduce_factor=None):
    """Reformat to keep only the data needed for subsequent processes"""
    def get_fn(fpath, location):
        fn = Path(fpath).name
        return f'{location}-{fn}'
    if debug:
        frac = 1/reduce_factor
        in_labels = in_labels.sample(frac=frac, random_state=42)
        
    apply_fn = in_labels.apply(lambda row: get_fn(row['File_Path'], row['Location']), axis=1)
    in_labels['Cropped_File_Name'] = apply_fn
    out_labels = in_labels[['File_Path', 'Cropped_File_Name', 'Species']].copy()
    return in_labels, out_labels


def mega_2_square(row, img_w, img_h):
    """Megadetector output is [x_min, y_min, width_of_box, height_of_box] (normalised COCO)
       corner of the box is the top left, origin is the top left.
       Want to output a square centred on the old box, with width & height = cfg.final_size"""
    final_size = row['Cropped_Size']
    x_min, y_min, width, height = row['x_min'], row['y_min'], row['Width'], row['Height']
    x_centre = (x_min + width/2) * img_w
    y_centre = (y_min + height/2) * img_h
    left = int(x_centre - final_size/2)
    top =  int(y_centre - final_size/2)
    right = left + final_size
    bottom = top + final_size

    # Corrections for when the box is out of the original image dimensions. Shifts by that amount
    # Then using max & min to catch the rare scenario where the original image is less than the final 
    # crop dimensions, so shifting like above leaves one dimension on the edge, the other out of range 
    
    if (left < 0) and (right > img_w):
        new_left, new_right = 0, img_w
    else:
        new_left   = max(0, left  - (left < 0) * left - (right > img_w)*(right - img_w))
        new_right  = min(img_w, right - (left < 0) * left - (right > img_w)*(right - img_w))
        
    if (top < 0) and (bottom > img_h):
        new_top, new_bottom = 0, img_h
    else:
        new_top    = max(0, top    - (top < 0) * top - (bottom > img_h) * (bottom - img_h))
        new_bottom = min(img_h, bottom - (top < 0) * top - (bottom > img_h) * (bottom - img_h))

    return new_left, new_top, new_right, new_bottom


def subtract_background(image, row):
    height, width, channels = image.shape
    dtype = image.dtype
    new_image = np.zeros((height, width, channels), dtype=dtype) 
    margin = row['Margin']*min(row['Height'], row['Width']) #scale the margin wrt the box size  
    clamp = lambda n: max(min(1, n), 0)
    x_min = int(clamp(row['x_min'] - margin)*width)
    y_min = int(clamp(row['y_min'] - margin)*height)
    x_max = int(clamp(row['x_min'] + row['Width'] + margin)*width)
    y_max = int(clamp(row['y_min'] + row['Height'] + margin)*height)
    crop = image[y_min:y_max, x_min:x_max] #crop the image
    new_image[y_min:y_max, x_min:x_max] = crop #broadcast on to the black background
    return new_image


def pad_to_size(image, size):
    height, width, channels = image.shape
    dtype = image.dtype
    square_image = np.zeros((size, size, channels), dtype=dtype)
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    square_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return square_image


def get_new_scale(row, width, height):
    """figures out how much to scale down the new image to, so the max(bounding-box) + buffer = the desired crop size
    only effects images where the crop box would be greater than the crop size"""
    clamp = lambda n: max(min(1, n), 0)
    x_min = clamp(row['x_min'] - row['Buffer'])
    y_min = clamp(row['y_min'] - row['Buffer'])
    x_max = clamp(row['x_min'] + row['Width'] + row['Buffer'])
    y_max = clamp(row['y_min'] + row['Height'] + row['Buffer'])
    max_dimension = max([(x_max - x_min)*width, (y_max - y_min)*height]) 
    return row['Cropped_Size']/max_dimension if max_dimension > row['Cropped_Size'] else None    


def crop_save_image(row):
    rescale = row['Rescale']
    in_fn = row['File_Path']
    img_folder =  row['Folder']
    crop_size = row['Cropped_Size']
    out_fn = img_folder / row['Cropped_File_Name']
    sub_background = row['Subtract_Background']

    try:
        img = cv2.imread(in_fn)
        h, w, _ = img.shape
        if sub_background:
            img = subtract_background(img, row)
        if rescale:
            scale = get_new_scale(row, w, h)
            if scale:
                w, h = int(round(w * scale)), int(round(h * scale))
                img = cv2.resize(img, (w, h), cv2.INTER_LANCZOS4)
        
        left, top, right, bottom = mega_2_square(row, w, h)
        img = img[top:bottom, left:right, :]
        
        if (right-left < crop_size) or (bottom-top < crop_size):
            img = pad_to_size(img, crop_size)

        cv2.imwrite(str(out_fn), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    except FileNotFoundError:
        print(f"The specified file '{in_fn}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return


def serial_image_processing(df, img_fldr, cfg):
    """Process the images from a single core, only gets used if cfg.EXTRA_CROP_CORES == 0"""
    size = cfg.CROP_SIZE 
    remove_background = cfg.REMOVE_BACKGROUND
    margin = cfg.BACKGROUND_MARGIN 
    img_bfr = cfg.BUFFER
    rescale = cfg.MD_RESAMPLE
    
    df['Margin']=margin
    df['Folder']=img_fldr
    df['Buffer']=img_bfr
    df['Rescale']=rescale
    df['Cropped_Size']=size
    df['Subtract_Background'] = remove_background
    for _, df_row in df.iterrows():
        crop_save_image(df_row)
    return


def parallel_image_processing(df, img_fldr, cfg, cores=None):
    size = cfg.CROP_SIZE 
    remove_background = cfg.REMOVE_BACKGROUND
    margin = cfg.BACKGROUND_MARGIN 
    img_bfr = cfg.BUFFER
    rescale = cfg.MD_RESAMPLE
    num_tasks = len(df)
    completed_tasks = 0
    
    def process_image(record):
        nonlocal completed_tasks
        crop_save_image(record)
        completed_tasks += 1
        pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
        df['Margin']= margin
        df['Folder'] = img_fldr
        df['Buffer'] = img_bfr
        df['Rescale'] = rescale
        df['Cropped_Size'] = size
        df['Subtract_Background'] = remove_background
        pbar = tqdm(total=num_tasks, desc="Cropping Images")
        futures = [executor.submit(process_image, record) for record in df.to_dict('records')]
        concurrent.futures.wait(futures)
        pbar.close()


def remove_extra_images(dir_pth, dataframe):
    finished_list = find_already_finished(dir_pth)
    finished_list = [str(f.name) for f in finished_list]
    expected_files =  dataframe['Cropped_File_Name'].to_list()
    extras = set(finished_list) - set(expected_files)
    print(f'Found {len(extras)} extra files in the images directory, not in the metadata file that will be removed')
    for f in extras:
        f_path = dir_pth / f
        os.remove(f_path)
    return


def remove_missing_images(dir_pth, dataframe):
    """If any images are missing from the image crops, they are removed from the dataframe"""
    names_in_dir = [str(f.name) for f in Path(dir_pth).rglob('*.*')]
    names_in_df = list(dataframe['Cropped_File_Name'].unique())
    missing_names = list(set(names_in_df) - set(names_in_dir))
    print(f'There are {len(missing_names)} files in the labels dataframe, not found in the image folder.  For example:')
    print(missing_names[:5])
    old_length = len(dataframe)
    mask = dataframe['Cropped_File_Name'].isin(missing_names)
    dataframe = dataframe[~mask]
    print(f'{old_length - len(dataframe)} rows were removed from the labels dataframe')
    return dataframe

# ----------------------------------- Main Script-----------------------------------------
# ----------------------------------------------------------------------------------------
def main(settings_pth = None):
    cfg = get_config(settings_pth)
    project_folder = Path(__file__).resolve().parent.parent
    input_folder = project_folder / cfg.DATA_FOLDER_NM / cfg.EXPS_FOLDER_NM / cfg.EXPERIMENT_NAME / cfg.INPUT_FOLDER_NM
    in_labels_path = input_folder / cfg.CLEANED_LABELS_NM
    out_labels_path = input_folder / cfg.CROP_LABELS_NM
    img_dstn_fldr = input_folder / cfg.IMAGE_FOLDER_NM

    if cfg.EXTRA_CROP_CORES is None:
        num_cores = cpu_count()
    else:
        num_cores = cfg.EXTRA_CROP_CORES + 1
    
    finished_list = find_already_finished(img_dstn_fldr)
    print(f'Found {len(finished_list)} previously processed files in {str(img_dstn_fldr)}')

    in_df = pd.read_parquet(in_labels_path)
    print(f'Found {len(in_df)} lines in the labels metadata file: {in_labels_path.name}')
    
    in_df, out_df = make_final_labels(in_df, cfg.DEBUG, cfg.DEBUG_SPEED)
    df = in_df[~in_df['Cropped_File_Name'].isin(finished_list)].copy() 
    print(out_df.head())
    print(df['File_Path'].nunique())
    print(df['Cropped_File_Name'].nunique())

    start_time = time.time()
    if cfg.PROCESS_IMAGES:
        if num_cores > 1:
            print(f'Processing images with {num_cores} cores')
            parallel_image_processing(df, 
                                      img_dstn_fldr,
                                      cfg,  
                                      num_cores)
        else:
            print(f'Processing images in serial')
            serial_image_processing(df, 
                                    img_dstn_fldr, 
                                    cfg,
                                    cfg.CROP_SIZE, 
                                    cfg.REMOVE_BACKGROUND, 
                                    cfg.BACKGROUND_MARGIN,
                                    cfg.BUFFER, 
                                    cfg.RESCALE)

    remove_extra_images(img_dstn_fldr, out_df)
    out_df = remove_missing_images(img_dstn_fldr, out_df)
    out_df.to_parquet(out_labels_path)
    end_time = time.time()
    print(f'Processing completed in {end_time-start_time:.2f} seconds')
    num_images = len(os.listdir(img_dstn_fldr))
    print(f'{num_images}, files stored in the output directory: {img_dstn_fldr}')
    print(f'The crop labels file is stored in : {out_labels_path}')
    print(out_df.head(3))
    print(f'There are {len(out_df)} rows in the labels dataframe, a difference of {len(out_df) - num_images}')
    return

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()