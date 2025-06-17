''' Accepts a dataframe of images, classes, camera locations, study area and bounding boxes
    Method-1    Performs image hash from a class + camera location to remove near dupicates
    Method-2    Group by class + study area
                Uses the MD boxes to crop the image, then generate embeddings 
                Select N images for the maximum diversity of embeddings
'''

import sys
import gc
import os
import base64
import random
from itertools import combinations, islice

from test_cuda import test_cuda
from Inference import PredatorDataset, ImagePrep, custom_collate

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import h5py

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from PIL import Image
import imagehash

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import timm


class DefaultConfig:
    '''Namespace for data balancing parameters'''
    # Random, & K-means based on MD predictions
    IMAGE_LIMIT = 500 # Maximum number of images of a given class at a particular location code
    MAX_PER_CLASS_PER_CAMERA = 250 #Maximum images from a particular animal category, from a given camera
    MIN_HASH_DIFFERENCE = 50  #arbitrary for now
    MAX_PER_CLASS_PER_LOCATION = 200


class ImageConfig:
    '''Namespace for image processing parameters'''
    EDGE_FADE = False
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


class Warn:  #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'

class Colour: 
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def hash_image(path, bbox):
    '''Uses a bounding box to crop out the animal detected,
    then runs a perceptual hash over that crop.  Returns the hash
    #Megadetector bbox output is [x_min, y_min, width_of_box, height_of_box] (normalised COCO)
    #The corner of the box is the top left, origin is the top left.'''
    img = load_image(path, mode='RGB')  #Rows x Columns x Channels (h,w,c)

    if img is None:
        print(f"Warning: Failed to load image {path}")
        return None  # Explicitly return None to handle it in Parallel()
    img_width = img.shape[1]
    img_height = img.shape[0]
    x_min = int(bbox[0] * img_width)  #Left 
    y_min = int(bbox[1] * img_height) #Top
    width = int(bbox[2] * img_width)
    height = int(bbox[3] * img_height)
    x_max = x_min + width #Right
    y_max = y_min + height #bottom
    crop = img[y_min:y_max,x_min:x_max,:]
    pil_image = Image.fromarray(crop)
    phash = imagehash.phash(pil_image, hash_size=16)
    phash_str = str(phash)
    phash_bytes = bytes.fromhex(phash_str)
    phash_array = np.frombuffer(phash_bytes, dtype=np.uint8)
    return phash_array


def encode_images(images_df, model, image_cfg):
    images_df = images_df.copy()
    images_df['Targets'] = 'cat'  #just a placeholder

    transforms = ImagePrep(image_cfg.INPUT_MEAN, image_cfg.INPUT_STD).transform

    dataset = PredatorDataset(images_df, 
                            transforms, 
                            crop_size=image_cfg.CROP_SIZE, 
                            buffer = image_cfg.BUFFER,
                            resize_method=image_cfg.RESIZE_METHOD, 
                            remove_background=image_cfg.REMOVE_BACKGROUND,
                            fade_edges=image_cfg.EDGE_FADE,
                            min_margin=image_cfg.MIN_FADE_MARGIN,
                            downsample=image_cfg.MD_RESAMPLE,
                            dt_formats=['%Y:%m:%d %H:%M:%S'])

    #no point using crazy high number as we're bottlenecked by retrieing from HDD
    num_workers = min(os.cpu_count() //2,  6)

    loader = DataLoader(dataset,
                             batch_size=32,
                             collate_fn=custom_collate,
                             shuffle=False,
                             num_workers=num_workers)
    
    pbar = tqdm(total=len(dataset), desc='Making image embeddings')
    model.eval()
    device = model.device
    paths_list, crops_list, dt_list, embeddings_list = [], [], [], []
    embeddings_dict = {}
    
    try:
        for batch in loader:
            if batch is None:
                print("Skipping batch due to None")
                continue

            images, targets, paths, ltrb_dims, d_times = batch

            if images is None or targets is None or paths is None or ltrb_dims is None or d_times is None:
                print(f"Skipping batch due to None elements: {batch}")
                continue

            images = images.to(device)
            with torch.no_grad():
                embeddings = model(images)
            #if __name__ == '__main__':
            pbar.update(images.shape[0])
            l2_noarmalised = F.normalize(embeddings, p=2, dim=1)
            cpu_embeddings = l2_noarmalised.detach().cpu().numpy()
            paths_list.extend(list(paths))
            embeddings_list.extend(list(cpu_embeddings))
            crops_list.extend(list(ltrb_dims))
            dt_list.extend(d_times)
            del images, embeddings, cpu_embeddings  # Free memory
            torch.cuda.empty_cache()
            gc.collect() 
        pbar.close()
    except TypeError as e:
        print(f"Error: {e}")
        print('Fatal error: No valid images were found for inference')
        print('This could occur because the filepaths in the MegaDetector output: mdPredictions.json are out of date. Try deleting this file and re-run')
        print('It could also be possible that all the images intended for inference are corrupted and cannot be opened')
        sys.exit()

    embeddings_dict = {key : val.squeeze() for (key, val) in zip(paths_list, embeddings_list)}
    return embeddings_dict


def dict_to_h5(my_dict, path):
    ''' Saves a dict with filepaths as keys, and numpy arrays as values'''
    def encode_key(key):
        """Encodes file paths to be HDF5-compatible using base64."""
        return base64.urlsafe_b64encode(key.encode()).decode()

    if not my_dict:
        print("Warning: No hdf5 data to save.")
        return

    print(f'Saving {len(my_dict)} items in hdf5 format to {path}...')

    encoded_keys = np.array([encode_key(k) for k in my_dict.keys()], dtype='S')
    max_length = max(len(k) for k in encoded_keys) if encoded_keys.size > 0 else 1
    
    values_list = list(my_dict.values())
    first_value = values_list[0]
    if isinstance(first_value, np.ndarray):
        inferred_dtype = first_value.dtype
    elif isinstance(first_value, list):
        inferred_dtype = np.array(first_value).dtype
    else:
        raise TypeError(f"Unsupported type {type(first_value)} in hash_dict values.")
    ###################################################################
    values = np.stack(values_list).astype(inferred_dtype)   ##### raise ValueError('all input arrays must have the same shape')
    with h5py.File(str(path), "w") as h5file:
        h5file.create_dataset("keys", data=encoded_keys, dtype=f"S{max_length}") 
        h5file.create_dataset("values", data=values, dtype=inferred_dtype)  # Dynamic dtype


def h5_to_dict(path):
    """Load an HDF5 file into a dictionary."""
    def decode_key(key):
        return base64.urlsafe_b64decode(key).decode()

    with h5py.File(path, "r") as h5file:
        encoded_keys = h5file["keys"][:]
        keys = [decode_key(k) for k in encoded_keys]
        values = h5file["values"][:]

    results = dict(zip(keys, values))
    return results


def encode_key(key):
    """Encodes file paths to be HDF5-compatible using base64."""
    return base64.urlsafe_b64encode(key.encode()).decode()

def decode_key(encoded_key):
    """Decodes base64-encoded file paths back to original form."""
    return base64.urlsafe_b64decode(encoded_key.encode()).decode()

def add_group_to_h5(h5_file, df, group_name):
    """Adds a group from a dataframe to an opened HDF5 file.
    
    - Maps filepaths to unique integer indices.
    - Stores base64-encoded filepaths as a separate dataset.
    - Uses integer indices for Image_0 and Image_1 datasets.
    """

    group = h5_file.require_group(group_name)

    # Step 1: Create a unique mapping of filepaths to integer indices
    unique_paths = sorted(set(df['Image_0']).union(df['Image_1']))  # Get unique filepaths
    path_to_index = {p: i for i, p in enumerate(unique_paths)}  # Map filepaths to indices

    # Step 2: Convert filepaths to indices in DataFrame
    image_0_indices = np.array([path_to_index[p] for p in df['Image_0']], dtype=np.uint16)
    image_1_indices = np.array([path_to_index[p] for p in df['Image_1']], dtype=np.uint16)
    
    # Step 3: Store integer values for Distance (clamped to 255)
    values = np.clip(df['Distance'].to_numpy(), 0, 255).astype(np.uint8)

    # Step 4: Encode filepaths with base64 and store them
    encoded_paths = np.array([encode_key(p) for p in unique_paths], dtype="S")

    # Step 5: Write datasets to HDF5
    group.create_dataset("Image_0", data=image_0_indices, dtype=np.uint16)
    group.create_dataset("Image_1", data=image_1_indices, dtype=np.uint16)
    group.create_dataset("Distance", data=values, dtype=np.uint8)
    max_length = max((len(e) for e in encoded_paths), default=1) 
    group.create_dataset("FilePaths", data=encoded_paths, dtype=f"S{max_length}")


def load_h5_group(hdf5, group_name):
    group = hdf5[group_name]

    # Load the integer keys for Image_0 and Image_1
    img0_keys = group["Image_0"][:]
    img1_keys = group["Image_1"][:]

    # Load the mapping of indices to file paths
    encoded_paths = group["FilePaths"][:]
    file_paths = [base64.urlsafe_b64decode(p).decode() for p in encoded_paths]

    # Convert integer keys back to file paths
    img_0_paths = [file_paths[k] for k in img0_keys]
    img_1_paths = [file_paths[k] for k in img1_keys]

    distances = list(group["Distance"])

    df = pd.DataFrame({'Image_0': img_0_paths,
                       'Image_1': img_1_paths,
                       'Distance': distances})

    return df

def get_image_hashes(current_df, h5_path, verbose=True):
    '''
    From a dataframe with image file-paths looks for an h5 file
    uses that to get image perceptual hashes, and compile a list of missing ones
    runs a hashing algorithm over the images missing hashes
    '''

    if verbose:
        print('The DataFrame from inside get_image_hashes')
        print(current_df.head())

    if h5_path.exists():
        current_hashes = h5_to_dict(h5_path)
    else:
        current_hashes = {}
    current_hash_set = set(current_hashes.keys())
    all_images_set = set(current_df['File_Path'])
    missing_hashes = all_images_set - current_hash_set
    filtered_df = current_df[current_df['File_Path'].isin(missing_hashes)]
    missing_dict = dict(zip(filtered_df["File_Path"], 
                      zip(filtered_df["x_min"], 
                          filtered_df["y_min"], 
                          filtered_df["Width"], 
                          filtered_df["Height"])))

    results = dict(
    Parallel(n_jobs=4)(
            delayed(lambda k, v: (k, hash_image(k, v)))(key, val)
            for key, val in tqdm(missing_dict.items(), desc='Creating hashes for images not matching existing filepaths')
            )
        )
    
    new_hashes_dict = {k: h for k, h in tqdm(results.items(),
                        desc='Removing failed hashes') if h is not None
                      } #remove any that failed to load

    all_hashes = current_hashes | new_hashes_dict
    dict_to_h5(all_hashes, h5_path)
    return all_hashes  


class CustomModel(pl.LightningModule):
    def __init__(self,
                 model_name='tf_efficientnetv2_l.in21k_ft_in1k',
                 ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.in_features = self.backbone.classifier.in_features
        print(f'There are {self.in_features} features in the embeddings')
        #Replace the classifier head with Identity because we only want the embeddings
        self.backbone.classifier = nn.Identity() 

    def forward(self, images):
        logits = self.backbone(images)
        return logits


def get_encoder():
    network = CustomModel()
    device, gpu = test_cuda()
    network.to(device)
    return network


def get_image_embeddings(current_df, h5_path, image_config):
    if h5_path.exists():
        embeddings  = h5_to_dict(h5_path)
    else: 
        embeddings = {}

    embedding_set = set(embeddings.keys())
    all_images_set = set(current_df['File_Path'])
    missing_embeddings = all_images_set - embedding_set
    if missing_embeddings:
        model = get_encoder()
        missing_embeddings_df = current_df[current_df['File_Path'].isin(missing_embeddings)]
        results = encode_images(missing_embeddings_df, model, image_config)
        new_embeddings = {k: h for k, h in tqdm(results.items(), desc='Removing failed embeddings') if h is not None}
        embeddings = embeddings | new_embeddings


    dict_to_h5(embeddings, h5_path)
    return embeddings


def remove_by_threshold(group, species, other_grouper, h5_file, hash_dict, threshold=50):
    #Distance dict should only contain distances related to this group
    group_name = species + '|' + other_grouper
    if h5_file.mode == 'r':
        df_dist = load_h5_group(h5_file, group_name)
    else:
        df_dist = get_dist(group, hashes=hash_dict)
        add_group_to_h5(h5_file, df_dist.copy(), group_name)
    
    df_dist = df_dist[df_dist['Distance'] < threshold]
    df_dist = df_dist.sort_values(by='Distance', ascending=True)

    removed_images = []
    removed_pairs = []

    while not df_dist.empty:
        remove = df_dist.iloc[0]['Image_0']
        removed_images.append(remove)
        remove_cond = (df_dist['Image_0'] == remove) | (df_dist['Image_1'] == remove)
        removing_rows = df_dist[remove_cond]
        pairs_to_remove = zip(removing_rows['Image_0'], removing_rows['Image_1'])
        unique_tuples = list(set(tuple(sorted(pair)) for pair in pairs_to_remove))
        removed_pairs.extend(unique_tuples)
        df_dist = df_dist.loc[~df_dist.index.isin(removing_rows.index)]
    return removed_images, removed_pairs


def get_dist(group, hashes, batch_size=256, use_gpu=True):
    paths = group['File_Path'].to_numpy()
    num_images = len(paths)

    if num_images < 2:
        return pd.DataFrame(columns=['Image_0', 'Image_1', 'Distance'])

    # Convert hashes into a NumPy array (assuming they are already unpacked into booleans)
    hash_arr = np.array([hashes[p] for p in paths], dtype=bool)

    # Move to GPU if available
    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    hash_tensor = torch.tensor(hash_arr, dtype=torch.bool, device=device)  # Keep dtype=torch.bool

    def batch_iterator(iterable, batch_size):
        """Yield slices of an iterable in batches."""
        iterator = iter(iterable)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    results = []

    for batch in batch_iterator(combinations(range(num_images), 2), batch_size):
        batch_pairs = torch.tensor(batch, dtype=torch.long, device=device)  # Use long for indexing

        # Use tensor indexing with boolean tensors
        hash1 = hash_tensor[batch_pairs[:, 0]]  # Shape: (batch_size, bit_length)
        hash2 = hash_tensor[batch_pairs[:, 1]]  # Shape: (batch_size, bit_length)

        # Compute Hamming distance using XOR
        distances = (hash1 ^ hash2).sum(dim=1)

        # Convert indices back to file paths
        results.append(pd.DataFrame({
            'Image_0': paths[batch_pairs[:, 0].cpu().numpy()],
            'Image_1': paths[batch_pairs[:, 1].cpu().numpy()],
            'Distance': distances.cpu().numpy()
        }))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def remove_duplicates_by_hash(df,
                              cols=['Species', 'Camera'],
                              h5_path=None,
                              hash_dist_h5_path=None,
                              min_distance=50,
                              recalculate=True,
                              verbose=False):
    hash_dict = get_image_hashes(df, h5_path, verbose=verbose)
    '''Generate perceptual image hashes based on crops of the image from 
       object detector bounding boxes.  Using thise hashes to filter out 
       one of any pair for which the hash value is below some threshold'''

    df = df[df["File_Path"].isin(hash_dict.keys())]

    if verbose:
        random_keys = random.sample(list(hash_dict.keys()), 5)
        print('Some example hashes that will be used to find duplicates:')
        for key in random_keys:
            print(f'Key: {key}')
            print(hash_dict[key])
        print('The dataframe for hash calculations:')
        print(df.head())
        print(f'Before filtering there are {len(df)} images')

    def unpack_single_hash(p, h):
        """Unpacks a single hash."""
        return p, np.unpackbits(np.array(h, dtype=np.uint8))

    if verbose:
        print(f'Unpacking {len(hash_dict)} uint8 hashes into bits')
    unpacked_hashes = dict(
                      Parallel(n_jobs=8)(
                      delayed(unpack_single_hash)(p, h) for p, h in hash_dict.items())
                     )

    grouped = df.groupby(cols)
    if verbose:
        grouped = tqdm(grouped, total=len(grouped), desc=f'Performing perceptual hash comparisons')
    
    h5_dist_f = h5py.File(hash_dist_h5_path, "w") if recalculate else h5py.File(hash_dist_h5_path, "r")

    hash_comparisons =  [
            remove_by_threshold(
                group,
                cols[0],
                cols[1],
                h5_file=h5_dist_f,
                hash_dict={fp: unpacked_hashes[fp] for fp in group["File_Path"] if fp in unpacked_hashes},
                threshold=min_distance,
            )
            for (cols[0], cols[1]), group in grouped
            ]

    h5_dist_f.close()

    images_to_remove = [removal[0] for removal in hash_comparisons]
    images_to_remove = [item for sublist in images_to_remove for item in sublist]
    pairs_to_remove = [removal[1] for removal in hash_comparisons]  #lists in Distance_Location groups

    df = df[~df['File_Path'].isin(images_to_remove)]
    if verbose:
        print(f'There are {len(images_to_remove)} images to remove')
        print(f'After filtering there are {len(df)} rows left')
        #Note that we have eliminated any images that could not be opened to hash.
    return df, pairs_to_remove


def farther_first(group, embeddings, select_n):
    '''From an initial random point, choose successive images that are furthest from the 
    pool of existing choices, until the required number number, or the whole dataset is reached'''
    file_paths = list(group['File_Path'])

    if len(file_paths) <= select_n:
        return file_paths

    valid_paths = [k for k in file_paths if k in embeddings]

    if not valid_paths:
        return []

    embeddings_array = np.stack([embeddings[k] for k in valid_paths])
    p, d = embeddings_array.shape
    select_n = min(select_n, p)

    # Randomly pick the first point
    selected_indices = [np.random.randint(0, p)]
    min_distances = np.ones(p)  # To store minimum distances to selected points

    for _ in range(1, select_n):
        last_selected = embeddings_array[selected_indices[-1]].reshape(1, -1)
        similarities = np.dot(embeddings_array, last_selected.T).flatten()
        distances = 1 - similarities  # Convert to cosine distance
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances)

        # Select the farthest point based on the minimum distance to any selected point
        farthest_index = np.argmax(min_distances)
        selected_indices.append(farthest_index)

    return [valid_paths[i] for i in selected_indices]


def limit_with_embeddings(df, 
                          h5_path,
                          img_cfg,
                          cols=['Species', 'Location'],
                          limit = 500,
                          verbose = False):
    '''Limit the number of images in a way that maximises the angle between normalised embeddings
       based on embeddings caluclated from the images cropped the way they would be cropped for training'''

    embedding_dict = get_image_embeddings(df, h5_path, image_config=img_cfg)

    if verbose:
        random_keys = random.sample(list(embedding_dict.keys()), 4)
        print('Some example embeddings that will be used to maximise diversity for a given location & class:')
        for key in random_keys:
            print(f'{embedding_dict[key]}, size {embedding_dict[key].shape}')

    def farther_first_wrapper(group, embedding_dict, select_n):
        relevant_keys = set(group['File_Path'])
        sub_embeddings = {k: embedding_dict[k] for k in relevant_keys if k in embedding_dict}
        selected = farther_first(group, embeddings=sub_embeddings, select_n=select_n)
        #print(f"Group ({group['Species'].iloc[0]}, {group['Location'].iloc[0]}): {len(selected)} selected out of {len(group)}")
        return selected

    filtered_filepaths = (
        df.groupby(cols)
        .apply(farther_first_wrapper, embedding_dict=embedding_dict, select_n=limit)
        .explode()
    )

    filtered_df = df[df['File_Path'].isin(filtered_filepaths)]
    return filtered_df


def load_image(image_path, mode):
    try:
        image = Image.open(image_path)
        if mode == 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def cluster_indices(df, limit):
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
    return indices


def check_duplicate_paths(df):
    duplicate_paths = df[df.duplicated('File_Path', keep=False)]
    if len(duplicate_paths) > 0:
        duplicate_paths = duplicate_paths.sort_values('File_Path')
        duplicate_paths.reset_index(drop=True, inplace=True)
        print(Warn.S + f"There are {duplicate_paths['File_Path'].nunique()} unique file paths with duplicates." + Warn.E)
        print('Only the row with the highest confidence will be used, but the reason for the duplication should be investigated')
        df_unique = df.loc[df.groupby('File_Path')['Confidence'].idxmax()].reset_index(drop=True)
        print(f'After removing duplicates with the lower Confidence score, there are {len(df_unique)} rows')
        df = df_unique.copy()
    return df


def limit_with_bbox_vals(df, 
                         cols=['Species', 'Location'],
                         limit=250):
    """
    Args:
        limit (int): The maximum to be allowed for this grouping
        limiting_col (str): The column header
    Returns:
        dataframe : Dataframe with no more than the allowed limit for each grouping
    """

    def remove_overs(row):
        limiter = getattr(row, cols[1])
        obs_cls = row.Species
        all_idxs = df2[(df2[cols[1]] == limiter) & (df2[cols[0]] == obs_cls)].index.to_list()
        group_df = df2.iloc[all_idxs]
        keep_idxs = cluster_indices(group_df, limit)
        return keep_idxs
    
    df2 = df.reset_index(drop=True)

    grouped_counts = df2.groupby(cols).size().reset_index(name='count')
    under_lim = grouped_counts[grouped_counts['count'] <= limit]
    over_lim = grouped_counts[grouped_counts['count'] > limit]

    description = f'Processing labels under max {cols[0]}-{cols[1]} limit'
    combined_groups = under_lim['Species'] + under_lim[cols[1]]
    
    df2['Location-Species'] = df2[cols[0]] + df2[cols[1]]
    under_max_lim_df = df2[(df2['Location-Species'].isin(combined_groups))]
    #print(f'there are {len(under_max_lim_df)} images that will be kept from under camera-limit combinations')

    description = f'Processing labels over max {cols[0]}-{cols[1]} limit'
    iterate_list = list(over_lim.itertuples())
    nested_list = Parallel(n_jobs=-1)(delayed(remove_overs)(row) for row in iterate_list)
    indices_to_keep = [item for sublist in nested_list for item in sublist]
    #print(f'There are {len(indices_to_keep)} indices that will be kept from over-limit combinations')
    max_limit_df = df2.iloc[indices_to_keep]
    new_df = pd.concat([under_max_lim_df, max_limit_df])
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def limit_randomly(df, 
                   cols=['Species', 'Location'],
                   limit=250):
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
        limiter = getattr(row, cols[1])
        obs_cls = getattr(row, cols[0])
        #obs_cls = row.at[cols[0]]
        #obs_cls = row[cols[0]]
        #obs_cls = row[cols[0]].iloc[0]
        all_idxs = df2[(df2[cols[1]] == limiter) & (df2[cols[0]] == obs_cls)].index.to_list()
        return random.sample(all_idxs, limit)

    df2 = df.reset_index(drop=True)

    grouped_counts = df2.groupby(cols).size().reset_index(name='count')
    under_lim = grouped_counts[grouped_counts['count'] < limit]
    over_lim = grouped_counts[grouped_counts['count'] >= limit]

    description = f'Processing labels under max {cols[0]}-{cols[1]}  limit'
    combined_groups = under_lim[cols[0]] + under_lim[cols[1]]
    
    df2['Location-Species'] = df2[cols[0]] + df2[cols[1]]
    under_max_lim_df = df2[(df2['Location-Species'].isin(combined_groups))]
    print(f'there are {len(under_max_lim_df)} images that will be kept from under camera-limit combinations')

    description = f'Processing labels over max {cols[0]}-{cols[1]} limit'
    iterate_list = list(over_lim.itertuples())
    nested_list = Parallel(n_jobs=-1)(delayed(remove_overs)(row) for row in tqdm(iterate_list, desc=description))
    indices_to_keep = [item for sublist in nested_list for item in sublist]
    print(f'There are {len(indices_to_keep)} indices that will be kept from over-limit combinations')
    max_limit_df = df2.iloc[indices_to_keep]
    new_df = pd.concat([under_max_lim_df, max_limit_df])
    new_df.reset_index(drop=True, inplace=True)
    return new_df


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

########################################################################################
########################################################################################

def main():
    cfg = DefaultConfig()
    img_cfg = ImageConfig()
    df_path = Path('/media/olly/Red_SSD/Alita/Data/Experiments/MD_Last_Run/all_labels.parquet')
    hash_h5_path = Path('/media/olly/Red_SSD/Alita/Data/Experiments/MD_Last_Run/image_hashes.h5')
    hash_dist_h5_path = Path('/media/olly/Red_SSD/Alita/Data/Experiments/MD_Last_Run/hash_pair_distances.h5')
    embedding_h5_path = Path('/media/olly/Red_SSD/Alita/Data/Experiments/MD_Last_Run/image_embeddings.h5')
    #df_path = Path(r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Alita\Data\Experiments\MD_Last_Run\all_labels.parquet')
    #hash_h5_path = Path(r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Alita\Data\Experiments\MD_Last_Run\image_hashes.h5')
    #embedding_h5_path = Path(r'C:\Users\User\OneDrive - Department of Conservation\Desktop\Alita\Data\Experiments\MD_Last_Run\image_embeddings.h5')

    df = pd.read_parquet(df_path)
    df=format_df(df)
    df = check_duplicate_paths(df)

    original_length = len(df)
    print(f'The original length before removing duplicates is {original_length}')

    df, pairs_to_remove = remove_duplicates_by_hash(df,
                                                    h5_path=hash_h5_path,
                                                    hash_dist_h5_path= hash_dist_h5_path,
                                                    cols=['Species','Camera'],
                                                    min_distance=cfg.MIN_HASH_DIFFERENCE,
                                                    recalculate=True)

    print(Colour.S + f'{len(df)} lines left after removing similar images by hashing, ' + 
          f'with a hash threshold of {cfg.MIN_HASH_DIFFERENCE}' + Colour.E)

    df = limit_with_bbox_vals(df, 
                              cols = ['Species', 'Location'], 
                              limit=cfg.MAX_PER_CLASS_PER_LOCATION)

    df = limit_randomly(df,
                      cols = ['Species', 'Location'],
                      limit=cfg.MAX_PER_CLASS_PER_LOCATION)

    df = limit_randomly(df,
                      cols=['Species', 'Camera'],
                      limit=cfg.MAX_PER_CLASS_PER_CAMERA)

    df = limit_with_embeddings(df,
                               h5_path=embedding_h5_path,
                               cols=['Species','Location'],
                               img_cfg=img_cfg,
                               limit=cfg.MAX_PER_CLASS_PER_LOCATION)

    print(f'{len(df)} lines left after applying all filters')

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()