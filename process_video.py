import cv2
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import piexif
import datetime as dt
import traceback
import time
from typing import Optional
import zlib
import base64
from multiprocessing import Pool, cpu_count
import tempfile
import shutil
import json

# ---------------- Functions & Classes for basic setup-----------------------------------------
# ---------------------------------------------------------------------------------------------
class DefaultConfig:
    def __init__(self):
        self.IMAGE_INTERVAL = 0.5
        self.MAX_VID_SAMPLES = 20

def get_config(settings_pth=None):
    """Gets an instance of the config class, then looks for the settings file, if it finds one evaluates specific strings to python expressions"""
    evaluate_list = []
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


def get_video_creation_time(video_path):
    try:
        # Try using last modified time (mtime)
        mtime = os.path.getmtime(video_path)
        date_time = dt.datetime.fromtimestamp(mtime)
    except OSError:
        try:
            # Fall back to creation time (ctime)
            ctime = os.path.getctime(video_path)
            date_time = dt.datetime.fromtimestamp(ctime)
        except Exception as e:
            print(f"Error extracting creation time: {e}")
            date_time = dt.datetime(1977, 10, 22, 0, 0, 0)
    return date_time


def extract_frames(video_path_tuple: tuple,
                   out_dir: Optional[Path] = None,
                   time_interval: float = 0.5,
                   max_samples: int = 20,
                   mapping_file: Optional[Path] = None
                   ):
    sub_dir_nm = video_path_tuple[0]  # Should be a unique integer per video
    video_path = Path(video_path_tuple[1])

    if out_dir is None:
        out_dir = video_path.parent / 'Frames'
    out_sub_dir = out_dir / f'video_{sub_dir_nm}'
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    creation_time = get_video_creation_time(video_path)
    if creation_time is None:
        creation_time = dt.datetime(1977, 10, 22, 0, 0, 0)

    mapping = []  # [(original_video_path, short_frame_filename)]

    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not cap.isOpened():
            print(f"Bugger! Couldn't open video file: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
    except Exception as e:
        print(f"Error opening video file {video_path}: {e}")
        return

    frame_interval = int(fps * time_interval)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frame_interval:
        frame_interval = max(1, total_frames - 1)

    if (total_frames // frame_interval > max_samples):
        frame_interval = total_frames // max_samples

    for i, frame_count in enumerate(range(0, total_frames, frame_interval)):
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        except Exception as e:
            print(f"Error reading frame {frame_count} from {video_path}: {e}")
            continue
        if not ret:
            break

        frame_filename = f"{i:05d}.jpg"
        frame_path = out_sub_dir / frame_filename
        current_time = creation_time + dt.timedelta(milliseconds=(frame_count / fps) * 1000)
        datetime_str = current_time.strftime('%Y:%m:%d %H:%M:%S')

        try:
            cv2.imwrite(str(frame_path), frame)
            exif_dict = {"Exif": {piexif.ExifIFD.DateTimeOriginal: datetime_str.encode('utf-8')}}
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(frame_path))
            mapping.append((str(video_path), frame_filename))
        except Exception as e:
            print(f"Error writing frame {frame_filename}: {e}")

        if frame_count == 0:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
    cap.release()

    if mapping_file is None:
        mapping_file = out_sub_dir / "frame_mapping.json"
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving mapping to {mapping_file}: {e}")
    return


def extract_wrapper(args):
    fp, out_dir, time_interval, max_samples = args
    return extract_frames(fp, out_dir=out_dir, time_interval=time_interval, max_samples=max_samples)


def main(root_dir_pth: Path,
         time_interval: Optional[float] = None,
         verbose: bool = True):  
    
    cfg = get_config()
    if verbose:
        print('Searching for video files')
    if not time_interval: 
        time_interval = cfg.IMAGE_INTERVAL
    frames_dir = root_dir_pth / 'Temp_Frames'
    video_extensions = ('*.[aA][vV][iI]', '*.[mM][pP]4', '*.[mM][kK][vV]', '*.[mM][oO][vV]')
    vid_paths = [f for pattern in video_extensions for f in root_dir_pth.rglob(pattern)]
    print(f'Extracting {len(vid_paths)} video files to jpg images')
    vid_path_tuples = [(idx, path) for idx, path in enumerate(vid_paths)]
    
    start_time = time.time()
    if vid_paths:
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True) 
        try: 
            #using multiprocessing instead of joblib so the code runs on windows with pyinstaller
            args_list = [(fp, frames_dir, time_interval, cfg.MAX_VID_SAMPLES) for fp in vid_path_tuples]
            num_workers = min(8, cpu_count())
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap_unordered(extract_wrapper, args_list), total=len(args_list)))
            cv2.destroyAllWindows()
        except Exception as main_exception:
            print(f"Error in the main function: {main_exception}")
            traceback.print_exc()

    total_time = time.time()-start_time
    if verbose:    
        print(f'Total video processing time was {total_time}') 


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    print(f'the project directory is {project_dir}')
    videos= project_dir / "data/vids_small"
    main(videos)
