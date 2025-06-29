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



def safe_video_path(video_path: Path) -> Path:
    """If path contains non-ASCII, copy to temp file with ASCII name."""
    if any(ord(c) > 127 for c in str(video_path)):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=video_path.suffix)
        shutil.copy(video_path, temp.name)
        return Path(temp.name)
    return video_path


def compress_path(path: str) -> str:
    """Compress a file path and encode it as a filename-safe string."""
    utf8_bytes = path.encode("utf-8")                     # 1. Encode to bytes
    compressed = zlib.compress(utf8_bytes, level=9)       # 2. Compress with zlib
    b64_bytes = base64.urlsafe_b64encode(compressed)      # 3. Base64 encode
    return b64_bytes.decode("ascii")                      # 4. Decode to str for filename use


def decompress_path(encoded: str) -> str:
    """Decode a filename-safe string back to the original path."""
    b64_bytes = encoded.encode("ascii")                   # 1. Convert to bytes
    compressed = base64.urlsafe_b64decode(b64_bytes)      # 2. Decode from base64
    utf8_bytes = zlib.decompress(compressed)              # 3. Decompress
    return utf8_bytes.decode("utf-8")                     # 4. Decode to original string


def extract_frames(video_path_tuple: tuple,
                   out_dir: Optional[Path] = None,
                   time_interval: float = 0.5,
                   max_samples: int = 20
                   ):
    sub_dir_nm  = video_path_tuple[0]  #Should be a unique integer for every video
    video_path = Path(video_path_tuple[1])
    if out_dir is None:
        out_dir = video_path.parent / 'Frames'
    out_sub_dir = out_dir / f'video_{sub_dir_nm}'
    encoded_path = compress_path(str(video_path))

    out_sub_dir.mkdir(parents=True, exist_ok=True) 
    creation_time = get_video_creation_time(video_path)
    if creation_time is None:
        creation_time = dt.datetime(1977, 10, 22, 0, 0, 0)
    try: 
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        try:
            if not cap.isOpened(): 
                print("Bugger! Couldn't open video file")
                return
        except Exception as err:
            print(err)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
    except Exception as e:
        print(f"Error opening video file {str(video_path)}: {e}")
        return

    frame_interval = int(fps * time_interval)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frame_interval:
        frame_interval = total_frames -1
    frame_count = 0

    if (total_frames // frame_interval > max_samples):
        frame_interval = total_frames // max_samples
    
    for frame_count in range(0, total_frames, frame_interval):
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        except Exception as e:
            print(f"Error reading a frame from {str(video_path)} e: {e}")
            continue
        if not ret:
            break
        
        frame_name = f'{out_sub_dir}/{encoded_path}_{frame_count // frame_interval:04d}.jpg'  #fix this up for linux
        current_time = creation_time + dt.timedelta(milliseconds=(frame_count / fps) * 1000)
        datetime_str = current_time.strftime('%Y:%m:%d %H:%M:%S')
        try:
            cv2.imwrite(frame_name, frame)
            exif_dict = {"Exif": {piexif.ExifIFD.DateTimeOriginal: datetime_str.encode('utf-8')}}
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, frame_name)
        except Exception as e:
            print(f"Error writing frame: {e}")
        
        #I don't know why this is needed, but if I don't then some .avi files crash silently on cap.read() 
        if frame_count == 0:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
    cap.release()
        


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
            #Parallel(n_jobs=8)(delayed(extract_frames)(fp,
            #                                           out_dir=frames_dir,
            #                                           time_interval=time_interval,
            #                                           max_samples=cfg.MAX_VID_SAMPLES) 
            #                                                for fp in tqdm(vid_path_tuples))
            #for fp in tqdm(vid_path_tuples):
            #    extract_frames(fp,
            #                   out_dir=frames_dir,
            #                   time_interval=time_interval,
            #                   max_samples=cfg.MAX_VID_SAMPLES)

            args_list = [(fp, frames_dir, time_interval, cfg.MAX_VID_SAMPLES) for fp in vid_path_tuples]
            num_workers = min(8, cpu_count())
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap_unordered(extract_wrapper, args_list), total=len(args_list)))

            
            cv2.destroyAllWindows()
        except Exception as main_exception:
            print(f"Error in the main function: {main_exception}")
            traceback.print_exc()
        
        #for fp in tqdm(vid_path_tuples):
            #extract_frames(fp, out_dir=frames_dir, time_interval=time_interval, max_samples=cfg.MAX_VID_SAMPLES)
            
    total_time = time.time()-start_time
    if verbose:    
        print(f'Total video processing time was {total_time}') 
    #Benchmarking on Anja's 2018 kiwi monitoring data, (1627 videos, sample_interval 0.5, max samples=20, typically 30 seconds each)
    #Single-thread, iterating through every frame: 904 seconds  (15:07)
    #8-threads, iterating through every frame: 259 seconds (4:15)
    #Single-thread, directly decoding only the frames to be sampled: 655 seconds (10:55)
    #8-threas, directly decoding only the frames to be sampled: 203 seconds (3:23)

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    print(f'the project directory is {project_dir}')
    videos= project_dir / "data/vids_small"
    main(videos)
