'''If running directly, 
The goal here is to take a predictions file, and the original MD output file, and produce a modfied MD output json file with the new classes and predictions
'''
import json
from pathlib import Path
import pandas as pd
import numpy as np

class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def load_json(json_path):
    """Opens a single json file and loads into an array of dictionaries, returns that array
    each dict has the keys 'file', 'max_detection_conf', 'detections' """
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        images_array = json_data['images']
        #info_array = json_data['info']
        classes = json_data['detection_categories']
        if __name__ == '__main__':
            print(Colour.S + "Number of images in the images array:" + Colour.E, len(images_array))
    return  images_array, classes, json_data  #info_array,


def get_classes_dict(class_list):
    sorted_list = sorted(class_list)
    classes_dict = {i+1: string for i, string in enumerate(sorted_list)}
    return classes_dict


def get_new_predictions(md_preds, cl_preds):
    for idx, item in enumerate (md_preds):    #The image file level
        standard_file = Path(item['file']).as_posix()
        if standard_file not in cl_preds:
            continue

        new_category = str(cl_preds[standard_file][0])
        classify_prob = cl_preds[standard_file][1]
        classification = [[new_category, classify_prob]]
        fallback_detection = {'category': '1',
                              'conf': 0,
                              'bbox': [0,0,1,1],
                              'classifications': classification}
        if 'detections' in item:
            conf_scores = [detection['conf'] for detection in item['detections']]
            if len(conf_scores) > 0:
                max_conf = np.max(np.array(conf_scores))
                max_det_idx = np.argmax(np.array(conf_scores))                
                md_preds[idx]['detections'][max_det_idx]['classifications'] = classification
                md_preds[idx]['detections'][max_det_idx]['prev_conf'] = max_conf
                md_preds[idx]['detections'][max_det_idx]['prev_category'] = '1'
            else:
                md_preds[idx]['detections'] = [fallback_detection]
                md_preds[idx]['detections'][0]['classifications'] = classification
                md_preds[idx]['detections'][0]['prev_conf'] = 0
                md_preds[idx]['detections'][0]['prev_category'] = '1'

        else:
            md_preds[idx]['detections'] = [fallback_detection]
            md_preds[idx]['detections'][0]['classifications'] = classification
            md_preds[idx]['detections'][0]['prev_conf'] = 0
            md_preds[idx]['detections'][0]['prev_category'] = '1'
    return md_preds


# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------

def main(df, 
         root_fldr, 
         md_source,
         classes=[],
         destn_path=None, 
         verbose=True):
    '''The goal here is to use the predictions dataframe from the classifier, to modify the original json file from 
    the MegaDetector, so that visualisation tools designed for MegaDetector can be used.'''

    def standardise_paths(file_path, folder_path):
        try:
            file_path_obj = Path(file_path).resolve()
            folder_path_obj = Path(folder_path).resolve()
            relative_path = file_path_obj.relative_to(folder_path_obj)
            return relative_path.as_posix()
        except ValueError:
            # fallback if relative_to fails
            file_path_str = Path(file_path).as_posix()
            folder_path_str = Path(folder_path).as_posix()
            if file_path_str.startswith(folder_path_str):
                return file_path_str[len(folder_path_str):].lstrip('/')
            else:
                return file_path_str

    special_class = classes[0] if len(classes) == 1 else None

    md_img_preds, classes_dict, json_data = load_json(md_source)
    if verbose:
        print(Colour.S + 'The parent directory for all the images: ' + Colour.E, str(root_fldr)) 
        print(Colour.S + 'Original MegaDetector Classes: ' + Colour.E, classes_dict)
        print(Colour.S + '\nThe first two image predictions from the MegaDetector output:' + Colour.E )
        print(md_img_preds[:2], '\n')
 
    if special_class is None:
        for special_case in ['Empty', 'Unknown']:
            if special_case not in classes:
                classes.append(special_case)
                classes.sort()

    new_classes_dict = get_classes_dict(classes)  
    old_classes_dict = {"1": "Unclassified Animal",
                        "2": "Human",
                        "3": "Vehicle"}

    df['File_Path'] = df['File_Path'].apply(lambda x: standardise_paths(x, root_fldr))
    reverse_map = {v: k for k, v in new_classes_dict.items()}

    if special_class is None:  #The special class plus 'Empty' and 'Unknown'
        #In this behaviour we're working with what ever class was the maximum prediction for that photo
        df['Prediction'] = df['Prediction'].map(reverse_map).fillna(0).astype(int)  #Turn the text strings to integers 
        df = df[['File_Path', 'Prediction','Probability']]
    else:
        print(f'The special class is {special_class}')
        df['Prediction'] = 1
        df = df[['File_Path', 'Prediction', special_class]]

    classifier_preds = {row[0]: (row[1], row[2]) for row in df.itertuples(index=False, name=None)}
    if verbose:
        print(Colour.S + '\nThe first three image predictions from Alita, as a dictionary:' + Colour.E )
        [print(item) for i, item in enumerate(classifier_preds.items()) if i < 3]

    prediction_data = get_new_predictions(md_img_preds, classifier_preds)
    json_data['images'] = prediction_data
    json_data['detection_categories'] = old_classes_dict
    json_data['classification_categories'] = new_classes_dict
    json_data["forbidden_classes"] = []

    if destn_path is None:
        destn_path = root_fldr / 'alita_predictions.json'
    
    with open(destn_path, 'w', encoding='utf8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    return

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    md_out_path = "E:\Alita\Data\corrupted_copy\mdPredictions.json"
    parent =  Path("E:\Alita\Data\corrupted_copy")
    predictions = pd.read_csv("E:\Alita\Data\corrupted_copy\Exp_26_Run_02_predictions.csv")
    classes = ["banded_dotterel", "banded_rail", "bellbird", "black_backed_gull", "black_billed_gull", 
               "black_fronted_tern", "blackbird", "canada_goose", "cat", "chamois", "chicken", "cow", 
               "crake", "deer", "dog", "dunnock", "fantail", "ferret", "finch", "fiordland_crested_penguin", 
               "fluttering_shearwater", "goat", "grey_faced_petrol", "grey_warbler", "hare", "harrier", 
               "hedgehog", "horse", "human", "kaka", "kea", "kereru", "kingfisher", "kiwi", 
               "little_blue_penguin", "magpie", "mallard", "mohua", "morepork", "mouse", "myna", "nz_falcon", 
               "oystercatcher", "paradise_duck", "parakeet", "pateke", "pheasant", "pig", "pipit", "plover",
               "possum", "pukeko", "quail", "rabbit", "rat", "redpoll", "rifleman", "robin", "rosella", "sealion", 
               "sheep", "shore_plover", "silvereye", "sparrow", "spotted_dove", "spurwing_plover", "starling", 
               "stilt", "stoat", "swallow", "swan", "tahr", "takahe", "thrush", "tieke", "tomtit", "tui", "wallaby", 
               "weasel", "weka", "welcome_swallow", "white_faced_heron", "whitehead", "wrybill", 
               "yellow_eyed_penguin", "yellowhammer"]
    output = main(predictions, 
                  parent, 
                  md_out_path, 
                  classes=[],
                  destn_path=None,
                  verbose=True)