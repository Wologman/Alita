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
    classes_dict = {i + 1: string for i, string in enumerate(sorted_list)}
    return classes_dict


def get_new_predictions(md_preds, cl_preds):
    new_preds = []
    for item in md_preds:    #[:2]:
        if item['file'] not in cl_preds:
            continue
        new_item = {}
        if 'detections' in item:
            conf_scores = [detection['conf'] for detection in item['detections']]
            if len(conf_scores) > 0:
                max_detection_idx = np.argmax(np.array(conf_scores))
                selected_detection = item['detections'][max_detection_idx]
                selected_bbox = selected_detection['bbox']
            else:
                selected_bbox = [0,0,1,1]
        else:
            selected_bbox = [0,0,1,1]
            
        new_item['file'] = item['file']
        new_category = cl_preds[item['file']][0]
        new_confidence = cl_preds[item['file']][1]
        new_detections = [{'category': new_category,
                           'conf': new_confidence,
                           'bbox': selected_bbox}]
        new_item = {'file':item['file'], 'detections': new_detections}
        new_preds.append(new_item)
    return new_preds

# ----------------------------------- Main Process-----------------------------------------
# ----------------------------------------------------------------------------------------

def main(df, image_fldr, md_source, classes):
    '''The goal here is to use the predictions dataframe from the classifier, to modify the original json file from 
    the MegaDetector, so that visualisation tools designed for MegaDetector can be used.'''
    
    def remove_folder_path(file_path, folder_path):
        try:
            file_path_obj = Path(file_path)
            relative_path = file_path_obj.relative_to(folder_path)
            return str(relative_path)
        except ValueError:
            folder_path_str = str(folder_path)
            return file_path.replace(folder_path_str, '', 1).lstrip('\\/')

    md_img_preds, classes_dict, json_data = load_json(md_source)
    print(Colour.S + 'The parent directory for all the images: ' + Colour.E, str(image_fldr)) 
    print(Colour.S + 'Original MegaDetector Classes: ' + Colour.E, classes_dict)
    print(Colour.S + '\nThe first two image predictions from the MegaDetector output:' + Colour.E )
    print(md_img_preds[:2], '\n')
    
    if 'empty' not in classes:
        classes.append('empty')
    
    new_classes_dict = get_classes_dict(classes)   
    print(Colour.S + 'TROUBLESHOOTING ONLY, REMOVE LATER' + Colour.E)
    print(f'The file object passed to predictions2json is: {image_fldr}')
    print(f'Then it is matched as  {image_fldr.name}')

    print(f'Converting {image_fldr.name}')
    df['File_Path'] = df['File_Path'].apply(lambda x: remove_folder_path(x, image_fldr))
    reverse_map = {v: k for k, v in new_classes_dict.items()}
    df['Prediction'] = df['Encounter'].map(reverse_map).fillna(0).astype(int)
    df = df[['File_Path', 'Prediction','Max_Prob']]
    print(Colour.S + '\nThe predictions dataframe after modifying to match MegaDetector syntax:' + Colour.E )
    print(df.head(3))
    
    classifier_preds = {row[0]: (row[1], row[2]) for row in df.itertuples(index=False, name=None)}
    print(Colour.S + '\nThe first three image predictions from Alita, as a dictionary:' + Colour.E )
    [print(item) for i, item in enumerate(classifier_preds.items()) if i < 3]

    prediction_data = get_new_predictions(md_img_preds, classifier_preds)
    json_data['images'] = prediction_data
    json_data['detection_categories'] = new_classes_dict
    
    with open(image_fldr / 'alita_predictions.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

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
                  classes)