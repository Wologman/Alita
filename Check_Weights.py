'''Ensures that weights files for both the classifier and the MegaDetector exist, and are valid pyTorch files'''

import torch
#from torchvision import models

class Warn:  #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Colour:  #bold red
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'
    

def check_weights(weight_type, filepaths, load=False):
        if filepaths:
            filepath = filepaths[0]

            
            folders = filepath.parts
            last_six_folders = folders[-min(6, len(folders)):]
            display_path = '/'.join(last_six_folders)
            if load:
                try:
                    _ = torch.load(filepath)
                    print(Colour.S + f'{str(display_path)} is a valid PyTorch weights file for the {weight_type}' + Colour.E)
                except Exception as e:
                    print(Warn.S + f'There was an error loading the weights file at {str(display_path)} for teh {weight_type}' + Warn.E)
                    print(e)
            else:
                print(Colour.S + f'{str(display_path)} is the weights file for the {weight_type}' + Colour.E)
        else:
            print(Warn.S + f'No valid weights file was found for the {weight_type}' + Warn.E)