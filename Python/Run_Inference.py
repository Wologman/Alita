'''This is the master file that:
- Accepts parameter arguments from shell scripts, specifically the settings file and dataset filepath
- Imports and runs the inference script.
'''
import Inference
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataPath", type=str, help="Filepath to the root directory of imagery to be processed")
parser.add_argument("--settingsPath", type=str, help="Filepath to the settings YAML file for this model")
args = parser.parse_args()

# Add some steps in here to detect 

print(f'Running Inference.py on {args.dataPath}, with the settings file {args.settingsPath}')
Inference.main(args.settingsPath, args.dataPath, )