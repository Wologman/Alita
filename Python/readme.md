# Wildlife Classifier Code

## Summary
This Python code builds a machine vision model in PyTorch optimised for working with large collections of classified images, from camera  & video footage.  It uses the MegaDetector YoloV5 object detection model to localise the animals, and crop to a uniform size, prior to a seperate classification step.  Training performed here is only for the second step, allowing a classifier to be made with New Zealand's unique combination of fauna and flora, but benefit from the large bounding box annotated dataset that MegaDetector was built from.


## Usage

### Setup
- Install MiniConda
- Run_On_Windows > Setup_Everything.bat
- I haven't tested on Linux, but in theory it should work if you have PowerShell set up

### Inference
1. Run_On_Windows > Setup_Everything.bat
2. Find a settings .yaml file from the Settings folder, copy as path, paste into the terminal
3. Find the folder containing your images or video, copy as path, paste the path into the terminal.
4. The output should be two .csv files, located in the image folder.  One simply has the prediction for the encounter, the second has the full probability scores for every image along with bounding box info.

### Training 
Most people will never need this, and if you are doing this, you'd probably prefer to run it through an IDE instead.  But anyway, here is the process:
1. Ensure the settings file is doing all the things you want, pointing to the right places etc.  You will at least need to update the experiment and run ID.
2. Run_On_Windows > Train_Evaluate_Log.bat
3. Point to the new settings file.
3. Training takes approximately 24 hours, or longer if you're re-scoring millions of new images with MegaDetector.  

## Code
In order of use, this is what each script does:

1. `Train_Evaluate_Log`: Runs all the files below, or at least the ones it needs to according to various flags.  This file needs the input of a `.yaml` settings file, to determine the Experiment ID, and Run ID, as well as various hyperparameters, data choices & class names.
2. `MegaDetector_Setup.ps1`:  Downloads the MegaDetector repository, and sets up the required Python environment.
2. `Make_Conda_Environmnent.ps1` Creats the Python environment for the classifier.
3. `MegaDetector_Run.ps1` Activates the MegaDetector Python environment and runs MegaDetector with parameters passed in as flags from other scripts.
4. `Reload_Images.py` Searches through previous MegaDetector outputs, looks through the dataset, and compares the two.  Any new image files are then copied to another location (ideally on a fast SSD), and MegaDetector is run to produce an annotations `.json` file
5. `InterpretJSON.py` Runs through all the megadetector `.json` outputs in a single folder, and consolidates them to a single dataframe, then saved as  a `.parquet` file.  Also goes through any images missing their EXIF data, and extracts that to a different `.parquet` file, which is joined to the previous one and passed on to the data exploration and cleaning steps.
6. `Data_Exploration.ipynb` Does some data analysis on the newly produced dataframe, generates some useful statistics about the dataset.  This file isn't actually an essential part of the processing, but useful for monitoring, It is a little out of date.
7. `Clean_Data.py` Does all the data cleaning steps, and saves out a cleaned datafile as a `.parquet`
8. `Preprocess_Images.py` Uses the now cleaned file, with selected samples, opens them from the long term storage, crops to a pre-determined box and size, then saves out the new much smaller dataset to fast storage.  At this point the dataset has been reduced from many TB to approximately 20Gb.
9. `Training.py` Trains the new PyTorch classifier model with the PyTorch Lightning framework.  The output is a model file, plus some perfomance metric data.
10. `Model_Evaluation.ipynb` A notebook to analyse the performance of the new model.  Looks at training metrics, and performance against randomly held out images, and also against several camera locations not used for training.
11. `Inference.py` This script simply runs inference by being pointed to a folder, produces a detailed file in `.parquet` format, and a basic one with just the photo exif time-stamp, most probable class, and the probability, in `.csv` format.