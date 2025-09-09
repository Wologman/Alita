# Alita Python Directory

## Summary
This Python code builds a machine vision model in PyTorch optimised for working with large collections of classified images, from camera  & video footage.  It uses the MegaDetector YoloV5 object detection model to localise the animals, and crop to a uniform size, prior to a seperate classification step.  Training performed here is only for the second step, allowing a classifier to be made with New Zealand's unique combination of fauna and flora, but benefit from the large bounding box annotated dataset that MegaDetector was built from. 

## Code
In order of use, this is what each script does:

1. `train_evaluate_log`: Runs all the files below, or at least the ones it needs to according to various flags.  This file needs the input of a `.yaml` settings file, to determine the Experiment ID, and Run ID, as well as various hyperparameters, data choices & class names.
2. `reload_images.py` Searches through previous MegaDetector outputs, looks through the dataset, and compares the two.  Any new image files are then copied to another location (ideally on a fast SSD), and MegaDetector is run to produce an annotations `.json` file
3. `interpretJSON.py` Runs through all the megadetector `.json` outputs in a single folder, and consolidates them to a single dataframe, then saved as  a `.parquet` file.  Also goes through any images missing their EXIF data, and extracts that to a different `.parquet` file, which is joined to the previous one and passed on to the data exploration and cleaning steps.
4. `data_exploration.ipynb` Does some data analysis on the newly produced dataframe, generates some useful statistics about the dataset.  This file isn't actually an essential part of the processing, but useful for monitoring, It is a little out of date.
5. `clean_data.py` Does all the data cleaning steps, and saves out a cleaned datafile as a `.parquet`
6. `prioritise_data.py` Various methods to sub-sample from the training images to reduce class imbalance.
6. `preprocess_images.py` Uses the now cleaned file, with selected samples, opens them from the long term storage, crops to a pre-determined box and size, then saves out the new much smaller dataset to fast storage.  At this point the dataset has been reduced from many TB to approximately 20Gb.
7. `training.py` Trains the new PyTorch classifier model with the PyTorch Lightning framework.  The output is a model file, plus some perfomance metric data.
8. `model_evaluation.ipynb` A notebook to analyse the performance of the new model.  Looks at training metrics, and performance against randomly held out images, and also against several camera locations not used for training.
9. `inference.py` This script simply runs inference by being pointed to a folder, produces a detailed file in `.parquet` format, and a basic one with just the photo exif time-stamp, most probable class, and the probability, in `.csv` format.
10. `detection.py` Runs what ever model is being used for object detection.  Initially this was Dan Moris's version of MegaDetector.  Then I tried the Pytorch Wildlife fork, before going back to Dan's.  Either way, the idea was to save the output in the same .json format as standard from MegaDetector.
11. `predictions_2_JSON.py`  Reformats the predictions dataframe into the `.json` format expected from Timelapse.
12. `alita_gui.py`  Sets up a simple PyQT GUI to run inference.
