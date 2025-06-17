# Alita v3.0

This file is written in Markdown.  It looks better in a Mardown Interpreter.  If you don't have one already I suggest Obsidian, or VSCode.

**If you simply want to download and run the inference code on a Windows computer then the deployment code along with all the model weights is linked from [wekaresearch.com](https://wekaresearch.com)**


## About Alita

Alita is a combination of two deep learning models that predicts the presence or absense of 77 classes of animal from camera trap images or video.  The classification step was made by Olly Powell for the Department of Conservation.  Alita works best on still images typical of DOC's standard trailcam setup.

Under the hood, there are two main stages to the process.  The first is an animal detection step, based on [Dan Morris's MegaDetector](https://github.com/agentmorris/MegaDetector).  This produces a  file `detections.json`, which predicts a bounding box around an animal within an image.  Only bounding boxes with probability scores over 0.05 are used.

The second stage is a pure classification step, that takes only the highest probability bounding box, and crops its own a box of 480x480 pixels around the centroid.  This crop is passed through a second neural network.  This network predicts the presence or absence of 77 species independently.  They are treated as 'multi-label' predictions, and do not sum to 1.

The possible outcomes are:
* Any of 77 animals,  all the prediction scores are provided in the CSV.
* Any of 77 animals, from the maximum score of all images taken within 30 seconds of each other.  This prediction is labelled an 'encounter'.
* OR  'Empty',  where **Both** models were below thresholds their preset thresholds.  
* OR  'Unknown', where the detection model predicted an animal witha score over 0.15, but the classifier provided no scores over the classifier threshold (selectable by the user in the GUI).


The model is imperfect, but if used carefully it should be orders of magnitude faster than manually checking images, for only a small loss in accuracy.  Typically one class will be over 0.9 and the rest very small, making the threshold choice unimportant.  However in some terrain the model does not do as well as expected.  At the time of writing, the accuracy on goats was noteably inconsistent.

If your goal is to locate a specific specific but sparce species.  For example you are trying to hunt down every last rat in an island sanctuary, then you can chose a relatively low classification threshold, like 0.3, and accept that you may have some false identifications that need manual checking.

If your goal is to monitor relative change in populations, then you should use a higher threshold like 0.6.  This will reduce false positives and help to keep your outcomes for each class more independent.  You could also go through the 'Unknown' predictions to investigate sources of error.

## Instructions for Use

* The zipped folder should contain everything needed to run Alita on a Windows desktop.  Just download and unzip to any convenient location.

* If you right-click on `launch_alita.exe` you could create a shortcut on your system tray or the start menu.

* There is no installation required.  To remove the program simply delete the folder.

* If your machine has an NVIDIA GPU, it is possible to use that for increased speed by selecting the check-box in the GUI.

* Follow the various prompts to run the tool.   It will produce three files.
    - `xxxx_full_predictions.csv`:  A CSV with probability scores for all animals, the top-3 animals, bounding boxes, and a column named *Encounter* where the top animal from all the images within a short burst of images.
    - `xxxx_predictions.csv`:   Only the *Encounter* and it's score.
    - `alita_predictions.json`:  A file in the format required to visualise the results in [Timelapse](https://timelapse.ucalgary.ca/)

*  If you are interested one particular species, you can select it from a dropdown box and an additional `.csv` and `.json` file will be produced with the probability scores for that species only.  For example, if you select 'Weka'  everything is a Weka, but with varying probability.  You could then play with different threshold settings in [Timelapse](https://timelapse.ucalgary.ca/).

* This was treated as a multi-label problem, the predictions are independent of each other and the scores do not necessarily sum to 1. In principle you could predict two species in the same image, though one of them would likely be wrong as this is very rare.

## The Data

* The dataset used to make this model is available on [LILA BC](https://lila.science/).  It has come from a variety of sources, and has been collated by Joris Timmermans, with the awesome help of our two dedicated volunteers Jan and Jane.

*  The model was trained on only a subset of this data, to address class embalance, whilst retaining maximum feature diversity.  Olly intends to make public the methods and Python code he has been developing for this process.

* Work on evaluating accuracy, adding new classes, additional training data is ongoing.  In particular Olly is interested in improving the variance in behaviour between test sites and setups, as we are trying to predict relative change.


## Acknowledgements
*  Joris Tinnemans, for his tireless energy getting this work started, and coordinating the dataset curation and processing. 
*  Jan Hewton and Jane Stevens, who between them manually checked most of our database of more than 2.5 million images.
*  A long list of parties that supplied additional datasets, including those on [Lila Science](https://lila.science/).
*  All our volunteers and rangers who collected images from more than 30 regions in New Zealand.
*  Dan Morris and his team, for his work producing and maintaining the MegaDetector.
*  The folks in the Threats Science and NPCP teams at DOC for their encouragement and support.
