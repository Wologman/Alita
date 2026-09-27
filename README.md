# Alita v3.03

**If you simply want to download and run the inference code on a Windows computer then the deployment code along with all the model weights is [abailable from my pCloud drive](https://filedn.eu/l1723vRFnsquJMoK85UThX0/Alita_Windows_App/)**

## About Alita

Alita is a combination of two deep learning models that predicts the presence or absense of 81 classes of animal from camera trap images or video.  The classification step was made by Olly Powell for the Department of Conservation.  Alita works best on still images typical of DOC's standard trailcam setup.

Under the hood, there are two separate models.  The first is an animal detection step, based on [Dan Morris's MegaDetector](https://github.com/agentmorris/MegaDetector).  This produces a  file `detections.json`, which predicts a bounding box around an animal within an image.  Only bounding boxes with prediction scores over 0.05 are used.

The second stage is a pure classification step, that takes only the highest scoring bounding box, and crops its own a box of 480x480 pixels around the centroid.  This crop is passed through a second neural network.  This network predicts the presence or absence of 81 species independently.  Predictions are treated as 'multi-label', and do not sum to 1.

You can work directly with those prediction scores, or use the provided post-processing logic, which gives the following possible outcomes:
* Any of 81 animals, from the maximum score of all images taken within 30 seconds of each other.  This prediction is labelled an 'encounter'.
* OR  'Empty',  where **Both** models were below thresholds their preset thresholds.  
* OR  'Unknown', where the detection model predicted an animal witha score over 0.15, but the classifier provided no scores over the classifier threshold (selectable by the user in the GUI).

The model is imperfect, and will certainly contain bias.  If used carefully it should be orders of magnitude faster than manually checking images, for only a small loss in accuracy.  Typically one class score will be over 0.9 and the rest very small.  However the perfomance can vary with location and camera setup.  Some classes are more easily confused than others (for example cats & possums look similar from some angles).

If your goal is to locate a specific specific but sparce species.  For example you are trying to hunt down every last rat in an island sanctuary, then you can chose a relatively low classification threshold, like 0.3, and accept that you may have some false identifications that need manual checking.

If your goal is to monitor relative change in populations, then some form of calibration and manual sampling & calibration is needed.  At a minimum you could set a relatively high threshold like 0.6 and go through the 'Unknown' predictions to investigate sources of error.  

To go a step further you could create your own independent test set by sub-sampling from your study population and calibrate the model against that.  For example you could use [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) (fit a logistic regression model) for each class to transform the scores into a meaningful estimate of probability, before the agregation into an 'encounter'.  You could then do the same for competing models and compare the results.

## Instructions for Use

* The zipped folder should contain everything needed to run Alita on a Windows desktop.  Just download and unzip to any convenient location.

* If you right-click on `launch_alita.exe` you could create a shortcut on your system tray or the start menu.

* There is no installation required.  To remove the program simply delete the folder.

* If your machine has an NVIDIA GPU, it is possible to use that for increased speed by selecting the check-box in the GUI.

* Follow the various prompts to run the tool.   It will produce three files.
    - `xxxx_full_predictions.csv`:  A CSV with prediction scores for all animals, the top-3 animals, bounding boxes, and a column named *Encounter* where the top animal from all the images within a short burst of images.
    - `xxxx_predictions.csv`:   Only the *Encounter* and it's score.
    - `alita_predictions.json`:  A file in the format required to visualise the results in [Timelapse](https://timelapse.ucalgary.ca/)

*  If you are interested one particular species, you can select it from a dropdown box and an additional `.csv` and `.json` file will be produced with the prediction scores for that species only.  For example, if you select 'Weka'  everything is a Weka, but with varying scores.  You could then play with different threshold settings in [Timelapse](https://timelapse.ucalgary.ca/).

* This was treated as a multi-label problem, the predictions are independent of each other and the scores do not necessarily sum to 1. In principle you could predict two species in the same image, though one of them would likely be wrong as this is very rare.

## The Data

* The dataset used to make this model is available on [LILA BC](https://lila.science/).  It has come from a variety of sources, and has been collated by Joris Timmermans, with the awesome help of our two dedicated volunteers Jan and Jane.

*  The model was trained on only a subset of this data, to address class embalance, whilst retaining maximum feature diversity.  Olly intends to make public the methods and Python code he has been developing for this process.

* Work on evaluating accuracy, adding new classes, additional training data is ongoing.  In particular Olly is interested in improving the variance in behaviour between test sites and setups, as we are trying to predict relative change.


## Acknowledgements
*  Joris Tinnemans, for his tireless energy getting this work started, and coordinating the dataset curation and processing. 
*  Jan Hewton and Jane Stevens, who between them manually checked most of our database of more than 3 million images.
*  A long list of parties that supplied additional datasets, including those on [Lila Science](https://lila.science/).
*  All our volunteers and rangers who collected images from more than 30 regions in New Zealand.
*  Dan Morris and his team, for his work producing and maintaining the MegaDetector.
*  The folks in the Threats Science and NPCP teams at DOC for their encouragement and support.


