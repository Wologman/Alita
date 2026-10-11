# Alita v3.03

If you simply want to download and run the desktop app on a Windows computer then the deployment code along with all the model weights is [abailable from my pCloud drive here](https://filedn.eu/l1723vRFnsquJMoK85UThX0/Alita_Windows_App/)

## About Alita

Alita is a combination of two deep learning models that predicts the presence or absense of 81 classes of animal from camera trap images or video.  The classification step was made by Olly Powell for the Department of Conservation.  Alita works best on still images typical of DOC's standard trailcam setup.

The first stage is an animal detection step, based on [Dan Morris's MegaDetector](https://github.com/agentmorris/MegaDetector).  This produces a  file `detections.json`, which predicts a bounding box around an animal within an image.  Only bounding boxes with prediction scores over 0.05 are used for the second stage.

The second stage is a pure classification step, that takes only the highest scoring bounding box, and crops its own a box of 480x480 pixels around the centroid.  This crop is passed through the second neural network.  This model predicts the presence or absence of 81 species independently.  Predictions are treated as 'multi-label', and do not sum to 1.

You can work directly with those prediction scores, or use the provided post-processing logic, which gives the following possible outcomes:
* Any of 81 animals, from the maximum score of all images taken within 30 seconds of each other.  This prediction is labelled an 'encounter'.
* OR  'Empty',  where **Both** models were below thresholds their preset thresholds.  
* OR  'Unknown', where the detection model predicted an animal witha score over 0.15, but the classifier provided no scores over the classifier threshold (selectable by the user in the GUI).

The raw model scores will certainly contain systematic bias, and the distribution between true positive scores and true negatives will overlap, meaning no mater where you put a binary threshold, there will be some false positives and false neagatives.  If used carefully it should be orders of magnitude faster than manually checking images, for only a small loss in accuracy.  Typically one class score will be over 0.9 and the rest very small.  However the perfomance can vary with location and camera setup.  Some classes are more easily confused than others (for example cats & possums in the dark can look similar from some angles).

If your goal is to locate a specific specific but sparce species.  For example you are trying to hunt down every last rat in an island sanctuary, then you can chose a relatively low classification threshold, like 0.3, and accept that you may have some false identifications that need manual checking.


## Instructions for use

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


## Calibration
The following argument applies to all deep learning camera trap models, not just Alita.

If your goal is to monitor **relative change** in populations, then some form of calibration is essential, and it makes no sense to work directly with the 'encounter' prediction as this is based on comparison between raw (biased) scores.  Also, it is easier for this use case to work individually with each class and ignore relative predictions between classes for a given image, even though in principle this may result in the counter-intuitive result of a single image predicting more than one species.

### Instance-level scaling
For most use cases we are estimating relative change, so instance level scaling should be sufficient so that the scaled scores from different classes can be fairly compared, and a single threshold used over all classes.  

A good starting point is would be [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) (fit a logistic regression model), with model predictions against ground truch for some pre-existing independent data.  

```P(y=1 | f) = 1 / (1 + exp(A·f + B))```
where *f* is the raw score, *y* is the ground truth (0 or 1), A & B are the slope and offset coefficients.   This also has the nice outcome of making the scaled model predictions interpretable as an estimate of probability.

If such data is unavailable or is too different from your study images, a *stratified random* sub-sample of your study data could be used, so long as you can label enough samples for each class of interest for a reasonable fit.

Note that if you do have good historical data to work with, this step is very cheap computationaly and requires no new human annotation time.  The calibration data can not come from DOC's publicly available training data from Lila.science, as Alita was trained on this.  At the time of the release of version 3.03 we did not have sufficient independent data to produce a useful collection of slope (A) and offset (B) coefficients, but this will hopefully be available with the next version.

### Calibration for estimating absolute counts

Suppose we require an absolute count for some reason, or need to estimate an uncertainty of that count.  For example we are trying to compare the sensitivity of two different camera setups, or even completely unrelated observation methods like cameras vs footprint tunnels.

Imagine a histogram of the true positive scores and true negatives.  If we have already scaled as above for instance level predictions the optimal theshold should be 0.5, and the true negatives and true posives for the bin at that value should be nearly equal.  However the shapes of the distributions have no reason to be symmetrical, meaning the area of false positives and false negatives, with the difference between them equal to the error in the absolute count.  Also it is entirely likely that our localised study images are subtly different in some unexpected way to those of the stored data used for the scaling, potentially adding to any asymmetry.

To correct for the above error, we could take a random sub-sample of the study data, then iteratively apply a temperature scaling step, or adjust the threshold slightly until the error is zero for that sub-sample. The advantage of temperature scaling rather than threshold adjustment is that you can leave the final threshold at 0.5 across all classes, and just sum up the 1s to get an estimate of the total, if that is how you aggregateed when performing the scaling step.  So the scores remain interperatable as something like probability.

To go a step further and give an agregate prediction with an uncertainty, you could implement bootstrapping with your random sub-sample.  

Now you have a solid estimate of your total, plus an uncertainty, based on a small sub-sample of your actual study data, with the model applied to the rest.  In doing so you have corrected for both internal model bias, and effects of any unexpected localised behaviour that may be occuring with this particualar season/location/camera setup/species variant etc.  You could even try to correct for the use of the model on a completely different species, if you had a model class that was very similar but not quite the same as the one in your study.


## The data

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


