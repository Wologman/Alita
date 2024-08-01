# Alita v5.0

This file is written in Markdown.  It looks better in a Mardown Interpreter.  If you don't have one already I suggest Obsidian, or VSCode.

**If you simply want to download and run the inference code on a Windows computer then the deployment code along with all the model weights is available [here](https://drive.google.com/drive/folders/1UWStcmoF3qRWodygs3mmQhvfmg2ALZj0), and also linked from [wekaresearch.com](https://wekaresearch.com)**


## Summary
Alita is a machine vision model created to automate the classification of New Zealand camera trap imagery.  It was made by  [Olly Powell](https://wekaresearch.com) for the [Department of Conservation](https://www.doc.govt.nz/) (DOC) national predator control program. The current version is trained with the 86 classes listed below:

['banded_dotterel', 'banded_rail', 'bellbird', 'black_backed_gull', 'black_billed_gull', 'black_fronted_tern', 'blackbird', 'canada_goose', 'cat', 'chamois', 'chicken', 'cow', 'crake', 'deer', 'dog', 'dunnock', 'fantail', 'ferret', 'finch', 'fiordland_crested_penguin', 'fluttering_shearwater', 'goat', 'grey_faced'_'petrol', 'grey_warbler', 'hare', 'harrier', 'hedgehog', 'horse', 'human', 'kaka', 'kea', 'kereru', 'kingfisher', 'kiwi' 'little_blue_penguin', 'magpie', 'mallard', 'mohua', 'morepork', 'mouse', 'myna', 'nz_falcon', 'oystercatcher', 'paradise_duck', 'parakeet',  'pateke', 'pheasant', 'pig', 'pipit', 'plover', 'possum', 'pukeko', 'quail', 'rabbit', 'rat', 'redpoll', 'rifleman', 'robin', 'rosella', 'sealion', 'sheep', 'shore_plover', 'silvereye', 'sparrow', 'spotted_dove', 'spurwing_plover', 'starling', 'stilt', 'stoat', 'swallow', 'swan', 'tahr', 'takahe', 'thrush', 'tieke', 'tomtit', 'tui', 'wallaby', 'weasel', 'weka', 'welcome_swallow', 'white_faced_heron', 'whitehead', 'wrybill', 'yellow_eyed_penguin', 'yellowhammer']

## How it works
Alita is actually the combination of two seperate models, both using the PyTorch deep learning framework. 
1. [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)   - A YOLOv5 object detection model.  From this we determine if the image is empty, and predict where the centre of the animal is located.
2. A custom classifier based on EfficientNetV2L.  The 480 pixels around the predicted animal centre cropped and fed to the neural network.  If multiple images are taken in quick succession, then the most confident prediction is made and referred to as one 'encounter'.  In the case of video, one video clip is considered a single encounter.

The advantage to this approach is the fine-grained classification is done efficiently with small images but without any loss of pixels in the region of interest. Some of the animals being classified here are only a few pixels across, if the whole image was downsized there would not be much information left to work with.

Since these two models each have their own Python environments and dependencies, they are run sequentially.  PowerShell scripts handle the whole process, including the transition.  Future versions will be less complex.  We followed this path as initially it wasn't clear what role, if any, object detection would play in inference, and also to ensure future versions of MegaDetector to be incorporated easily if needed.  

## Usage

### Requirements:
- Windows 10 or later
- NVIDIA GPU with minimum 4Gb VRAM (Or it will do CPU-only processing slowly)
- [miniConda](https://docs.anaconda.com/miniconda/), miniForge or Anaconda

### Setup
1.  Install [miniConda](https://docs.anaconda.com/miniconda/), make sure it is included in the Path variable (probably just a box tick on install)
2. Move this folder to where you want the code to sit permanently.  I suggest somewhere without troublesome file paths, such as a root directory (eg. `c:\\`).

3.  Find the *Run_On_Windows* folder, and click on *Setup_Everything.bat*

I haven't tested on Linux, but in theory it can be used you have PowerShell set up.  

### Inference
1. Run_On_Windows > Infer_Dataset.bat
2. Find the folder containing your images or video, copy as path, paste the path into the terminal.
3. The output should be two .csv files, located in the image folder.  One simply has the prediction for the encounter, the second has the full probability scores for every image along with bounding box info.


## Performance
Scores below are for all 86 classes.  The go up if post-processing merges classes (for example calling both rat and mouse a rodent).  

On randomly split samples from the same pool as the training samples:

- Balanced Accuracy (BA):  **0.78**  
- macro Average Precision (mAP):  **0.86**  
- F1  **0.82**  

The above scores are an over-estimate of real-world performance, as we only randomly split this test set from the total collection of samples. Since they came from the same original cameras they are somewhat correlated to each other. To reduce this data leakage we hold a second test set aside (not publicly available) from entirely different camera locations.  These more realistic scores are below:

- BA:  **0.683**   
- mAP:  **0.76**     	
- F1:  **0.70**  

I listed the two sets of scores to highlight the importance of image diversity and careful selection of validation schemes when working with camera trap data.

### Retraining
In principle you can re-train on windows without any Python coding, from the I have shell scripted this too, all that would need changing is the settings file.  You then run `Train_Evaluate_Log.py`.  

You can even re-train multiple times without intervention by setting up a cue of settings files.

I was exploring ways to make complex Python code accessable within an organisation that doesn't necessarily have much Python expertise.

In practice, this is a much more advanced use-case and has not had nearly as much debugging.  I would be a little surprised if it ran smoothly first time in a completely different setting with a new dataset.  

If I was to start this project from scratch I would have set up all the training scripts in Linux, and just made the inference for Windows.  Then installed a dual booting system, or WSL on the relevent Windows machine.

## Links and Related Work

- [The entire collection of DOC camera trap training imagery](https://lila.science/datasets/nz-trailcams).  Alita was trained on a subset of this imagery. 
- [Pytorch Wildlife](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)  
- [Peter van Lunteren's projects](https://addaxdatascience.com/projects/)

### Future Improvements
We have a TODO list for various improvements, but if you have further suggestions, please contact Olly (wekaresearch.com) or Joris at DOC.  Some plans we already have include:

- Integrate the MegaDetector model, rather than running as a separate  script with it's own dependencies and Python environment.  When this project was started it wasn't clear we would continue to use MD.  Now we're past that point it makes sense to integrate it more cleanly.
- Upgrade to MegaDetector V6.  This should improve our accuracy on empties, as well as inference speed.
- Package the set up and dependencies into a `.exe` file
- A Linux version
- Checkpointing.   Currently if you have a crash or a power cut in the middle of processing a million images, you're stuffed.  Though if the MegaDetector part has completed, you can leave the `.json` file in place and re-start and at least that stage will be skipped.  MegaDetector is the slowest step at the moment.
- Implement a secondary pass to check for empties.  The most common error on real world use cases is false predictions for the empty class, since typically this is by far the most common class.  So this is the lowest hanging fruit in terms of usefulness with regards to performance.
- Image hashing, localised on the region around the predicted region of interest.  So we can store some metadata for the uniqueness of each image, estimate the value of the image to DOC, and use for decisions on use for future training, or validation.

## Acknowledgements
- Joris Tinnemans, for his tireless energy getting this work started, and coordinating the dataset curation and processing. 
- Jan Hewton and Jane Stevens, who between them manually checked most of our database of more than 2.5 million images.
- A long list of parties that supplied additional datasets, including those on [Lila Science](https://lila.science/).
- All our volunteers and rangers who collected images from more than 30 regions in New Zealand.
- Dan Morris and his team, for his work producing and maintaining the MegaDetector.
- The folks in the Threats Science and NPCP teams at DOC for their encouragement and support.
