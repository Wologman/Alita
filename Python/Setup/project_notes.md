# Project Notes
## Strategy
### MegaDetector
The MegaDetector repo is a fork of the YOLOv5 repo, frozen at a particular time.  YOLOv5 uses the YOLO (You Only Look Once) algorithm to perform object localisation in conjunction with a convolutional neural network for classification.  

In principle this method could perform the entire task, but this would mean re-training with our new dataset and classes, and we would need to annotate this dataset with bounding boxes for training.  In that case, we do not necessarily need the MegaDetector repo, we could use the original YoloV5 one, or newer implementations YoloV8, or YOLO_x, or a more customised object detection framework of our own design.

### This project
Use MegaDetector only for object localisation, taking advantage of it's training on a very large internationally sourced bounding box annotated dataset.  This means that images can be cropped to smaller sizes, trainable with 6Gb of GPU memory or less, but without loosing any important image detail (like the texture or fur for example).   Then a seperate classifier based on cropped images has been optimised.

As an extension, for inference an option could be added to evaluate larger images with a moving window, so eliminating the object localisation step, at the expense of processing speed.  This could also be implemented only for cases where no highly probable oject is detected, That would hopefully get the best of both approaches.

### Setup of a browser based GUI for the end users
I am thinking a simple browser-based GUI might be quite a nice way to tie everything together and keep the explanations where they are needed.  Just to run various shell scripts, or processes and make common settings adjustments.  Things that would be nice to run would be:

- Setup Megadetector (with environments, and repo downloads)
- Setup Classifier (environments only)
- Run training
- Run inference
- Open the settings file to manually edit anything at all, or for specific tasks, use the GUI only:
- Adjust file paths to original training datasources, then retrain
- Adjust file paths to the independent test set, then retrain, or just re-evaluate
- Adjust file paths to the inference test set
- Exclude some datasets
- Assign different datasets to the independent validation purpose
- Exclude some classes
- Combine some classes

### Other ideas
There are many ways to approach this problem, I have listed a few  other possibilities here for completion:

1. Ignore Megadetector, and build a classifier with the whole image, by re-sizing it down to something we could process with reasonable speed and memory usage.  The risk with this approach is that too much pixel detail would be lost on the objects of interest, effecting accuracy.  However we can test the concept simply by adjusting upward the size of the "buffer" created around the image crops.  It may be possible to improve this approach by introducing a learning step to the downsizing step.
2. Annotate a large collection of the training images and build a multiclass object detector of our own.
4. Use Megadetector to generate the training dataset, train the network, but for inference, work on smaller chunks of the larger image.  So only the classifier is needed for inference, we would not be less limited by the accuracy of Megadetector.  The obvious downside is that inference could end up being substantially slower.
5. Build our own object detector, using a more traditional machine vision approach, instead of deep learning. The point of difference with this appliction is that all the things we are interested in move. We could use the pixel differences over sequential bursts of photos to pick out movement.  This could potentially be less computationaly intensive than running a Neural Network object detection model.


## Speed Benchmarks
Things I've tried to increase speed: Multi-core dataloading, refactoring and switching to Pytorch Lightning, upgrading to Python 3.11, installing and using TurboJPEG.  At present, speed appears to be limited by the read-speed of the storage medium, so there is little to gain by futher changes downstream of that.  Also if I run with the full dataset and 6 cores, I run out of CPU memory at the first validation step.  I later removed TurboJPEG as it introduced extra dependencies that made transfering to other machines more complicated.

### Speed Benchmarking results 
Divided the dataset (~88k images) by 20, 3 epochs, note that the number of cores is actually one more than the num_workers parameter.  That refers to the number of child processes spawned: 
- 615 seconds, 5 cores, data on external SSD,
- 610 seoncs, 2 cores, data on external nVMe SSD, batch size 64, 
- 560 seconds 3 cores, on internal SSD, size 64
- 610 seconds 5 cores, internal SSD, size 64
- 500 seconds, 3 cores, internal SSD, Batch size 32
- 478 seconds, 2 core, internal SSD, Batch Size 32.
- 461 seconds, 2 core, internal SSD, batch size 32, no image augmentation
Up to this point, I actually ran 6 epochs, divided the number by 2.   Now running 3, and switching to Python 3.11
- 590 seconds, 5 cores, internal ssd, batch 64, python 3.11, opencv
- 586 seconds, as above but using PyTurboJPEG   
- 494 seconds, 3 cores, internal ssd, batch 64, python 3.11, opencv
- 467 seconds, as above  2 core, 
- 540 seconds, as above 1 core (num_workers = 0)

So for now, the training step appears optimised with just two processors running the data-loading but this could change with faster data-reading from storage.

I haven't re-visited this issue, but have noted that the process is running at more or less the same speed on two different machines, with the common factor bein both have NVIDIA 3060 GPU's.  So the bottleneck is probably the GPU.

## Perfomance Investigation

These are all the interesting things I would like to optimise or investigate:

1. **Limit to camera-class number** | Mice and some other species have really high occurance at some cameras, leading to the original dataset having > 2M images, mostly mice.  Since these images are highly correlated, and create excessive dataset imbalance, I have arbitrarliy set a limit of 500 per camera/class combination.  This should be tested.
Proposed experiments: limit max camer/class to [2000, 1000, 500, 250, 125]

2. **Buffer size** | How much space should be left around boundign boxes for cases where the bounding box is greater than the proposed crop size.  A larger buffer means more lossy downsizing, but more context and less chance of missing the animal.

3. **Pretrain with 21k weights** | Check if we get a perfomance boost as expected from literature using the 21k training weights instead

4. **Smaller networks** | Check if there is any gain or loss from using the M or S networks from the same EfficientNetV2 family.  

5. **Mixup $\alpha$** | Adjust the $\alpha$ as per literature survey

6. **Focal Loss $\gamma$** | Adjust the $\gamma$ as per literature survey

7. **Image augmentation** | How much is too much, try increasing magnitudes and probabilities of various augmentation methods

8. **Look again** | Can we gain some perfomance imrovement by using a moving window instead of object detection for inference

9. **Empty class** | Can we gain some perfomance imrovement by training with an empty class, created from data we know has no animals.

10. **Validation dataset** | See how well the validation performance lines up with the independent test set performance.  Consider using weights from a pre-set number of earlier epochs on stopping the training.

11. **Smarter dataset sub-sampling** | Look at using timestamps, or the MD prediction values to sample the larger datasets more cleverly.  To be specific, use the values of time, probability, and box location to determine how different the images are from each other.

12. **Compare with a simple square centre-crop** | Adjust the image loading to ignore the MegaDetector boxes.  Find out how much performance was gained from this approach, compared to simply cropping to square, then downsizing each image.

13. **Backbone fine tuning** | See if it helps by enabling backbone fine tuning (will need to 
run with a smaller batch size and/or a smaller network)


## Results
### Class-Location limits
The original dataset had over 2 million images, most of which were Mice, 800,000 mouse images coming from the Hawden Dataset alone.

All the experiments below were performed with the same training settings, early in the optimisation process.  So the training routine is not highly optimised, but the relative performace should still be relevant.

| **Exp** | **Run_ID** | **Limit** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat mAP** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_06 | Run_01 | 4000 | 0.838 | 0.661 |0.604| 0.652 | 0.364 | 0.808 | 0.620 |0.699 | 
|Exp_05 | Run_01 | 2000 |0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| Exp_03 | Run_13 | 1000 |0.790 |0.676 | 0.586 |0.689  |0.374 |0.809 |0.523 |0.745 |
| Exp_01 | Run_01 | 500 |0.821 |0.674 | 0.566 |  0.653|	0.352|0.828|0.662 |0.757 |
| Exp_04 | Run_02 | 250 |0.78| 0.693 | 0.582 | 0.732 | 0.301 | 0.804 | 0.506 | 0.653 |
| Exp_02 | Run_01 | 125| 0.868 |0.402 | 0.363 | 0.414 | 0.313 | 0.69 | 0.551 | 0.496 |

The most common classes were Mouse and Possum, which peaked for the 2000 per class/location.  Rat and stoat performance peaked for the 500 per class/location limit.  Rats are often mistaken for mice, so their performance can be expected to suffer from excessive mouse class size.

Each location has different numbers of cameras, with the largest having 90 cameras.  With hind sight it would have been better if the camera ID had been recorded, then we could have set a more relavent class-camera limit.  

The main reason for limiting the number from a given location is that the images are highly correlated, all having the same backgrounds.  Too many images from a particular class/location risks the algorithm learning based on irrelevant features. 

### Learning Rate

Initial learning rates of $10^{-2}$, $10^{-3}$, $10^{-4}$ were tried, with weight decay $10^{-5}$ of using an Adam optimiser with the `CosineAnnealingWarmRestarts()` class from `torch.optim.lr_scheduler`.  Best results were found with $10^{-3}$.

### Focal Loss
A custom implementation of focal loss was implemented, that used both the $\gamma$ and $\alpha$ terms to combat the effect of class imbalance and encourage firm predictions.  However the $\alpha$ term did not seem to help with performance.  It was left as a changable prameter, but set to `FOCAL_ALPHA = False`