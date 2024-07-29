# Experiment Tracking
## Explanation
For a general description of rational for these experiments, refer to [project_notes.md](project_notes.md). This document is a summary of the experimental details in chronological order.  The *Experiment* here, refers to a completely new training dataset.  The *Run_ID* refers to a retraining on the same dataset.  The project is set up so that the relavent new folders created, and processes run get created each time the settings file changes either of these.  

With every new experiment run through the [Train_Evaluate_Log.py](../Train_Evaluate_Log.py) script a file is ammended to the results folder with analysis of the performance metrics, and the original settings file is moved to the inputs folder for that experiment.  The *Independent Dataset* is kept unchanged throughout, and should not overlap with any of the camera locations used for training.  It has been balanced by randomly discarding some images to keep the main classes of interest are reasonably similar in size.

## Experiments

At this stage, without much optimisation it is clear from looking at confusion matrices that class imbalance is causing problems.  There are a lot of false positives in the mouse category, which is dragging everything else down.  Focus initially should be on addressing class imbalance.
### 0. Exp_03 | Run_02
- 4 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-3}$
- Used $\alpha$ term for focal loss, but possibly not correctly implemented at this point (I later normalised the weights by dividing by the mean) 

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_03 | Run_02 | 0.492 |0.365|0.371|0.264|0.182|0.458|0.351|0.483

This didn't train very well.  There were some confusing factors here.  The lack of normalising the alpha weights lead to a very different loss (3 orders of magnitude higher) so potentially poorly matched to the learning rate.  This experiment should be repeated for completion.

### 1. Exp_02 | Run_01
- 4 October 23
- Limit 125 for the class-location (10,034 image files), 
- Learning rate $10^{-3}$

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_02 | Run_01 | 0.868 |0.402 | 0.363 | 0.414 | 0.313 | 0.69 | 0.551 | 0.496

Scored very well in all metrics on the random test set, but poorly on the independent set.  The discarding of large datasets was so high here that the class imbalance was only a factor of 2 between mouse and the next most common class (rats)

### 2. Exp_01 | Run_01
- 4 October 23
- Limit 500 for the class-location, (107,060 image files)
- Learning rate $10^{-3}$

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_01 | Run_01 | 0.821 |0.674 | 0.566 |  0.653|	0.352|	0.828|	0.662 |0.757|




### 3. Exp_05 | Run_01
- 4 October 23
- Limit 2000 for the class-location, 220,596 images
- Learning rate $10^{-3}$

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_05 | Run_01 | 0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 


This is an interesting one.  Comparing to others with same parameters but different class sizes (1000, 500, 125) below.  The mouse performance is actually improving with size of the dataset.  

So Mouse and Possum improved with more samples (and imbalance), Rat & Stoat got worse.  None of the examples in this table had weighted sampling, so potentially the Rat's and stoats were not shown enough by the time the model had overtrained on Mice and Possums.

| **Exp** | **Run_ID** | **Sample Lim** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|**Exp_05** | **Run_01** | 2000 |0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| Exp_01 | Run_01 | 500 |0.821 |0.674 | 0.566 |  0.653|	0.352|	0.828|	0.662 |0.757|
| Exp_02 | Run_01 | 125| 0.868 |0.402 | 0.363 | 0.414 | 0.313 | 0.69 | 0.551 | 0.496

### 4. Exp_01 | Run_05
- 4 October 23
- Limit 500 for the class-location, (107,060 image files)
- Learning rate $10^{-3}$
- No weighted sampling or $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_01 | Run_05 | 0.869 | 0.676|	0.566|	0.578|	0.295|	0.812|	0.566|	0.738


### 5. Exp_03 | Run_08
- 4 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-4}$
- Used both $1/\sqrt{N}$ weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_03 | Run_08 | 0.866 | 0.661 | 0.531 | 0.617 |	0.280 |	0.775 |	0.526 |	0.70

The t appears that weighted sampling has not helped (if it is assumed the change of learning rate did not change the result much)

### 6. Exp_03 | Run_09
- 4 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-4}$
- Used both $1/\sqrt{N}$ weighted sampling and the $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_03** | **Run_09** | 0.851 |0.631 | 0.537 | 0.580 | 0.245 | 0.775 | 0.500 | 0.651
| Exp_03 | Run_08 | 0.866 | 0.661 | 0.531 | 0.617 |	0.280 |	0.775 |	0.526 |	0.70

Comparing to Exp_03, Run_08 it would appear that the $\alpha$ term has not helped, both with overall performance and the mouse figure.

### 7. Exp_03 | Run_11
- 12 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-4}$
- No weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_03** | **Run_11** | 0.836 | 0.650 | 0.581 | 0.636 | 0.310 | 0.814 | 0.521 |  0.732 	

This perfomance was a bit out of line with the others without weighted sampling or alpha term.  It was probably a mistake to change the learning rate.  I'm going to re-run with 10-3 just so I can make consistent decisions about the dataset size and trend.  Investigate these lesser details later.

### 8. Exp_03 | Run_12
- 12 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-4}$
- No weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_03 | Run_11 | 0.836 | 0.650 | 0.581 | 0.636 | 0.310 | 0.814 | 0.521 |  0.732 |
| **Exp_03** | **Run_12** | 0.836 | 0.648 |	0.582 |	0.643 |	0.282 |	0.801 |	0.489 |	0.729 	|


This was an accident, I intended to change the learning rate, but somehow didn't.  It's interesting that the previous result wasn't quite repeated, close but not identical, so maybe there is a random variable that isn't getting seeded properly.

### 9. Exp_03 | Run_13
- 12 October 23
- Limit 1000 for the class-location, (138,985 image files)
- Learning rate $10^{-3}$
- No weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_03 | Run_13 |0.790 |0.676 | 0.586 |0.689  |0.374 |0.809 |0.523 |0.745 |


### 9. Exp_06 | Run_01
- 12 October 23
- Limit 4000 for the class-location, (~324,000 image files)
- Learning rate $10^{-3}$
- No weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_04** | **Run_02** | 0.838 | 0.661 |0.604| 0.652 | 0.364 | 0.808 | 0.620 |0.699 |


### 10. Exp_04 | Run_02
- 12 October 23
- Limit 250 for the class-location, (49,992 image files)
- Learning rate $10^{-3}$
- No weighted sampling, no $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_04** | **Run_02** |0.78| 0.693 | 0.582 | 0.732 | 0.301 | 0.804 | 0.506 | 0.653 |


Interesting questions hopefully answered at this point:
- Do mouse and possum keep improving with more data
- Do rat and stoat keep getting worse
- Is there an obvious optimal dataset size
- Was $10^{-4}$ clearly worse than $10^{-3}$  (may need more runs to determine this)?


All runs summarised here used a learning rate of $10^{-3}$, no weighted sampling, no balancing term $\alpha$ for focal loss
| **Exp** | **Run_ID** | **Sample Lim** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_06 | Run_01 | 4000 | 0.838 | 0.661 | 0.604 | 0.652 | 0.364 | 0.808 | 0.620 |0.699 | 
| Exp_05 | Run_01 | 2000 | 0.834 | 0.666 | 0.591 | 0.656 | 0.42  | 0.841 | 0.624 | 0.723 | 
| Exp_03 | Run_13 | 1000 | 0.790 | 0.676 | 0.586 | 0.689 | 0.374 | 0.809 |0.523 |0.745 |
| Exp_01 | Run_01 | 500  | 0.821 | 0.674 | 0.566 | 0.653 | 0.352 |0.828 |0.662 |0.757 |
| Exp_04 | Run_02 | 250  | 0.78  | 0.693 | 0.582 | 0.732 | 0.301 | 0.804 | 0.506 | 0.653 |
| Exp_02 | Run_01 | 125  | 0.868 | 0.402 | 0.363 | 0.414 | 0.313 | 0.69 | 0.551 | 0.496 |

**Next questions**
Looking at the largest (and most imbalanced dataset)
- Would this situation change by applying weighted sampling.
- How about a balancing term to the loss function.


### 11. Exp_05 | Run_03
- 16 October 23
- Limit 2000 for the class-location, (220,596 images)
- Learning rate $10^{-3}$
- $\alpha$ term for focal loss
- No weighted sampling, 


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_04** | **Run_02** |0.143 | 0.164 |	0.141 |	0.095 |	0.167 |	0.463 |	0.265 |	0.089|

Terminated at the end of epoch 1 with no improving scores.  Something very wrong with this setting!

### 12. Exp_05 | Run_04
- 16 October 23
- Limit 2000 for the class-location, (220,596 images)
- Learning rate $10^{-3}$
- Weighted Sampling

| **Exp** | **Run_ID** | **Weighted** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_01 |No |0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| **Exp_05** | **Run_04** |1/$\sqrt{n}$ |0.874|0.614|0.539|0.668|0.295|0.773|0.562|0.729|

Weighted sampling doesn't seem to have helped here compared to the same settings without it.

### 12. Exp_05 | Run_05
- 16 October 23
- Limit 2000 for the class-location, (220,596 images)
- Learning rate $10^{-3}$
- Weighted Sampling
- $\alpha$ term for focal loss

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_01 |0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| **Exp_05** | **Run_05** |0.837|0.662|0.609|0.628|0.321|0.795|0.588|0.707|

This experiment seems pretty decisive.  The lower learning rate hurts performance.  At this point it might be worth trying $10^{-2}$

### 12. Exp_05 | Run_06
- 17 October 23
One more crack at using focal $\alpha$, this time with weighted sampling and Learning rate of $10^{-3}$

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_01 |0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| **Exp_05** | **Run_06** |0.855|0.604|0.512|0.580|0.271|0.747|0.491|0.671|

Consistent with previous experiments, we are still getting no benefit from focal $\alpha$, this time with weighted sampling.

### 13. Exp_05 | Run_07
- 16 October 23
- Limit 2000 for the class-location, (220,596 images)
- **Learning rate $10^{-2}$**

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_07 |0.090 |  0.116 | 0.101 | 0.083 | 0.127 | 0.297 | 0.162 | 0.072 | 

- The model jumped stright for a solution that called everything a black billed gull, and got stuck there for 12 epochs, then started shooting up in precision, but at that point it happened to go for 3 non-minimum loss values and self terminate.

- It would be worth re-trying this with a warmup rate of $10^{-3}$.

### 14. Exp_08 | Run_15
(Mislabeled, should have been Exp_05, Run_08)
- 17 October 23
- Limit 2000 for the class-location, (220,596 images)
- Learning rate $10^{-3}$
- Focal $\gamma$ = 3
- Weighted Sampling
- Focal $\alpha$ = True

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_08 | 0.816 | 0.557 | 0.499 |0.506 |.205 | 0.739 | 0.413 |0.589 | 

- The intention was just to change focal $\gamma$ (from 2 to 3).
- Epic fail on all metrics, but not clear the reason since I changed too many things at once.


### 15. Exp_05 | Run_09
- 17 October 23
- Limit 2000 for the class-location, (220,596 images)
- Mixup $\alpha$ = 0.3

| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_09 |0.861 | 0.684 | 0.625 | 0.708 | 0.413 | 0.816 | 0.613 | 0.725 | 

### 15. Exp_05 | Run_10
- 17 October 23
- Limit 2000 for the class-location, (220,596 images)
- `MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'`
- `USE_MIXUP = False`

| **Exp** | **Run_ID** | **Cutmix** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|**Exp_05** | **Run_10** | No | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |
|Exp_05 | Run_09 | $\alpha$=0.3 |0.861 | 0.684 | 0.625 | 0.708 | 0.413 | 0.816 | 0.613 | 0.725 | 
| Exp_05 | Run_01 |  $\alpha$=0.5 | 0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 

Either Mixup doesn't help with this scenario, or it has been implemented incorrectly.  For the next round of experiments just leave it out.  Maybe re-visit later if there is time.  I wonder if mixing up color images with greyscale causes trouble, or the added confusion is just too much to pick out the animals at all.

### 15. Exp_05 | Run_11
- 18 October 23
- Limit 2000 for the class-location, (220,596 images)
- Focal $\gamma$ = 4
- Mixup $\alpha$ = 0.5

| **Exp** | **Run_ID** | **$\gamma$** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_11 | 4 | 0.854 | 0.673 | 0.582 | 0.637 | 0.298 | 0.794 | 0.596 | 0.702 | 
| Exp_05 | Run_01 | 2| 0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 


### 15. Exp_05 | Run_12
- 18 October 23
- Limit 2000 for the class-location, (220,596 images)
- Focal $\gamma$ = 3
- Mixup $\alpha$ = 0.5

| **Exp** | **Run_ID** | **$\gamma$** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_12 | 3 | 0.864 | 0.644 | 0.591  | 0.643 | 0.347| 0.806 |0.547 | 0.670  |

The performance doesn't seem to be super sensitive to $\gamma$ but it is helping with the mouse score (which is mostly effected by false positives) Actually, so far this is doing what it is expected to do.  Improving recall on the rare classes, at the expense of the common classes.  Hence the overall improvement, but with some loss of performance on the key species we're tracking.

### 15. Exp_05 | Run_13
- 19 October 23
- Limit 2000 for the class-location, (220,596 images)
- Focal $\gamma$ = 1
- Mixup $\alpha$ = 0.5

| **Exp** | **Run_ID** | **$\gamma$** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_11 | 4 | 0.854 | 0.673 | 0.582 | 0.637 | 0.298 | 0.794 | 0.596 | 0.702 | 
|Exp_05 | Run_12 | 3 | 0.864 | 0.644 | 0.591  | 0.643 | 0.347| 0.806 |0.547 | 0.670  |
|Exp_05 | Run_01 | 2| 0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 | 
| **Exp_05** | **Run_13** | 1 | 0.877 | 0.663 | 0.610 | 0.683 | 0.348 | 0.809 | 0.543 | 0.669 |

At this point it seems like the performance might not be very sensitive to $\gamma$, but the excellent performance of Exp_05 on the independent set, but slightly below trend performance on the test set, hints that many of these results are suffering from over-training.

### 15. Exp_05 | Run_13a
(was running 13 on two different machines by mistake)
- 19 October 23
- Limit 2000 for the class-location, (220,596 images)
- Changed to a timm model *efficientnet_v2_l_in21ft1k*

| **Exp** | **Run_ID** | **Model** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_05 | Run_01 | 1k | 0.834 |  0.666 | 0.591 | 0.656 | 0.42 | 0.841 | 0.624 | 0.723 |   
| **Exp_05** | **Run_13a** | 2k/1k | 0.856 | 0.617 | 0.537| 0.594|0.326 |	0.778 |	0.512 |	0.645 	|

Since this experiment showed a pretty classic improvement in test performance, but drop in performance on the independent set.  I should really run a model where I produce several checkpoints, and evaluate them all to look at the effects of overtraining.

#### Efficientnet V2 Pretrained models to chose from

```python
import timm
list_of_models = timm.list_models(pretrained=True)
print(list_of_models)
```
'tf_efficientnetv2_b0.in1k', 'tf_efficientnetv2_b1.in1k', 'tf_efficientnetv2_b2.in1k', 'tf_efficientnetv2_b3.in1k', 'tf_efficientnetv2_b3.in21k', 'tf_efficientnetv2_b3.in21k_ft_in1k', 'tf_efficientnetv2_l.in1k', 'tf_efficientnetv2_l.in21k', 'tf_efficientnetv2_l.in21k_ft_in1k', 'tf_efficientnetv2_m.in1k', 'tf_efficientnetv2_m.in21k', 'tf_efficientnetv2_m.in21k_ft_in1k', 'tf_efficientnetv2_s.in1k', 'tf_efficientnetv2_s.in21k', 'tf_efficientnetv2_s.in21k_ft_in1k', 'tf_efficientnetv2_xl.in21k', 'tf_efficientnetv2_xl.in21k_ft_in1k'

#### New Priorities:
Fix up the two errors in the class names.   fernberd => fernbird, goldfinnch => goldfinch, & consider removing brown creeper since it's influencing the results too much.
**Stick to:** `FOCAL_GAMMA=1`, `FOCAL_ALPHA=False`, `USE_CUTMIX=False`, `IMAGE_LIMIT=2000`, `LEARNING_RATE = 1e-3`.
**Test**: Background Removal, Alternative EffnetV2 pretrained models.

Store a few checkpoints, and investigate optimal checkpoint selection.
Then review again. Consider updating the dataset with new imagery.


### 15. Exp_08 | Run_03 & Run_04
**This result was invalid, max_epochs was set to 3**
- 25 October 23
- Limit 2000 for the class-location, (220,596 images)
- Timm model *efficientnet_v2_l_in21ft1k*
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- Empty_backgrounds, with faded edges, Margin:0.4, Fade:0.2-0.05


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_08** | **Run_03** | 0.756 |0.529 |	0.410 |	0.528| 	0.168 |	0.721 |	0.518 |	0.636|
| **Exp_08** | **Run_04** | 0.823 |0.612 |	0.461 |	0.686 | 0.168 | 0.794 | 0.595 | 0.702 |
|Exp_05 | Run_10 | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

Run 04 was still a little better than run_03, but still well down on the baseline.


### 15. Exp_09 | Run_01 | 25 October 23
First attempt was invalid as the number of epochs was set to 3.  Run_02 was good.
- Limit 2000 for the class-location, (220,596 images)
- Timm model *efficientnet_v2_l_in21ft1k*
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- Empty_backgrounds, with faded edges, Margin:0.4, Fade:0.4-0.05


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_09 | Run_01 | 0.710 | 0.486 | 0.345 | 0.514 | 0.187 | 0.721 | 0.369 | 0.608 |
|**Exp_09** | **Run_02** | 0.813 | 0.554 | 0.471 | 0.661 | 0.141 | 0.725 | 0.460 | 0.630 |
|Exp_05 | Run_10 | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

Run 02 was still a little better than run_01, but still well down on the benchmark with the baseline.


### 15. Exp_10 | Run_03 | 25 October 23
Repeat of run 02, which accidentally set the max epochs to 3
- Limit 2000 for the class-location, (220,596 images)
- Timm model *efficientnet_v2_l_in21ft1k*
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- Empty background fading 0.05-0.2, margin=0.4, buffer = 0.1


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_10 | Run_02 | 0.783 | 0.61 | 0.436 | 0.613 | 0.231 | 0.816 | 0.556 | 0.697 |
| **Exp_10** | **Run_03** | 0.875 | 0.682 | 0.529 | 0.734 | 0.160 | 0.837 | 0.685 | 0.744 |
|Exp_05 | Run_10 | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

This was the best of the no-background trials, but still there does not appear to be any improvement over the current baseline.

### 15. Exp_11 | Run_01 & Run_03
- 25 October 23
- Limit 2000 for the class-location, (220,596 images)
- Timm model *efficientnet_v2_l_in21ft1k*
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- No Background remove or fading, 
- `buffer = 0.1`

This experiment was  a bit of a stuff-up.  I had intended to test the buffer parameter with `buffer = 0.3`, but later noticed it was left at 0.1.  So effectively this is just a similar but not identical dataset to Exp_05.  The improvement in Run_03 was likely due to a bug-fix in the inference step, rather than a change of hyperparameters.
| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_11 | Run_01 | 0.892 |0.699 |	0.594 |	0.708| 	0.462 |	0.847 |	0.603 |	0.75|
| **Exp_11** | **Run_03** | **0.944** | **0.811** |	0.717 | 0.848 | 0.586 | 0.886 | 0.747 | 0.792 |
|Exp_05 | Run_10 | 0.92  | 0.774 | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |


Run_01 was invalid  `MAX_EPOCHS` was accidentally set to 3.  Also I discovered a bug in the treatment of the images in the inference script.

Run_01 was the new best, with big jumps in performance across all the classes of interest, and overall. Unforunately it was run a little out of order because by the time I got this experiment repeated. I had re-run a lot of others with `buffer = 0.1`  buffer `buffer = 0.3` should be the new baseline, as soon as the architecture experiments are completed.

### 15. Exp_12 | Run_01
- 1 November 23
- Limit 2000 for the class-location, (220,596 images)
- Timm model *efficientnet_v2_l_in21ft1k*
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- `BUFFER=1` (Effectively no MegaDetector)


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_12** | **Run_01** | 0.894 | 0.721 | 0.621 | 0.799 | 0.296 | 0.866 | 0.590 | 0.694 |
|Exp_05 | Run_10 | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

The overall macro Average Precision score on the independent dataset was down 5% from the baseline example which used a buffer size of 0.1.  It would be interesting to re-visit this experiment after the training is further optimised to see if this difference remains.


## Experiments with Network Architecture.
Keep the above settings, use dataset from experiment 5.

- Limit 2000 for the class-location, (220,596 images)
- `MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'`
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- No Background removal
- Buffer = 0.1

1. Try a smaller custom classifier head
2. Try 3 different sizes: S, M, L  
*tf_efficientnetv2_s.in21k*, *tf_efficientnetv2_m.in21k*, *tf_efficientnetv2_l.in21k*


### 15. Exp_05 | Run_17 & Run_20

Using a simplified classifier head. Run 17 was repeated because I made minor changes to the inference step because I found a problem with the way the buffer was being handled.
-  6 November 23 
- `HEAD_NAME = 'BasicHead'`
- `MODEL_NAME = 'tf_efficientnetv2_l.in21k_ft_in1k'`
- `FOCAL_GAMMA=1`
- `LEARNING_RATE = 1e-3`
- `USE_CUTMIX=False`
- `BUFFER=0.1`

| **Exp** | **Run_ID** | **Head** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_05 | Run_17 | Basic |0.913 | 0.703 | 0.591 | 0.761 | 0.352 | 0.846 | 0.544 | 0.791 |
| **Exp_05** | **Run_20** | Custom |0.913 | 0.740 | 0.629 | 0.770 | 0.441 | 0.854 | 0.684 | 0.721 |
|Exp_05 | Run_10 | Custom | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

These results are consistent.  The simpler head hurt the performance a little, and the performance improved when I de-bugged the image pre-processing.  

I could re-visit the question of the simpler head again later, note that this experiment did use a slightly different network.  And backbone fine tuning is likely to have the same effect as the extra head layers. But anyway for now stick with the larger head to keep the experiments comparible.

### 15. Exp_05 | Run_18 & Run_19

Going back to the larger classifier head, but testing the *EffNetV2s*, and *EffNetV2m*.

-  7 November 23 
- `HEAD_NAME = 'ClassifierHead'`
- `MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'` (Run_19), 
- `MODEL_NAME = 'tf_efficientnetv2_m.in21k_ft_in1k'` (Run_18)
- `BUFFER=0.1`

| **Exp** | **Run_ID** | **Model** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Exp_05 | Run_10 | *EffNetV2l*| **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |
|Exp_05 | Run_19 | *EffNetV2m*|0.927 | 0.757 | 0.638 | 0.801 | 0.533 | 0.783 | 0.687 | 0.693 |
|Exp_05 | Run_18 | *EffNetV2s*|0.916 | 0.732 | 0.621 | 0.782 | 0.461 | 0.775 | 0.674 | 0.651 |

The difference between the networks was small, but overall the larger network did perform best.  It's worth noting the substantial speed difference though.  20, 33 & 48 images per second for L, M & S respectively.  This topic could be re-visited if speed or memory requirements become more important.

## Experiments pretrain weights & backbone fine tuning
- With *EffNetV2l*, try different network pretrain weight combinations:  *in1k*, *in21k*, *in21k_ft_in1k*
- With the best from above, try releasing some additional backbone layers.


### 15. Exp_05 | Run_21, Run_22 & Run_23

-  7 November 23 
- `HEAD_NAME = 'ClassifierHead'`
- `MODEL_NAME =` *in1k*, *in21k*, *in21k_ft_in1k*
- `BUFFER=0.1`

| **Exp** | **Run_ID** | **Weights** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_11 | Run_03 | *in21ft1k* |0.944 | 0.811 |	0.717 | 0.848 | 0.586 | 0.886 | 0.747 | 0.792 |
| Exp_05 | Run_21 | *in21k* | 0.926 | 0.759 | 0.652 | 0.827 | 0.508 | 0.820 | 0.644 | 0.704 
| Exp_05 | Run_22 | *in1k* | 0.926 | 0.756 | 0.648 | 0.817 | 0.538 | 0.798 | 0.671 | 0.695 | 
| Exp_05 | Run_23 | *in21ft1k* | 0.933 | 0.808 | 0.702 | 0.856 | 0.556 | 0.881 | 0.755 | 0.763 |

Experiment 11 & Experiment 05 had some random variation, but this is still a clear win for *in21k_ft_in1k*.

### 16. Exp_18 | Run_02 & Run_03

-  14 November 23 
- Run_03 repeat of Exp_18 Run_02, but with backbone fine-tuning of the next 4 layers, starting after 6 epochs regularisation layers were not unfrozen.  For reasons unknown VSCode Crashed, before the training was complete.
- Run_03 repeat, unfreezing 2 layers, after 10 epochs.

| **Exp** | **Run_ID** | **BB-FT** | **Lyrs** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_18 | Run_02 | No | N/A |0.917 | **0.828** | 0.728 | 0.884 | 0.650 | 0.884 | 0.794 | 0.877 |
| Exp_18 | Run_03 | Ep-6 | 4 | Crashed 
| Exp_18 | Run_04 | Ep-10 | 2 | 0.920 | **0.828** | 0.744 | 0.851 | 0.648 | 0.879 | 0.795 | 0.866|



| **Exp** | **Run_ID** | **BB-FT** | **Lyrs** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_19 | Run_01 | No | N/A | 0.936 | 0.818 | 0.727 | 0.855 | 0.624 | 0.866 | 0.761 | 0.805
| Exp_19 | Run_02 | EP-10 | 2-lyr | 0.937 | **0.821** |	0.730 |	0.834 | 0.667 |	0.874 |	0.766 |	0.819 	

Experiments 18 & 19 suggests a small but useful boost from fine-tuning

### 16. Exp_20 | Run_01

-  16 November 23 
- Backbone fine-tuning, 4 layers, but  KNN sampling 250-350 per camera-class

| **Exp** | **Run_ID** | **BB-FT** | **Lyrs** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_20 | Run_01 | Yes | 4 | 0.919 | 0.811 | 0.713 | 0.859 | 0.554 | 0.881 | 0.788 | 0.877 |

This appears to be a step in the wrong direction, but for now it's hard to know if it's the larger dataset or the fine-tuning.



## Experiments with the buffer size
Keep the above fixed, and retrain with `BUFFER= 0.0`, `BUFFER= 0.2`, `BUFFER= 0.3` & `BUFFER= 0.4`



| **Exp** | **Run_ID** | **Buffer** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_12 | Run_01 | 1 |0.894 | 0.721 | 0.621 | 0.799 | 0.296 | 0.866 | 0.590 | 0.694 |
| Exp_14 | Run_01 | 0.3 | 0.934 | 0.726 | 0.635 | 0.806 | 0.337 | 0.837 | 0.638 | 0.772 |
| Exp_15 | Run_01 | 0.2 | 0.942 | 0.742 | 0.637 | 0.805 | 0.419 | 0.832 | 0.630 | 0.743 |
| Exp_15 | Run_23 | 0.1 | 0.933 | 0.808 | 0.702 | 0.856 | 0.556 | 0.881 | 0.755 | 0.763 |
| Exp_16 | Run_01 | 0 | 0.928 | **0.815** | 0.725 | 0.840 | 0.548 | 0.881 | 0.767 | 0.846 | 

Also the following, from exp12 with buffer=1


| **Exp** | **Run_ID** | **Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_12** | **Run_01** | 0.894 | 0.721 | 0.621 | 0.799 | 0.296 | 0.866 | 0.590 | 0.694 |
|Exp_05 | Run_10 | **0.92** | **0.774** | 0.647 | 0.799 | 0.491 | 0.859 | 0.625 | 0.784 |

A nice clear result.  The buffer was not helping at all, and should be set to 0.  This raises the question whether actually rescaling should also be turned off entirely.  The penalty would be some larger animals getting partial crops, but the potential gain is we loose no texture information.

## Experiments with dataset sub-sampling methods
Try to boost perfomance by smarter sub-sampling methods from the dataset.
The first of these used `BUFFER=0.1`, the remainder `BUFFER=0.0`

When a range is given, the lower limit simply takes everything, the upper limit filters from the total number of samples to take only that number.  If the samples from that grouping falls in between those two values, the filtering is progressively more selective until the upper limit is reached.

| **Exp** | **Run_ID** | **Limit** | **Method** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Exp_17 | Run_01 | 50-250 | kMeans | 0.887 | 0.778 | 0.713 | 0.839 | 0.632 | 0.867 | 0.734 | 0.845 |
| Exp_21 | Run_01 | 150 | kMeans | 0.907 | 0.829 | 0.733  | 0.866 | 0.599 | 0.896 | 0.780 | 0.871 |
| Exp_18 | Run_02 | 180 | kMeans | 0.911 | 0.820 | 0.731 | 0.852 | 0.628 | 0.872 | 0.786 | 0.887 |
| Exp_18 | Run_02 | 200 | kMeans |0.917 | 0.828 | 0.728 | 0.884 | 0.650 | 0.884 | 0.794 | 0.877 |
| Exp_24 | Run_02 | 200 | Random |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_18 | Run_04 | (above) | fine-tuned | 0.920 | 0.828 | 0.744 | 0.851 | 0.648 | 0.879 | 0.795 | 0.866|
| Exp_?? | Run_01 | 220 | kMeans | 0.911 | 0.820 | 0.731 | 0.852 | 0.628 | 0.872 | 0.786 | 0.887 |
| Exp_19 | Run_01 | 250 | kMeans | 0.936 | 0.818 | 0.727 | 0.855 | 0.624 | 0.866 | 0.761 | 0.805 |
| Exp_13 | Run_01 | 250 | Random | 0.934 | 0.826 | 0.727 | 0.839 | 0.649 | 0.872 | 0.762 | 0.787 |
| Exp_20 | Run_01 | 250-350 | kMeans | 0.919 | 0.811 | 0.713 | 0.859 | 0.554 | 0.881 | 0.788 | 0.877 |

**Exp_17** was interesting.  There were no rats getting called mouse, but Kereru was getting called Blackbird.  The total training dataset was just 77,000.  It's quite a good score considering the dataset was so much smaller than others.  However it doesn't really answer the question of whether k-means was helping, as it was a drop compared to experiment 13.  For this reason I'm starting Exp_19, to be more comparable.

**Notes:** Experiments 13 and 19 had `BUFFER = 0.1`, also there's a mistake in the EXP_19 settings file, max and min got mixed, so it's just a limit of 250, not 250-300 as intended.

Experiments 18 and 21 are a bit tricky to compare, as 21 included fine-tuning.   Given the very marginal improvement overall, and drop on the more important classes, maybe the optimum value for limit might be near or over 200.

It would be worth repeating at 200 with random instead of k-means, since EXP21 stands out as being quite high.

##Performance on Empties

I got a big jump in performance by ensembling both models, and taking the sum of the Confidence < Max Probability => empty   Results shown here are all with the MD threshold and the classifier threshold equal values.  The table below shows this new arrangement, with an additional 500 images that were known to be empty, with different values for the thresholds (both set to the same value)

Experiment_26, Run_01, requiring Confidence < MD_THRESHOLD > AND Probability < THRESHOLD to re-assign the score to 'empty'
| **Thresholds** | **TP** | **FP** | **TN** |**FN** | **Emp Prec** | **Emp Recall** | **Bal Acc** | **Precision** |  **Recall**  |
:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.6 |  556  | 338  |  3299  |  5   |  0.62  |  0.99  |       |      |      |
| 0.5 |  550  | 242  |  3395  |  11  |  0.69  |  0.98  |       |      |      |
| 0.4 |  543  | 166  |  3471  |  18  |  0.77  |  0.97  |  0.76 |      |      |
| 0.3 |  535  | 108  |  3529  |  26  |  0.83  |  0.95  |  0.76 | 0.77 | 0.81 |
| 0.2 |  527  | 70   |  3567  |  34  |  0.88  |  0.94  |  0.77 | 0.78 | 0.81 | 
| 0.15|  516  | 43   | 3594   |  45  |  0.92  |  0.92  |  0.78 |  0.78| 0.81 |  
| 0.1 |  500  | 29   | 3608   |  61  |        |        | 0.76  | 0.81  | 0.78|

For the whole of dataset metrics, reporting the mean over each class for all the classes with more than 40 samples:
- Precision  = tp / (tp + fp): For each class, the chance of getting this class right if it is predicted
- Recall = tp / (tp + fn):   The chance of getting this class right if it is true  
- Accuracy  = tp / Total:  The overall chance of a prediction in this class being correct  
The TP, FP, TN, FN, Empty Prec, Empty recall  colums refer to the score only for the empty class.


### Experiment 26

Run 01 was with the usual classes, but an updated (larger) dataset.
Run 02 was using a reduced number of classes, to see how that affected performance.  Classes were chosen based on a 'DOC use case': cat, rat, mouse, possum, ferret, stoat, weasel, livestock, deer, thar, chamois, pig, goat, walaby, hedgehog, rabit, hare, sealion, dog, human, bird, kea, kiwi.

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_26 | Run_01 | 200 | 2000 | 0.912 | 0.840 | 0.737 | 0.859|  0.620 | 0.884 | 0.770| 0.875 |
| Exp_26 | Run_02 | 200 | 2000 | 0.912 | 0.801 | 0.702 | 0.832 | 0.505 | 0.870 | 0.739 | 0.839 | 	

There was no improvement with the reduced classes.  It appears we might as well stick to the same class definition regardless of use case.

Interestingly the performance also dropped a little just with the increased dataset.  No other parameters were changed since Exp_24.


## Experiments with much larger dataset
Previous experiments accidentally had their 2000 image limit per dataset left in place, so when subsetting by camera location was implemented, the full effect was not tested.  The following experiments are re-testing the size limits, now that there is an additional option of adjusting both parameters.

### Experiment 27

This was the first attempt at bringing down the site-limit. It also included a big update to the dataset, introducing a wider varierty of image sources.

Experiment 27: Site-limit 8000, camera-limit 200, random.  



| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_27 | Run_01 | 200 | 20,000 | 0.898 | 0.784 | 0.684 | 0.805 | 0.545 | 0.857 | 0.737 | 0.860 |

This was a significant drop in performance compared to Exp_24.  But this is raising the question, is the hidden test set now too limited.  It contains images all taken in the same way, with similar resolution?

### Experiment 28

- This experiment used knn for image selection.  
- Run_01 used a larger network *EfficientNetV2XL*
- Run_01 used a larger network *EfficientNetV2L*

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_27 | Run_01 | 200 | 20,000 | 0.898 | 0.784 | 0.684 | 0.805 | 0.545 | 0.857 | 0.737 | 0.860 |
| Exp_28 | Run_01 | 150 | 10,000 | 0.888 | 0.796 | 0.694 | 0.868 | 0.493 | 0.851 | 0.787 | 0.860 |	
| Exp_28 | Run_01 | 150 | 10,000 | 0.910 | 0.802 | 0.697 | 0.802 | 0.554 | 0.868 | 0.738 | 0.862 |	

This seems to be a small win for the smaller network  *EfficientNetV2L* over *EfficientNetV2XL*.  At least for the chosen crop size.  The issue could be re-visited some time with larger crops.

The more resrictive image selection seems to have helped over Exp_27.

### Experiment 30

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_28 | Run_01 | 150 | 10,000 | 0.910 | 0.802 | 0.697 | 0.802 | 0.554 | 0.868 | 0.738 | 0.862 |
| Exp_30 | Run_01 | 200 | 5,000 | 0.905 | 0.808 | 0.711 |  0.828 | 0.546 | 0.875 | 0.778 | 0.863 | 	

Balanced accuracy with empties included: 0.62


### Experiment 31
Site-limit 8000, camera-limit 200, random

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_30 | Run_01 | 200 | 5,000 | 0.905 | 0.808 | 0.711 |  0.828 | 0.546 | 0.875 | 0.778 | 0.863 |
| Exp_31 | Run_01 | 200 | 8000 | 0.890 | 0.792 | 0.699 | 0.813|0.558|0.861|0.768|0.872|

Balanced accuracy with emties: 0.7  This result seems a little high, considering the above perforamnce metrics weren't especially impressive.  Unfortunately the whole experiment seems to have gone missing.

It would appear that the smaller dataset of Exp_30, was a little better.

### Experiment 32

Site-limit 3000, camera-limit 200, random

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_30 | Run_01 | 200 | 5,000 | 0.905 | 0.808 | 0.711 | 0.828 | 0.546 | 0.875 | 0.778 | 0.863 |
| Exp_32 | Run_01 | 200 | 3,000 | 0.905 | 0.809 | 0.716 | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |

Balanced Accuracy with empties on Exp_32 was 0.64


### Experiment 33, 34, 35
33: Site-limit 2000, camera-limit 200, random
34: Site-limit 4000, camera-limit 180, random
35: Site-limit 2000, camera-limit 180, random
36: Site-limit 3000, camera-limit 180, random
There was a typo in the settings description,  says limit 3000, it was actually 2000 (confirmed from the actual setting file)

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_32 | Run_01 | 200 | 3,000 | 0.905 | 0.809 | 0.716 | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |
| Exp_33 | Run_01 | 200 | 2,000 | 0.906 | 0.800 | 0.706 | 0.822 | 0.541 | 0.879 | 0.753 | 0.85  |
| Exp_34 | Run_01 | 180 | 4,000 | 0.901 | 0.801 | 0.700 | 0.826 | 0.580 | 0.871 | 0.796 | 0.88  |
| Exp_35 | Run_01 | 180 | 2,000 | 0.904 | 0.794 | 0.691 | 0.831 | 0.545 | 0.875 | 0.778 | 0.861 | 
| Exp_36 | Run_01 | 180 | 3,000 | 0.911 | 0.801 | 0.703 | 0.817 | 0.559 | 0.874 | 0.748 | 0.859 |

It looks like the optimum is aroud 3000 per site, but anything in the 2000-4000 range is fairly similar. It might be good to check 220/4000 for completion.  At this point the evidence points towards a few things:


- The limit on site numbers isn't so different from before.  Potentially this is driven by the images all having similar resolution, time of year, vegetation etc.  So to a degree it doesn't matter that they are from different locations within the site.
- That we are consistently measuring worse performance on the secret test set, after introducing a wider variety of images, may actually point to limitations of the test set.  Further progress on this may need a different test set, with a wider variety of image resolutions.
- The dataset size limit in the 2000 - 4000 range is somewhere near optimal.
- The models from experiments 30 and 32 may actually be more useful than those from experiment 24, but we can't show this from the above evaluation.

The only useful experiment left I can think of for now, without putting work into improving the evatuation metrics, would be to increase the scale augmentation, to make the model less sensitive to scale.  We got a hint that this might be an issue when running on video and finding that the results changed with changing image resolution.

## Augmentation Improvements
Goal: Improve the model's robustness to different image formats & scales

### Experiment 32

Site-limit 3000, camera-limit 200, random.  
The default scale limit setting is (1 plus/minus 0.1)
If a tuple is used eg (a, b) scales [1 + a, 1 + b]  eg (-.5, 3) => [0.5, 4]

| **Exp** | **Run_ID** | **scale_limit** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 0.1, p=0.5 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_32 | Run_01 |0.1, p=0.5 | 0.905 | 0.809 | 0.716 | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |
| Exp_32 | Run_02 | 1, p=0.5| 0.901 | 0.805 | 0.705 | 0.813 | 0.549 | 0.883| 0.791 | 0.883 |
| Exp_32 | Run_03 | (-.5, 3), p=0.5 | 0.87 | 0.764 | 0.706 | 0.766 | 0.469 | 0.857 | 0.754 | 0.866 |
| Exp_32 | Run_04 | (-.75, 2), p=0.5 | 0.888 | 0.774 | 0.686 | 0.802 | 0.476 | 0.866 | 0.762 | 0.854 |
| Exp_32 | Run_05 | (-.75, 1), p=0.8 | 0.882 | 0.779 | 0.681 | 0.779 | 0.496 | 0.848 | 0.720 | 0.720 | 0.864 |
| Exp_32 | Run_06 | (-.75, 0.2), p=0.8 | 0.899 | 0.795 | 0.695 | 0.822 | 0.549 | 0.89 | 0.759 | 0.868 |
| Exp_32 | Run_07 | (-.75, 0.2), p=0.8 | 0.894 | 0.795 | 0.695 | 0.833 | 0.504 | 0.869 | 0.692 | 0.873 |
| Exp_32 | Run_08 | (-.75, 0.1), p=0.8 | 0.891 | 0.790 | 0.683 | 0.827 | 0.524 | 0.875 | 0.729 | 0.843 |

For run_07 onwards I consolidated the three kinds of Plover (Shore Plover, Spurwing Plover and Plover, apparently it doesn't make a lot of difference to overall metrics though.  In fact weirdly it seemed to hurt the mouse and rat score! I presume that is just random)

Run 02 doesn't look very exciting, except that I have increased the augmentation without hurting the performance much on a fairly uniform test dataset.  So it might acutally be a better model, but it can't be proved yet.  Run 03 and 04 have much higher values on the scale up side, resulting in the animal getting missed entirely sometimes.

Run_05 is fairly close to the scales of run_02, but increased likelyhood of scaling, so as to prevent the model converging on the 50% that aren't scaled.

it's not clear from the above if the upper limit should be 0.1 or 0.2, but there isn't really a meaningful difference, I'll stick to 0.1 for now.  For more meaningful changes of scale I might need to think more about the whole question of how to create the crops from the bounding box.

### Experiment 37
Revisit the question if I could make a small improvement or at least train more quickly if I had a slightly smaller dataset and used KNN to get maximally different images.

Experiment 37 uses KNN, reduce camera limit to 180, keep site limit to 3000.  For whatever reason this just doesn't seem to help.  It was quite a big drop compared to the others.

| **Exp** | **Run_ID** | **Cam-Lim** | **Site-Lim** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 2000 |0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_32 | Run_01 | 200 | 3,000 | 0.905 | 0.809 | 0.716 | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |
| Exp_36 | Run_01 | 180 | 3,000 | 0.911 | 0.801 | 0.703 | 0.817 | 0.559 | 0.874 | 0.748 | 0.859 |
| Exp_36 | Run_02 | 180 | 3,000 | 0.903 | 0.795 | 0.700 | 0.806 | 0.503 | 0.891 | 0.770 | 0.850 |
| Exp_37 | Run_01 | 180 | 3,000 | 0.866 | 0.768 | 0.658 | 0.818 | 0.502 | 0.864 | 0.750 | 0.868 |

From this it still appears that experiment 24 is the best, but at this point I think it is a bit unfair.  Experiment 36 run_01 actually has a lot more data diverstity, and 36 run_02 also has a big increase in scale augmentation. Even 37_01 might be better than thought.   Potentially with a more diverse dataset it would be apparent that it is really the best.

Experiment 37 uses KNN, reduce camera limit to 180, keep site limit to 3000.  For whatever reason this just doesn't seem to help.  It was quite a big drop compared to the others.

All I could usefully do at this point is try to increase augmentation without further hurting model performance.  For the time being I will stick with experiment 36, mainly because it trains faster with less data.

### Experiment 36 - Revisiting Augmentation
Revisit the question if I could make a small improvement or at least train more quickly if I had a slightly smaller dataset and used KNN to get maximally different images.



| **Exp** | **Run_ID** | **Augmentation** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_32 | Run_01 | 200 |  0.905 | **0.809** | **0.716** | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |
| Exp_36 | Run_01 | 180 |  0.911 | 0.801 | 0.703 | 0.817 | 0.559 | 0.874 | 0.748 | 0.859 |
| Exp_36 | Run_02 | 180 |  0.903 | 0.795 | 0.700 | 0.806 | 0.503 | 0.891 | 0.770 | 0.850 |
| Exp_36 | Run_03 | + new augs        | 0.875 | 0.772 | 0.680 | 0.748 | 0.502 | 0.868 | 0.795 | 0.840 |
| Exp_36 | Run_04 | - prob new        | 0.879 | 0.792 | 0.681 | 0.787 | 0.499 |0.870 | 0.760 | 0.868 |
| Exp_36 | Run_05 | - extra rgb shift | 0.879 | 0.781 | 0.676 | 0.787 | 0.514 | 0.885 | 0.788 | 0.848 |
| Exp_36 | Run_06 | - chnl shuffle| 0.883 | 0.798 | 0.698 | 0.815 | 0.558 | 0.855 | 0.803 | 0.862 | 
| Exp_36 | Run_07 | change blur   | 0.864 | 0.767 | 0.673 | 0.672 | 0.531 | 0.864 | 0.75 | 0.864 |
| Exp_36 | Run_08 | change blur   | 0.885 | 0.792 | 0.697 | 0.801 | 0.501 | 0.878 | 0.753 | 0.880 |
| Exp_36 | Run_09 | change blur   | 0.879 | 0.788 | 0.685 | 0.811 | 0.530 | 0.879 | 0.753 | 0.861 |
| Exp_36 | Run_10 |  removed blur | 0.889 | 0.795 | 0.690 | 0.791 | 0.529 | 0.865 | 0.723 | 0.867 |
| Exp_36 | Run_11 |  put back blur| 0.881 | 0.779 | 0.684 | 0.779 | 0.541 | 0.875 | 0.783 | 0.856 |
| Exp_36 | Run_12 | - togray      | 0.891 | **0.798** | **0.705** | 0.805 | 0.507 | 0.872 | 0.781 | 0.871 |

Interesting that rat performance was so good on Run_03.

**Run_03** re-organised the augmentation to group the chanel color augmentations into a one-of group.  Added Channel Swap,  HSV. with p=0.8  Also added JPG compression seperately, with min/max [60,100] p=0.2.

**Run_04** Kept the above, but changed the min JPG compression from 60 to 70, and reduced the probability of the chanel augmenations to from 0.8 to p=0.5.

**Run_05** I noticed that I had repeated the rgb_shift augmentation within the channel augmentations, so this was happening 2x more than the others.  Removed in run_05 to see what happens.  Training performance was identical, but on the secret set it went backwards a little.  Seo either rgb_shift was useful, or one of the other options is bad, but happening more often.

**Run_06** removed channel shuffle.  Since this is one of the new ones and not tested yet.  Performance improved.  Will keep this out for future experiments.

**Run_07** increased probability of gaussian blur from 0.1 to 0.3 to see if this is helping or hurting.

**Run_08** removed sigma limit from gaussian blur, and returned p to 0.1

**Run_09** (I think there may have accidentally been two gaussian blurs at this point, removed the one with default settings and put p=0.2 for the remaining one sith sigma limit = 0.9)

**Run_10**  Removed gaussian blur entirely

**Run_11** Put back Gaussian blur with sigma limit 0.9, p=0.2.  This doesn't quite make sense, looks like run_09

**Run_12** Gaussian blur back to the way it was (but with only one, sigma limit 0.9, p=0.1), reduced the togray from 0.3 to 0.2

From the above, I concluded that the Gaussian Blur is somewhat useful, but not a big factor.  The other new augmentions aren't making much difference either, so I'm going to pause this for now.  They could well be helping to generalise, but I'm not seeing it with my current test set, which might not have enough diversity.


### Experiment 36 - Experimenting with Learning rate Schedule

Up until this point all learning rate scheduling was using CosineAnealingWithWarmRestarts.  But the cycle length was set to 16 cycles.  This meant that the learning was effectively being terminated at 16 epochs, because at this point there was an unreasonably large jump in learning rate.  This explains why I often saw a big drop in accuracy at this step.

Further experiments hold the augmentations as per Exp_36 Run_12, but adjusted learning rate schedule.

| **Exp** | **Run_ID** | **Augmentation** |**Test_mAP** | **Ind_mAP** | **Ind_F1** | **Cat** | **Mouse** |  **Possum**  |  **Rat** | **Stoat** | 
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **Exp_24** | Run_02 | 200 | 0.918 | **0.844** | **0.751** | 0.889 | 0.587 | 0.890 | 0.778 | 0.858 |
| Exp_32 | Run_01 | 200 |  0.905 | **0.809** | **0.716** | 0.807 | 0.568 | 0.877 | 0.772 | 0.864 |
| Exp_36 | Run_12 | - togray      | 0.891 | **0.798** | **0.705** | 0.805 | 0.507 | 0.872 | 0.781 | 0.871 |
| Exp_36 | Run_13 |  LR cycle = 5 |  |  |  |  |  |  |  |  |
| Exp_36 | Run_14 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_15 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_16 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_17 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_18 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_19 |       |  |  |  |  |  |  |  |  |
| Exp_36 | Run_20 |       |  |  |  |  |  |  |  |  |

**Run_13**  Reduced the learning rate cycle to 5 to see what happens if the LR is bumped up earlier in the training.  This didn't work, performance dropped after the 5th cycle and stopped at 6.5.

**Run_14**  Increased the LR cycle to 20, and set the minimum to 1e-5.  This looked promising as the val precision was still improving even when training stopped at 17 cycles due to val loss, but it was the loss rate change was nice and stable.  Maybe considering increasing the patience?

**Run_15** Had another try at a lower learning cycle.  6 cycles, 2 cycles warmup, max 1e-3, min 1e-6.    As with run 13, training fell over at the end of the cycle.

**Run_16** 