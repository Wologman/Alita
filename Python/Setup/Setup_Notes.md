# Setup on Windows

[Instructions from the repo](https://github.com/agentmorris/MegaDetector/blob/main/megadetector.md#using-the-model)

## Introduction
MegaDetector has a fairly simple environment yaml file `environment-detector.yml` located in it's repo.  That can be setup as per usual with Conda, ideally using Mamba as the solver. MegaDetector uses Python 3.8.15, PyTorch 1.1, & CudaToolkit 11.3.  All quite old compared to the environment I've made for my classifier, but it would be worth investigating if I can install my addditional packages into it.  The risk to that is that any future versions of MD could make my classifier fall over.

Additional requirement to run MegaDetector is to download the repo to a sensible location, and ensure that the location is included in the PythonPath variable.  This doesn't work on the DOC network as Anaconda is blocked.  It works fine from home, or by hot-spotting from a mobile with reliable data speeds.

There are two models to choose from,  MDv5a &  MDv5b, but  MDv5a was trained on a more diverse dataset, so use that.  


## Instructions to setup MegaDetector
1. Install [libturbojpg](https://libjpeg-turbo.org/), try downloading from [here](https://sourceforge.net/projects/libjpeg-turbo/files/) 
2. Install [Anaconda](https://www.anaconda.com/download) or [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) 
3. Ensure that Conda is in the system PATH variable, so that it can be found through terminal commands. Usually this is just a box tick during installation, but can be done after the fact through windows control panel.
4. Put all the folders from *Project* down in fast storage location that you want to run it from.  Ideally this should be internal SSD or at least an external NVMe SSD with a fast connection like Thunderbolt-3 (USB4/Thunderbolt-3 at 40Gb/s roughly matches NVMe read/write speeds at 5000MB/s).
4. Open PowerShell as administrator, and type `conda --version`  If you get an anwser then so far so good.
Make [LibMamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) the default solver:
```bash
    conda update -n base conda
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
```
Run the file `MegaDetector_Setup.ps1` to setup Megadetector
5. (optional) If you want to run PowerShell through VScode, first open it up in its own terminal, and enter the following:  

```bash
conda init powershell
set-executionpolicy unrestricted
conda config --set auto_activate_base false
```
Then close and open VSCode.  Now the PowerShell terminal in VScode will let you activate environments. I'm not convinced this is working as it should on DOC computers, as PowerShell seems to have some restrictions.

## Instructions to setup the Classifier
After doing all the above:
1. Open PowerShell as administrator
2. Run the file `Classifier_Setup.ps1`