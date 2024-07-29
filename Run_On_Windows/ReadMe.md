#Readme

This folder contains bat files, that themselves run other files located in hte Python Directory and sub-directory.  

My plan for Windows is to use a `.bat` to initiate anything, just so that it can temporarily set permisions for the powershell script to do the grunt work for installing libraries, handling Python Environments, and runing Python Scripts.

## `Infer_Dataset.bat`
This is how to use the classifier on new data.  All you need to do is point to a settings file, and also the directory containing the dataset.  It is OK to have data in sub-folders, the directory will be searched recursively.  The outputs will be saved to the parent directory by default.
