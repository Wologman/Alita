@echo off
cd ..\Python\Setup
powershell -ExecutionPolicy Bypass -Command "conda init powershell"
powershell -ExecutionPolicy Bypass  -File "MegaDetector_Setup.ps1"
powershell -ExecutionPolicy Bypass -NoExit -File "Classifier_Setup.ps1"
