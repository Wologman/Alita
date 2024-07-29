Write-Host "Running Inference on new imagery data"
$scriptDir = $PSScriptRoot
$parentPath = (Split-Path -Path $scriptDir -Parent)
$grandparentPath = (Split-Path -Path $parentPath -Parent)
explorer $grandparentPath
#$settingsPath = Join-Path -Path $grandparentPath -ChildPath 'Settings\Current\'
$defaultSettingsPath = Join-Path -Path $grandparentPath -ChildPath 'Settings\Default\'
$pythonscriptPath = Join-Path -Path $parentPath -ChildPath 'Inference.py'
$condaEnvName = "cv_pytorch"
$megaEnvName = "cameratraps-detector"


#Check the python environments exist
$classifierEnvironmentExists = (conda info --envs | Select-String -Pattern "^$condaEnvName").Count -gt 0
$megaDetectEnvironmentExists = (conda info --envs | Select-String -Pattern "^$megaEnvName").Count -gt 0

if (-not $classifierEnvironmentExists){
    Write-Host "Exiting because the Conda Environment $condaEnvName can not be detected "
    exit}
if (-not $megaDetectEnvironmentExists){
    Write-Host "Exiting because the Conda Environment $megaEnvName can not be detected "
    exit}

Write-Host "Both required Conda environments $megaEnvName and $condaEnvName have been detected"

# Prompt the user to select a file
$settingsFiles = Get-ChildItem -Path $defaultSettingsPath -Recurse -Filter '*.yaml' -File

if ($settingsFiles.Count -eq 1){
    $settingsFile = $settingsFiles[0].FullName
} else {

    Write-Host "Please enter the path to the settings file you want to use. On Windows 10 you can use shift 
    + right click on the folder, then copy as path. On Windows 11 just right click.
    Then right click on this command window to paste (or left click and press enter)."
    pause
    $settingsFile = Get-Clipboard
    $settingsFile = $settingsFile -replace '"'  #remove any extra quotes
    $settingsFile = $settingsFile.Trim()  #remove any unwanted surrounding spaces

    # Check if the path exists
    if (Test-Path -LiteralPath "$settingsFile"  -PathType leaf) {
        Write-Host "Selected the settings file: $settingsFile"
    } else {
        Write-Host "Warning: The specified settings file  $($settingsFile) does not exist, exiting PowerShell."
        return
    }
}

Write-Host "Please right click on the folder containing the images you want to run the classifier on, and copy the file path.  
Then right click on this command window (or left click and press enter)"
pause
$dataPath = Get-Clipboard
$dataPath = $dataPath -replace '"'  #remove any extra quotes
$dataPath = $dataPath.Trim()  #remove any unwanted surrounding spaces

# Check if the path exists
if (Test-Path -LiteralPath $dataPath -PathType Container) {
    Write-Host "Selected root folder of the dataset you want to run the classifier on: $dataPath"
} else {
    Write-Host "The specified path  $($dataPath) does not exist."
}


$foundFiles = Get-ChildItem -Path $dataPath -Recurse -Filter 'mdPredictions.json' -File
# Display the full paths of the found files
if ($foundFiles.Count -gt 1) {
    Write-Host "The following MegaDetector output files mdPredictions.json were found"
    $foundFiles | ForEach-Object { $_.FullName }
    Write-Host "If any filepaths to images have changed, you will need delete mdPredictions.json manually now, before continuing"
    Write-Host "If no filpeaths to images have changed, you can continue and use this version of mdPredictions.json"
    $userResponse = Read-Host "Do you want to continue? (Y/N)"

    if ($userResponse -ne 'Y' -and $userResponse -ne 'y') {
        Write-Host "Script terminated by user."
        exit
    } 
}

# Now start up a python environmnent, and run Inference.py with the imagery folder and settings filepath as arguments
conda deactivate
conda activate $condaEnvName
Set-Location -Path $parentPath 
python $pythonscriptPath --settingsPath $settingsFile --dataPath $dataPath
conda deactivate