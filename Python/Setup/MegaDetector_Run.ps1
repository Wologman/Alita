param ([string]$externalImagePath = "C:\Users\ollyp\Desktop\Project\Data\Independent_Images")

# Directory & File Names
$dataFolderName = "Data"
$defaultImagesFolderName = "Inference_Images"
$mdOutputFileName = "mdPredictions.json"
$mdScriptRelativelPath = "MegaDetector_Repo\MegaDetector\detection\run_detector_batch.py"
$condaEnvName = "cameratraps-detector"
$dir1Name = "MegaDetector" 
$dir2Name = "yolov5"
$mdModelName = "md_v5a.0.0.pt"
$downloadScriptName = "MegaDetector_Download_Model.ps1" 

# Derived Paths
$scriptDir = $PSScriptRoot
$parentPath = Split-Path -Path $scriptDir -Parent
$grandparentPath = Split-Path -Path $parentPath -Parent
$defaultImagePath =  Join-Path -Path $grandparentPath -ChildPath "\$dataFolderName\$defaultImagesFolderName"
$mdScriptPath =  Join-Path -Path $grandparentPath -ChildPath $mdScriptRelativelPath

# Check if a file path argument was provided for the images
if ($externalImagePath) {
    Write-Host "Sourcing images from: $externalImagePath"
    $imagesPath =  $externalImagePath
} else {
    Write-Host "No alternative file path provided. Sourcing Images from the default location: $defaultImagePath"
    $imagesPath = $defaultImagePath
}
$mdOutputPath = (Join-Path -Path  $imagesPath -ChildPath $mdOutputFileName)

# Search for in grandparent's child directories for the directories and model file
$dir1 = Get-ChildItem -Path $grandparentPath -Recurse -Directory | Where-Object { $_.Name -eq $dir1Name } | Select-Object -First 1
$dir2 = Get-ChildItem -Path $grandparentPath -Recurse -Directory | Where-Object { $_.Name -eq $dir2Name } | Select-Object -First 1
$mdModel = Get-ChildItem -Path $grandparentPath -Recurse -File | Where-Object { $_.Name -eq $mdModelName } | Select-Object -ExpandProperty FullName -First 1
# Check if both directories were found
if ($null -ne $dir1 -and  $null -ne $dir2) {
    # Combine the paths of dir1 and dir2 into the PYTHONPATH variable
    $pythonPath = $dir1.FullName + ";" + $dir2.FullName
    # Set the PYTHONPATH environment variable
    [System.Environment]::SetEnvironmentVariable("PYTHONPATH", $pythonPath, [System.EnvironmentVariableTarget]::Process)
    Write-Host "PYTHONPATH has been set to:"
    Write-Host $pythonPath
} else {
    Write-Host "One or both of the directories ($dir1Name and $dir2Name) were not found,\
     re-install with MegaDetector_Setup.ps1."
}

# Check if the model file was found, and if not, try to download it
if ($null -ne $mdModel){
        Write-Host "The MegaDetector Pytorch Model was found at $mdModel"                                                                                                      ]
} else{
    Write-Host "The MegaDetector Pytorch Model was not found, attempting to download"
    $downloadScript = Join-Path -Path $scriptDir -ChildPath $downloadScriptName
    if (Test-Path -Path $downloadScript) {
        Write-Host "Calling the downloading script"
        & $downloadScript
        $mdModel = Get-ChildItem -Path $grandparentPath -Recurse -File | Where-Object { $_.Name -eq $mdModelName } | Select-Object -ExpandProperty FullName -First 1
    } else {
        Write-Host "Error: The downloading shell script was not found, you could try to manually download put the \
        pyTorch model md_v5a.0.0.pt, and put it in a folder of your choosing under the project folder."
    }  
}

conda deactivate
conda activate $condaEnvName
#Now hopefully it's all in place, run MegaDetector!
python $mdScriptPath $mdModel $imagesPath $mdOutputPath --recursive --output_relative_filenames --quiet 
#Other options: --use_image_queue --checkpoint_frequency 10000 --checkpoint_path "some path" --resume_from_checkpoint "some_path"
conda deactivate
conda activate cv_pytorch