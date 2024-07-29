#Housekeeping
conda deactivate
conda update conda
conda clean --all --yes

$newDirectoryName = "MegaDetector_Repo"
$modelsDirectoryName = "Models"
$modelFileName = "md_v5a.0.0.pt"
$modelURL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
$scriptDir = $PSScriptRoot
$parentPath = (Split-Path -Path $scriptDir -Parent)
$grandparentPath = (Split-Path -Path $parentPath -Parent)
$downloadModel = $false
$megaEnvironment = 'cameratraps-detector'
$megaFilePath = "MegaDetector/envs/environment-detector.yml"


$environmentExists = (conda info --envs | Select-String -Pattern "^$megaEnvironment").Count -gt 0


# Combine the grandparent directory path and the new directory name
$newDirectoryPath = Join-Path -Path $grandparentPath -ChildPath $newDirectoryName
$modelsDirectoryPath = Join-Path -Path $grandparentPath -ChildPath $modelsDirectoryName
$modelFilePath =  Join-Path -Path $modelsDirectoryPath -ChildPath $modelFileName


if ($environmentExists) {
    conda activate $megaEnvironment
    Write-Host "The Conda environment '$megaEnvironment' already exists, you shouldn't need to do anything more to use it."
    Write-Host "To reinstall, first run the following command in a terminal: conda env remove --name cameratraps-detector"
    Write-Host "`nPackages already installed in the '$megaEnvironment' Conda environment:"
    conda list
    conda deactivate
    } else {
    # Setup of the MegaDetector Python environment
    Set-Location -Path "$newDirectoryPath"
    Write-Host "Changed directory to: $newDirectoryPath"    
    # Doesn't work on DOC LAN network.  Go home, or hotspot from a phone, or try the guest wifi network
    Write-Host "Attempting to setup the megaDetector environment now"    
    conda env create --file $megaFilePath
    $environmentExists = (conda info --envs | Select-String -Pattern "^$megaEnvironment").Count -gt 0
    if ($environmentExists) {"The Conda environment '$megaEnvironment' has been create successfully"}
    }

# Check if the models directory already exists
Set-Location -Path $scriptDir
if (!(Test-Path -Path $modelsDirectoryPath -PathType Container)) {
    # Ask the user for confirmation to proceed
    $confirmation = Read-Host "The directory '$modelsDirectoryName' is missing. Do you want to download it? (Y/N)"

    if ($confirmation -eq 'Y' -or $confirmation -eq 'y') {
        # Create the new directory
        New-Item -Path $modelsDirectoryPath -ItemType Directory
        Write-Host "Models Directory created successfully."
        $downloadModel = $true
    } else {
        Write-Host "You still need to manually create a Models directory then download a model file to run the MegaDetector."
        $downloadModel = $false
    }
}

if ($downloadModel -eq $true) {
    Set-Location -Path $modelsDirectoryPath
    Write-Host "Changed directory to: $mpdelsDirectoryPath"
    Invoke-WebRequest -Uri $modelURL -OutFile $modelFilePath
}

Write-Host "The MegaDetector setup script has completed without error"