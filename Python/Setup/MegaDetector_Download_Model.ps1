$modelUrl = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
$modelName ="md_v5a.0.0.pt" 
$scriptDir = $PSScriptRoot
$modelsDirName = 'Models'

$parentPath = (Split-Path -Path $scriptDir -Parent)
$grandparentPath = (Split-Path -Path $parentPath -Parent)


# Combine the grandparent directory path and the new directory name
$modelsDir = Join-Path -Path $grandparentPath -ChildPath $modelsDirName
$modelFilePath = Join-Path -Path $modelsDir -ChildPath $modelName

# Ensure the 'Models' directory exists, create it if not
if (-not (Test-Path -Path $modelsDir -PathType Container)) {
    New-Item -Path $modelsDir -ItemType Directory
}

# Check if the file already exists
if (Test-Path -Path $modelFilePath) {
    Write-Host "Model File already exists at $modelFilePath. No need to download."
} else {
    # Download the file using Invoke-WebRequest
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelFilePath
    Write-Host "File downloaded and saved to $modelFilePath"
}
