$scriptDir = $PSScriptRoot
$parentPath = (Split-Path -Path $scriptDir -Parent)
$pythonscriptPath = Join-Path -Path $parentPath -ChildPath 'Train_Evaluate_Log.py'
$condaEnvName = "cv_pytorch"

# Now start up a python environmnent, and run Train_Evaluate_Log.py with the imagery folder and settings filepath as arguments
conda deactivate
conda activate $condaEnvName
Set-Location -Path $parentPath 
python $pythonscriptPath
conda deactivate