# This script should be run from the 'Setup' folder
$predatorEnvironment = 'cv_pytorch'
$predatorYamlFile = 'win_predator_env.yaml'
$environmentExists = (conda info --envs | Select-String -Pattern "^$predatorEnvironment").Count -gt 0


if ($environmentExists) {
    conda activate $predatorEnvironment
    Write-Host "The Conda environment '$predatorEnvironment' already exists, you should not need to do anything further to run it." 
    Write-Host "To reinstall first run the following command in a terminal: conda env remove --name cv_pytorch"
    } else {   
    # Doesn't work on DOC LAN network.  Go home, or hotspot from a phone, or try the guest wifi network
    conda env create --file $predatorYamlFile
    conda activate $predatorEnvironment
    Write-Host "The Conda environment '$predatorEnvironment' was created"
    }

conda deactivate
Write-Host "The classifier environment setup (cv_pytorch) script has completed without error"
Write-Host "The the following Conda Environments are currently installed"
conda deactivate
conda env list