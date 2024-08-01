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

    $environmentExists = (conda info --envs | Select-String -Pattern "^$predatorEnvironment").Count -gt 0
    if ($environmentExists) {
        Write-Host "The Conda environment '$predatorEnvironment' was created"
    } else { 
            Write-Host "The Conda environment '$predatorEnvironment' has not been detected even though we tried to create it"
            Write-Host "The Most likely reason for this is your computer fell asleep or your internet connection was disrupted"
            Write-Host "Sorry all I can suggest is you try to re-run the setup_everything.bat script, and/or try from a more solid internet connection"
            }
    }

conda deactivate
Write-Host "The classifier environment setup (cv_pytorch) script has completed"
Write-Host "The the following Conda Environments are currently installed"
conda deactivate
conda env list
Write-Host "You should see at least '$predatorEnvironment' and 'cameratraps-detector' listed"
Write-Host "If this is not the case, sorry you will need to try running setup_everything.bat again"