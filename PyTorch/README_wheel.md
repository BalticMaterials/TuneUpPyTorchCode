#Add Jupyther 

$poetry config installer.modern-installation false  --local
Link -> https://github.com/python-poetry/poetry/issues/7686 


### INSTALLING PREVIOUS VERSIONS OF PYTORCH
# https://pytorch.org/get-started/previous-versions/
### Torch 1.13
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

### Torch 2.0
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118


### Wheels:
https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl
