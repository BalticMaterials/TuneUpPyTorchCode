# TuneUpPyTorchCode

$ sudo apt update && sudo apt install software-properties-common -y


$ sudo add-apt-repository ppa:deadsnakes/ppa 

$ sudo apt update && sudo apt install python3.11 

$ poetry env use python3.11  

and install 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

direct after activating the env with $ poetry shell


For the Usage of VSC - Remote Explore 
( in some cases it is needed to get the user sudo to work)
$ sudo chown -R non_root_username /path/to/directory