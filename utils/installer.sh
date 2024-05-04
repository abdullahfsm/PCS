cd ~/PCS/utils

#installing cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


#installing libcudnn
export last_public_key=3bf863cc #
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${last_public_key}.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8


# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
eval "$(/$HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda create -y -n osdi24 python=3.6.10
echo "conda activate osdi24" >> ~/.bashrc


# get ray version
curl -o ray-2.0.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl "https://zenodo.org/records/11111741/files/ray-2.0.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl?download=1"

# installing packages
python3 -m pip install -r requirements.txt


# Driver + cuda toolkit

# # install latest pip
# sudo wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
# python3 get-pip.py

# # install tensorflow + ray
# sudo python3 -m pip install pandas
# sudo python3 -m pip install tabulate
# sudo python3 -m pip install matplotlib
# sudo python3 -m pip install hiredis
# sudo python3 -m pip install ray
# sudo python3 -m pip install statsmodels
# sudo python3 -m pip install jmetal_ray-1-py3-none-any.whl


# Driver + cuda toolkit

# cd /opt

# #wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# #sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# #wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu1804-11-2-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# # cuDnn
# sudo dpkg -i cudnn-local-repo-ubuntu1804-8.4.1.50_1.0-1_amd64.deb
# sudo cp /var/cudnn-local-repo-ubuntu1804-8.4.1.50/cudnn-local-BA71F057-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get install libcudnn8=8.4.1.50-1+cuda11.6
# sudo apt-get install libcudnn8-dev=8.4.1.50-1+cuda11.6


# # install latest pip
# sudo wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
# python3 get-pip.py

# # install tensorflow + ray
# sudo python3 -m pip install tensorflow
# sudo python3 -m pip install pandas
# sudo python3 -m pip install tabulate
# sudo python3 -m pip install matplotlib
# sudo python3 -m pip install hiredis
# sudo python3 -m pip install ray-2.0.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl
# #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64