# Before running this script, please visit 
# https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1710&target_type=runfilelocal
# and download suitable installers. 
# Then modifiy the command below where needed.

cd YOUR_PATH # Fill in the installation path of installers.
echo "Installing cuda 9.2"
sudo sh cuda_9.2.148_396.37_linux.run # This may differ for different platforms. Please refer to the website for exact command.
sudo sh cuda_9.2.148.1_linux.run # This may differ for different platforms. Please refer to the website for exact command.
echo "Configurating new conda environment..."
source ./.set_env
conda create -n fastflow python=3.6
conda activate fastflow
echo "Installing packages..."
conda install pytorch=0.4.1 cuda92 -c pytorch
conda install opencv
conda install -c conda-forge pillow
conda install -c conda-forge pyyaml
cd ./FastFlowNet/models/correlation_package/
python setup.py build
python setup.py install
echo "Finished"