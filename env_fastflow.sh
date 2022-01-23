cd /xmotors_ai_shared/cvbj/luojw4
echo "Installing cuda 9.2"
sudo sh cuda_9.2.148_396.37_linux.run
sudo sh cuda_9.2.148.1_linux.run
echo "Configurating new conda environment..."
source ./.set_env
conda create -n fastflow python=3.6
conda activate fastflow
echo "Installing packages..."
conda install pytorch=0.4.1 cuda92 -c pytorch
conda install opencv
conda install -c conda-forge pillow
cd ./FastFlowNet/models/correlation_package/
python setup.py build
python setup.py install
echo "Finished"