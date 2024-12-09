# XS-ML-Opt
NERS 590 Project Repo - using machine learning to determine optimal coarse group energy bounds for multigroup neutron transport simulations

## ModelBuilding
The model building folder contains the Python scripts used in hyperparameter tuning and builing the CNN and ResNet models. The files CNN-with-tuning.py and ResNet-with-tuning.py both go through the entire hyperparameter tuning process and build and evaluate the final models. The files CNN-Final.py and ResNet-Final.py build and evaluate the final model with the hyperparameters found during tuning. To run these files a Python virtual environment was used with the following commands. 

```
conda create -n tfgpu python=3.11
conda activate tfgpu
pip install tensorflow[and-cuda]
#Install other relevant packages
pip install pandas matplotlib scikit-learn seaborn numpy scikit-optimize
```
Note: the CNN file takes approximately 1 hour to run and the ResNet model about 5 hours to run. To run the files the compiled_data file must be unzipped and placed in the same folder as the Python sctips. 


Please add to this README as you alter the directory structure of the repo.
