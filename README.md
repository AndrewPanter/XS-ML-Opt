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

## Simulated Annealing
The SimulatedAnnealing director contains the scripts used to perform the simulated annealing optimization.

In order to run the simulated annealing, you will first need to extract the .keras files for the optimized surrogate models:

```
tar -xvf CNN-Keras.tar.gz
tar -xvf ResNet-Keras.tar.gz
```

Next, you will need to call the runner script for the simulated annealing process.  If you wish to run the simulated annealing with the CNN surrogate model call:

```
python SimulatedAnnealing/Simulated_Annealing_Runner.py cnn
```

And if you wish to run with the ResNet model, call:

```
python SimulatedAnnealing/Simulated_Annealing_Runner.py resnet
```

The plots of the convergence of the predicted reactivity error will be saved into the Convergence_Plots directory.  And a .txt file will be generated that contains the optimized group structure from each of the simulated annealing runs.  If the .txt file already exists, the new results will simply be appended to the end of file.

