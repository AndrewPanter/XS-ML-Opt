# Simulated Annealing Methods for Final Project
# NERS 590: ML for Nuclear Engineers
# Thomas Jayasankar, Andrew Panter, Meredith Thibeault

# Import necessary libraries:
from Simulated_Annealing_Functions import *

import warnings
warnings.filterwarnings("ignore")

# Basic packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import sys
import os

# Keras:
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Input
from keras.layers import RepeatVector, TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint

# To display images:
from IPython.display import Image

# Import TensorFlow and set up GPU memory configuration:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Initialize random number generator seed:
random.seed(42)

# Input Handling:
if sys.argv[1] == "cnn": #If you wish to run with CNN surrogate model:
    best_model_path = "cnn_best_model.keras"
    coarse_figure_name = "coarse_errors_cnn"
    fine_figure_name = "fine_errors_cnn"
    save_file = "best_structures_cnn.txt"
elif sys.argv[1] == "resnet":
    best_model_path = "resnet_best_model.keras"
    coarse_figure_name = "coarse_errors_resnet"
    fine_figure_name = "fine_errors_resnet"
    save_file = "best_structures_resnet.txt"
else:
    raise ValueError("Unrecognized Surrogate Model Type Input. Please Use Either 'cnn' or 'resnet'.")

# Create the Save File if it Does Not Yet Exist:
os.system("touch " + save_file)

# Loading the best model into memory from the .keras file:
best_model = load_model(best_model_path)

# Set up simulated annealing parameters:
num_outer_iterations = 100
num_inner_iterations = 50
initial_temperature = 10 # This (and the cooling schedule) should be tuned
cooling_exponent = 0.8 #Some number between 0 and 1 (lower numbers mean more aggressive cooling)
jump_size = 1
num_runs = 30

# Set up problem parameters:
num_fine_bounds = 399
num_coarse_bounds = 49

start_time = time.time()

for run_number in range(num_runs):

    # Create Initial Group Structure:
    initial_group_structure = Create_Random_Group_Structure(num_fine_bounds,num_coarse_bounds)

    # Output Initial Predicted Error:
    initial_error = best_model.predict(np.array(initial_group_structure).reshape(-1,399,1))
    print("Predicted Error of Initial Group Structure" + str(initial_error))


    current_group_structure = initial_group_structure
    current_error = initial_error
    coarse_errors = [initial_error[0][0]]
    fine_errors = [initial_error[0][0]]
    temperatures = []

    minimum_error = 1000
    minimum_error_group_structure = initial_group_structure

    for m in range(num_outer_iterations):

        temperature = initial_temperature * (cooling_exponent ** m)

        for n in range(num_inner_iterations):

            new_group_structure = Create_Neighboring_Group_Structure(current_group_structure,jump_size=jump_size)
            new_error = best_model.predict(np.array(current_group_structure).reshape(-1,399,1))

            if new_error[0][0] < minimum_error:

                minimum_error_group_structure = new_group_structure
                minimum_error = new_error[0][0]

            if Decide_Acceptance(current_error[0][0],new_error[0][0],temperature):

                current_group_structure = new_group_structure
                current_error = new_error

            temperatures.append(temperature)
            fine_errors.append(current_error[0][0])

        coarse_errors.append(current_error[0][0])
        print("Error at end of outer iteration #" + str(m) + ": " + str(current_error))

    with open(save_file, "a") as save_writer:
        save_writer.write("\n"+(str(minimum_error_group_structure)[1:-1] + ", " + str(minimum_error)).replace(" ", ""))
    save_writer.close()


    #print(errors)
    plt.figure()
    plt.plot(range(num_outer_iterations+1),coarse_errors)
    plt.xlabel("Outer Iteration #")
    plt.ylabel("Predicted k-eff Error")
    plt.title("Simulated Annealing k-eff Error Convergence Plot")
    plt.savefig("Convergence_Plots/" + coarse_figure_name + "_" + str(run_number+1)+ ".png")

    #print(errors)
    plt.figure()
    plt.plot(range(len(fine_errors)),fine_errors)
    plt.xlabel("Inner Iteration #")
    plt.ylabel("Predicted k-eff Error")
    plt.title("Simulated Annealing k-eff Error Convergence Plot")
    plt.savefig("Convergence_Plots/" + fine_figure_name + "_" + str(run_number+1)+ ".png")

    #print(errors)
    plt.figure()
    plt.plot(range(len(temperatures)),temperatures)
    plt.xlabel("Inner Iteration #")
    plt.ylabel("Temperature")
    plt.title("Simulated Annealing Cooling Schedule")
    plt.savefig("Convergence_Plots/cooling_schedule.png")

    print("Predicted Error of Initial Group Structure: " + str(initial_error[0][0]))
    print("Predicted Error of Optimal Group Structure: " + str(minimum_error))

end_time = time.time()

print("Total Runtime: " + str(end_time-start_time) + " seconds")




