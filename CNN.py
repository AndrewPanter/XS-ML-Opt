# %%
# Import packages and load data
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

#Keras/TensorFlow
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Add, ReLU, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout
from tensorflow.keras.metrics import R2Score
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from skopt import gp_minimize

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")
    
# load data
data = np.loadtxt('compiled_data.txt', delimiter=',') 

xdata = data[:, 0:-1]
ydata = data[:, -1]
xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata, test_size=0.2, random_state=42)

X_train = xtrain.reshape(-1, 399, 1)
X_test = xtest.reshape(-1, 399, 1)

# %%
# Build CNN Model
def build_CNN(layers, nodes, dropout_rate, learning_rate, input_shape=(399, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(nodes, kernel_size=3, activation='relu'))
    for _ in range(layers - 1):
        model.add(Conv1D(nodes, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2)) #2
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(learning_rate), metrics=['mae', 'R2Score'])
    return model

def train_evaluate(model, batch_size, epochs, X_train, ytrain, X_val, yval):
    history = model.fit(X_train, ytrain, epochs=epochs, validation_data=(X_val, yval), batch_size=batch_size, verbose=0)
    loss, mae, r2 = model.evaluate(X_test, ytest)
    return mae, r2, history

metrics_list = []

# %%
#hyper parameter tuning

#begin with manual tuning
print("manual training")
layers = [2, 4, 6]
nodes_per_layer = [32, 64, 128]
batch_size = 64
learning_rate = 0.001
dropout_rate = 0.2
epochs = 50

best_baseline_mae = float('inf')
best_baseline_config = None
best_baseline_model = None

for layer in layers:
    for nodes in nodes_per_layer:
        print("Current Model:", layer, nodes, dropout_rate, batch_size, learning_rate, epochs)
        model = build_CNN(layer, nodes, dropout_rate, learning_rate)
        mae, r2, manual_history = train_evaluate(model, batch_size, epochs, X_train, ytrain, X_test, ytest)
        if mae < best_baseline_mae:
            best_baseline_mae = mae
            best_baseline_config = (layer, nodes, dropout_rate, batch_size, learning_rate, epochs)
            best_baseline_model = model

        metrics_list.append(["Baseline", layer, nodes, batch_size, learning_rate, dropout_rate, epochs, mae, r2])

print("Best baseline configuration:", best_baseline_config)
print("Best baseline MAE:", best_baseline_mae, "Best baseline R2:", r2)

# %%
# random search
print("random search")
def RandomSearch(best_baseline, num_trials=10): #change back to 10
    best_random_mae = float('inf')
    best_random_config = None

    layers, nodes, dropout_rate, batch_size, learning_rate, epochs = best_baseline

    for _ in range(num_trials):
        lr = random.uniform(1e-4, 1e-2)
        dr = random.uniform(0.1, 0.5)
        bs = random.choice([32, 64, 128, 256, 512])

        print("Current Model:", layers, nodes, dr, bs, lr, epochs)
        model = build_CNN(layers, nodes, dr, lr)
        mae, r2, random_hisotry = train_evaluate(model, bs, epochs, X_train, ytrain, X_test, ytest)
        if mae < best_random_mae:
            best_random_mae = mae
            best_random_r2 = r2
            best_random_config = (layers, nodes, dr, bs, lr, epochs)

        metrics_list.append(["Random", layers, nodes, bs, lr, dr, epochs, mae, r2])

    return best_random_config, best_random_mae, best_random_r2

best_random_config, best_random_mae, best_random_r2 = RandomSearch(best_baseline_config)
print("Best random search configuration:", best_random_config)
print("Best random search MAE:", best_random_mae, "Best random search R2:", best_random_r2)

# %%
# grid search
print("grid training")
def GridSearch(best_random_config):
    best_grid_mae = float('inf')
    best_grid_config = None

    layers, nodes, dropout_rate, batch_size, learning_rate, epochs = best_random_config
    
    fine_learning_rates = np.linspace(learning_rate - 0.0005, learning_rate + 0.0005, num=5) #change back to 5
    fine_dropout_rates = np.linspace(dropout_rate - 0.05, dropout_rate + 0.05, num=5) #change back to 5
    fine_batch_sizes = [batch_size//2, batch_size, batch_size*2]

    for lr in fine_learning_rates:
        for dr in fine_dropout_rates:
            for bs in fine_batch_sizes:
                print("Current Model:", layers, nodes, dr, bs, lr, epochs)
                model = build_CNN(layers, nodes, dr, lr)
                mae, r2, grid_history = train_evaluate(model, bs, epochs, X_train, ytrain, X_test, ytest)
                if mae < best_grid_mae:
                    best_grid_mae = mae
                    best_grid_r2 = r2
                    best_grid_config = (layers, nodes, dr, bs, lr, epochs)
                metrics_list.append(["Grid", layers, nodes, bs, lr, dr, epochs, mae, r2])

    return best_grid_config, best_grid_mae, best_grid_r2

best_grid_config, best_grid_mae, best_grid_r2 = GridSearch(best_random_config)
print("Best grid search configuration:", best_grid_config)
print("Best grid search MAE:", best_grid_mae, "Best grid search R2", best_grid_r2)

# %%
print ("evolutionary search")
def initialize_population(best_grid_result, population_size):
    layers, nodes, dropout_rate, batch_size, learning_rate, epochs = best_grid_result
    population = []
    for _ in range(population_size):
        individual = {
            'layers': random.randint(layers-1, layers+1),
            'nodes': random.choice([nodes//2, nodes, nodes*2]),
            'dropout_rate': random.uniform(dropout_rate - 0.1, dropout_rate + 0.1),
            'batch_size': random.choice([batch_size//2, batch_size, batch_size*2]),
            'learning_rate': learning_rate * random.uniform(0.5, 1.5),
            'epochs': epochs,
            'fitness': None,
            'r2': None
        }
        population.append(individual)
    return population

def select_top_individuals(population, k=5):
    sorted_population = sorted(population, key=lambda ind: ind['fitness'])
    return sorted_population[:k]

def get_best_individual(population):
    return min(population, key=lambda ind: ind['fitness'])

def crossover_and_mutate(parents, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = {
            'layers': random.choice([parent1['layers'], parent2['layers']]),
            'nodes': random.choice([parent1['nodes'], parent2['nodes']]),
            'dropout_rate': random.choice([parent1['dropout_rate'], parent2['dropout_rate']]),
            'batch_size': random.choice([parent1['batch_size'], parent2['batch_size']]),
            'learning_rate': random.choice([parent1['learning_rate'], parent2['learning_rate']]),
            'epochs': parent1['epochs'],  # Keep epochs consistent
            'fitness': None,
            'r2': None
        }
        
        # Mutate some parameters
        if random.random() < 0.1:
            child['layers'] += random.randint(-1, 1)
        if random.random() < 0.1:
            child['nodes'] = random.choice([child['nodes']//2, child['nodes'], child['nodes']*2])
        if random.random() < 0.1:
            child['dropout_rate'] = random.uniform(0.1, 0.5)
        if random.random() < 0.1:
            child['batch_size'] = random.choice([child['batch_size']//2, child['batch_size'], child['batch_size']*2])
        if random.random() < 0.1:
            child['learning_rate'] *= random.uniform(0.5, 1.5)
        
        new_population.append(child)
    return new_population

def EvolutionarySearch(best_grid_config):
    population_size = 20  # Define a population size for evolution- change back to 20
    generations = 10  # Define number of generations- change back to 10
    population = initialize_population(best_grid_config, population_size)

    for generation in range(0, generations):
        # Evaluate fitness of the current population
        for individual in population:
            print("Current Model:", individual['layers'], individual['nodes'], individual['dropout_rate'], individual['batch_size'], individual['learning_rate'], individual['epochs'])
            model = build_CNN(individual['layers'], individual['nodes'], individual['dropout_rate'], individual['learning_rate'])
            mae, r2, hisotry = train_evaluate(model, individual['batch_size'], individual['epochs'], X_train, ytrain, X_test, ytest)
            individual['fitness'] = mae
            individual['r2'] = r2

            metrics_list.append(["Evolution", individual['layers'], individual['nodes'], individual['batch_size'], individual['learning_rate'], individual['dropout_rate'], individual['epochs'], mae, r2])

        # Select top performers for the next generation
        selected_individuals = select_top_individuals(population)

        # Generate new individuals through mutation and crossover
        new_population = crossover_and_mutate(selected_individuals, population_size)
        population = new_population

    for individual in population:
        model = build_CNN(individual['layers'], individual['nodes'], individual['dropout_rate'], individual['learning_rate'])
        mae, r2, hisotry = train_evaluate(model, individual['batch_size'], individual['epochs'], X_train, ytrain, X_test, ytest)
        individual['fitness'] = mae
        individual['r2'] = r2

    return get_best_individual(population)



best_evolution = EvolutionarySearch(best_grid_config)
print("Best evolutionary search configuration:", best_evolution['layers'], best_evolution['nodes'], best_evolution['dropout_rate'], best_evolution['batch_size'], best_evolution['learning_rate'], best_evolution['epochs'])
print("Best evolutionary search MAE:", best_evolution['fitness'], "Best evolutionary search R2", best_evolution['r2'])


# %%
# print("bayesian optimization")
# def BayesianOptimization(best_evolution):
#     bounds = [(best_evolution[0]['layers']-1, best_evolution[0]['layers']+1),
#         (min(best_evolution[0]['nodes'], best_evolution[1]['nodes']), max(best_evolution[0]['nodes'], best_evolution[1]['nodes'])),
#         (min(best_evolution[0]['learning_rate'], best_evolution[1]['learning_rate']), max(best_evolution[0]['learning_rate'], best_evolution[1]['learning_rate'])),
#         (best_evolution[0]['dropout_rate']-0.1, best_evolution[0]['dropout_rate']+0.1),
#         (min(best_evolution[0]['batch_size'], best_evolution[1]['batch_size']), max(best_evolution[0]['batch_size'], best_evolution[1]['batch_size']))
#     ]

#     def objective(params):
#         print("Current Model:", params['layers'], params['nodes'], params['dropout_rate'], params['batch_size'], params['learning_rate'])
#         model = build_CNN(params['layers'], params['nodes'], params['dropout_rate'], params['learning_rate'])
#         mae, r2, history = train_evaluate(model, params['batch_size'], params['epochs'], X_train, ytrain, X_test, ytest)
#         metrics_list.append([params['layers'], params['nodes'], params['batch_size'], params['learning_rate'], params['dropout_rate'], params['epochs'], mae, r2])
#         return mae

#     print("BOUNDS")
#     print(bounds)

#     result = gp_minimize(objective, bounds, n_calls=5) # change back to 50
#     return result

# best_bayesian = BayesianOptimization(best_evolution)
# layers, nodes, dropout_rate, batch_size, learning_rate = best_bayesian.x
# print("Best Bayesian optimization configuration:", layers, nodes, dropout_rate, batch_size, learning_rate)

# %%
metrics_df = pd.DataFrame(metrics_list, columns =['Hyperparameter Strategy', 'Layers', 'nodes_per_layer', 'batch_size', 'learning_rate', 'dropout_rate', 'epochs', 'MAE', 'R2'])

metrics_df = metrics_df.sort_values(by='MAE', ascending=True)

metrics_df.to_csv('metrics_CNN.txt', sep='\t', index=False)

layers = best_evolution['layers']
nodes = best_evolution['nodes']
dropout_rate = best_evolution['dropout_rate']
batch_size = best_evolution['batch_size']
learning_rate = best_evolution['learning_rate']
epochs = best_evolution['epochs']

final_model = build_CNN(layers, nodes, dropout_rate, learning_rate)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'R2Score'])
callback = ModelCheckpoint('checkpoints/cnn_best_model.keras', 
                                      monitor='val_mae', 
                                      save_best_only=True)
final_history = final_model.fit(X_train, ytrain, epochs=300, batch_size=batch_size, validation_split=0.2, callbacks=[callback])
final_loss, final_mae, final_r2 = final_model.evaluate(X_test, ytest)
y_pred_test = final_model.predict(X_test)
y_pred_train = final_model.predict(X_train)

print("Test loss: ", final_loss, "Test MAE: ", final_mae, "Test R2", final_r2)
train_err=final_history.history['mae']
val_err=final_history.history['val_mae']
plt.figure()
plt.plot(train_err, label='Training')
plt.plot(val_err, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('MAE_plot_CNN.png')

train_loss = final_history.history['loss']
val_loss = final_history.history['val_loss']
plt.figure()
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_CNN.png')

plt.figure()
plt.scatter(y_pred_test, ytest, label = 'Testing Data')
plt.scatter(y_pred_train, ytrain, label = 'Traininging Data')
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.legend()
plt.show()
plt.savefig('diagonal_validation_CNN.png')
