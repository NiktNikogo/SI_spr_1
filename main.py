import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import os

#Determine folder name
current_datetime = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
folder_name = 'Results/' + str(current_datetime) + '/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Create the base model using the functional API so that the Input isn’t a separate layer
def create_initial_model(input_shape, hidden_layers, activations):
    layers = [keras.Input(shape=(input_shape,))]
    for units, activation in zip(hidden_layers, activations):
        layers.append(keras.layers.Dense(units, activation=activation))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    
    model = keras.Sequential(layers)
    
    # Initialize all Dense layers with random weights in [0, 1]
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            weights = layer.get_weights()
            if weights:
                new_weights = [np.random.rand(*w.shape) for w in weights]
                layer.set_weights(new_weights)
    return model

# Save the initial model to a file
def save_initial_model(model, filename='model_start.keras'):
    model.save(folder_name + filename)
    print(f"Model saved to {folder_name + filename}")

# Plot graph for given model
def plot_graph(history, graph_name, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    ax1.plot(history.history["loss"], label="Train loss")
    ax1.plot(history.history["val_loss"], label="Test loss")
    ax1.legend()
    ax2.plot(history.history["accuracy"], label="Train accuracy")
    ax2.plot(history.history["val_accuracy"], label="Test accuracy")
    ax2.legend()
    plt.savefig(folder_name + graph_name + '.png')
    plt.close()    

# Train the given model and return metrics
def train_model(model, X_train, y_train, X_test, y_test, graph_name='default', title='TEMP', error_threshold=0.05, epochs=100, batch_size=32):
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=1, shuffle=False)
    training_time = time.time() - start_time

    final_training_error = history.history['loss'][-1]
    final_test_error = history.history['val_loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_test_accuracy = history.history['val_accuracy'][-1]
    iterations = len(history.history['loss'])
    
    plot_graph(history, graph_name, title)
    
    return model, final_accuracy, final_test_accuracy, final_training_error, final_test_error, iterations, training_time

# Prepare the data and optionally normalize it
def prepare_data(df, sample_indices, normalize=False):
    x = np.delete(df.drop(columns=["Patient_ID", "Diagnosis"]).values, sample_indices, axis=0)
    y = np.delete(df["Diagnosis"].values, sample_indices)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
def copy_weights(base_model, variant_model):
    """ Copies weights from base_model to variant_model, adapting for shape differences. """
    for i, layer in enumerate(variant_model.layers):
        if isinstance(layer, keras.layers.Dense):  # Ensure it's a Dense layer
            try:
                base_weights = base_model.layers[i].get_weights()
                variant_weights = layer.get_weights()

                if base_weights and variant_weights:  # Both must have weights
                    old_w, old_b = base_weights  # Base model weights
                    new_w, new_b = variant_weights  # Variant model weights

                    # Determine the minimum size that can be copied
                    min_rows = min(old_w.shape[0], new_w.shape[0])
                    min_cols = min(old_w.shape[1], new_w.shape[1])
                    min_bias = min(len(old_b), len(new_b))

                    # Copy the overlapping weight values
                    new_w[:min_rows, :min_cols] = old_w[:min_rows, :min_cols]
                    new_b[:min_bias] = old_b[:min_bias]

                    # If the variant layer is larger, fill extra neurons with zeros
                    if new_w.shape[0] > old_w.shape[0] or new_w.shape[1] > old_w.shape[1]:
                        print(f"Warning: Layer {i} is larger in the variant. Filling extra values with zeros.")

                    # Assign the modified weights back to the variant model
                    layer.set_weights([new_w, new_b])
                    print(f"Successfully copied weights for layer {i}.")
                else:
                    print(f"No weights to copy for layer {i}.")

            except Exception as e:
                print(f"Warning: Could not copy weights for layer {i}: {e}")

# Create a variant model, copy overlapping weights, and optionally apply normalization
def create_variant_model(X_train, y_train, X_test, y_test, 
                         base_model, new_hidden_layers, new_activations, 
                         learning_rate=0.001, momentum=0.9, 
                         normalize=False, output_file_name='model_variant.keras', graph_name='default', title='TEMP'):
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Build the variant model
    layers = [keras.Input(shape=(X_train.shape[1],))]
    for units, activation in zip(new_hidden_layers, new_activations):
        layers.append(keras.layers.Dense(units, activation=activation))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    variant_model = keras.Sequential(layers)

    # Copy weights from base model
    copy_weights(base_model, variant_model)

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    variant_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    variant_model, final_accuracy, final_test_accuracy, final_training_error, final_test_error, iterations, training_time = train_model(
        variant_model, X_train, y_train, X_test, y_test, error_threshold=0.05, epochs=50, batch_size=32, graph_name=graph_name, title=title ##ILOŚĆ POKOLEŃ
    )

    variant_model.save(folder_name + output_file_name)
    return variant_model, final_accuracy, final_test_accuracy, final_training_error, final_test_error, iterations, training_time


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    df = pd.read_csv('schizofrenia_dataset.csv')
    
    sample_indices = np.random.choice(df.shape[0], 3, replace=False)
    
    # Prepare data for the base model (without normalization)
    X_train, X_test, y_train, y_test = prepare_data(df, sample_indices, normalize=True) ##GLOBALNA NORMALIZACJA
    
    #Future
    #base_model = tf.keras.models.load_model('my_model.h5')
    
    # Create the base model with architecture [10, 10, 10] and sigmoid activations
    hidden_layers = [10, 10, 10]
    activations = ['sigmoid', 'sigmoid', 'sigmoid']
    base_model = create_initial_model(X_train.shape[1], hidden_layers, activations)
    save_initial_model(base_model)
    
    # Define variant configurations:
    # (new_hidden_layers, new_activations, learning_rate, momentum, normalize, output_file_name, graph_name, graph_title)
    variants = [
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.01, 0.9, False, 'model_1.keras', 'model_1', 'Tanh, 10 10 10, 0.01, 0.9'),
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.001, 0.9, False, 'model_2.keras', 'model_2', 'Tanh, 10 10 10, 0.001, 0.9'),
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.0001, 0.9, False, 'model_3.keras', 'model_3', 'Tanh, 10 10 10, 0.0001, 0.9'),
        
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.01, 0.0, False, 'model_4.keras', 'model_4', 'Tanh, 10 10 10, 0.01, 0.0'),
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.001, 0.0, False, 'model_5.keras', 'model_5', 'Tanh, 10 10 10, 0.001, 0.0'),
        ([10, 10, 10], ['tanh', 'tanh', 'tanh'], 0.0001, 0.0, False, 'model_6.keras', 'model_6', 'Tanh, 10 10 10, 0.0001, 0.0')
    ]
    
    for i, (new_layers, new_acts, lr, mom, norm, outfile, graph_name, title) in enumerate(variants):
        print(f"\nTraining variant model {i+1}...")
        variant_model, final_acc, final_test_acc, final_train_err, final_test_err, iterations, train_time = create_variant_model(
            X_train, y_train, X_test, y_test, base_model, new_layers, new_acts, lr, mom, norm, outfile, graph_name, title
        )
        print(f"\nResults for variant model {i+1}:")
        print(f"Accuracy: {final_acc:.4f}")
        print(f"Test Accuracy: {final_test_acc:.4f}")
        print(f"Training Error: {final_train_err:.4f}")
        print(f"Test Error: {final_test_err:.4f}")
        print(f"Iterations: {iterations}")
        print(f"Training Time: {train_time:.2f} seconds")
        variant_model.summary()

if __name__ == "__main__":
    main()

