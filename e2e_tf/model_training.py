# model_training.py

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

def train_model(model, train_input_dynamic_final, cate_feature_final, vlt_input, rp_in, initial_stock_in, df_out, rp_out, vlt_out, model_file_name, log_dir, epochs, batch_size):
    start = time.perf_counter()

    # Define callbacks
    checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
        write_graph=True)
    callbacks_list = [checkpoint, tensorboard_callback]

    # Train the model
    history = model.fit(
        [train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in],
        [df_out, rp_out, vlt_out],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
    )

    # After model training, print loss and accuracy statistics
    print(f"Training completed in {time.perf_counter() - start:.2f} seconds")

    # Print the final training loss
    final_loss = history.history['loss'][-1]  # Get the last loss value
    print(f"Final Training Loss: {final_loss}")

    # Optionally, if you are tracking other metrics like accuracy (if applicable), you can print them as well
    if 'accuracy' in history.history:
        final_accuracy = history.history['accuracy'][-1]
        print(f"Final Training Accuracy: {final_accuracy:.4f}")

    # Plot the loss curve during training
    plt.plot(history.history['loss'])
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return history

def check_overfitting(history, save_plots=True):
    # Extract loss values
    train_loss = history.history['loss']
    epochs = range(1, len(train_loss) + 1)

    # Calculate moving averages to smooth out fluctuations
    window = 5
    train_ma = np.convolve(train_loss, np.ones(window) / window, mode='valid')

    # Define overfitting criteria
    min_epochs = 10
    sustained_period = 5  # Number of epochs to confirm trend

    # Initialize analysis results
    is_overfitting = False
    analysis = {
        'train_loss': train_loss,
        'train_loss_trend': np.diff(train_ma[-sustained_period:]).mean(),
        'min_loss': min(train_loss),
        'final_loss': train_loss[-1],
        'improvement_rate': None
    }

    # Calculate recent improvement rate
    if len(train_loss) >= sustained_period:
        recent_improvement = (train_loss[-sustained_period] - train_loss[-1]) / train_loss[-sustained_period]
        analysis['improvement_rate'] = recent_improvement

    # Create detailed loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1)
    plt.plot(epochs[window - 1:], train_ma, 'r-', label='Moving Average', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_plots:
        plt.savefig("training_loss.png")

    # Check for overfitting conditions
    if len(train_loss) >= min_epochs:
        # Condition 1: Loss plateauing
        if analysis['improvement_rate'] is not None and analysis['improvement_rate'] < 0.001:
            is_overfitting = True
            analysis['message'] = 'Training has plateaued with minimal improvement'

        # Condition 2: Loss increasing in recent epochs
        elif analysis['train_loss_trend'] > 0:
            is_overfitting = True
            analysis['message'] = 'Training loss is increasing'

        # Condition 3: Significant deviation from minimum loss
        elif (train_loss[-1] - analysis['min_loss']) / analysis['min_loss'] > 0.1:
            is_overfitting = True
            analysis['message'] = 'Current loss significantly higher than best achieved'

        else:
            analysis['message'] = 'No clear signs of overfitting detected'

    else:
        analysis['message'] = 'Not enough epochs to determine overfitting'

    # Add training statistics
    analysis['statistics'] = {
        'total_epochs': len(train_loss),
        'best_epoch': np.argmin(train_loss) + 1,
        'best_loss': min(train_loss),
        'final_loss': train_loss[-1],
        'loss_improvement': (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    }

    plt.close()
    return is_overfitting, analysis