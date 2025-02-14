import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def check_overfitting(train_losses, val_losses=None, train_accuracies=None, val_accuracies=None, save_plots=True):
    # Extract loss and accuracy values
    epochs = range(1, len(train_losses) + 1)

    # Calculate moving averages to smooth out fluctuations
    window = 5
    train_loss_ma = np.convolve(train_losses, np.ones(window) / window, mode='valid')
    val_loss_ma = np.convolve(val_losses, np.ones(window) / window, mode='valid') if val_losses else None
    train_acc_ma = np.convolve(train_accuracies, np.ones(window) / window, mode='valid') if train_accuracies else None
    val_acc_ma = np.convolve(val_accuracies, np.ones(window) / window, mode='valid') if val_accuracies else None

    # Define overfitting criteria
    min_epochs = 10
    sustained_period = 5  # Number of epochs to confirm trend

    # Initialize analysis results
    is_overfitting = False
    analysis = {
        'train_loss': train_losses,
        'train_loss_trend': np.diff(train_loss_ma[-sustained_period:]).mean() if len(train_loss_ma) >= sustained_period else None,
        'min_train_loss': min(train_losses),
        'final_train_loss': train_losses[-1],
        'improvement_rate': None,
        'val_loss': val_losses if val_losses else None,
        'val_loss_trend': np.diff(val_loss_ma[-sustained_period:]).mean() if val_losses and len(val_loss_ma) >= sustained_period else None,
        'min_val_loss': min(val_losses) if val_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'train_accuracy': train_accuracies if train_accuracies else None,
        'val_accuracy': val_accuracies if val_accuracies else None,
        'message': None
    }

    # Calculate recent improvement rate for training loss
    if len(train_losses) >= sustained_period:
        recent_improvement = (train_losses[-sustained_period] - train_losses[-1]) / train_losses[-sustained_period]
        analysis['improvement_rate'] = recent_improvement

    # Check for overfitting conditions
    if len(train_losses) >= min_epochs:
        # Condition 1: Training loss plateauing
        if analysis['improvement_rate'] is not None and analysis['improvement_rate'] < 0.001:
            is_overfitting = True
            analysis['message'] = 'Training has plateaued with minimal improvement'

        # Condition 2: Training loss increasing in recent epochs
        elif analysis['train_loss_trend'] > 0:
            is_overfitting = True
            analysis['message'] = 'Training loss is increasing'

        # Condition 3: Significant deviation from minimum training loss
        elif (train_losses[-1] - analysis['min_train_loss']) / analysis['min_train_loss'] > 0.1:
            is_overfitting = True
            analysis['message'] = 'Current training loss significantly higher than best achieved'

        # Condition 4: Validation loss increasing while training loss decreases
        elif val_losses and analysis['val_loss_trend'] > 0 and analysis['train_loss_trend'] < 0:
            is_overfitting = True
            analysis['message'] = 'Validation loss is increasing while training loss decreases'

        # Condition 5: Validation accuracy decreasing while training accuracy increases
        elif val_accuracies and train_accuracies and np.diff(val_acc_ma[-sustained_period:]).mean() < 0 and np.diff(train_acc_ma[-sustained_period:]).mean() > 0:
            is_overfitting = True
            analysis['message'] = 'Validation accuracy is decreasing while training accuracy increases'

        else:
            analysis['message'] = 'No clear signs of overfitting detected'

    else:
        analysis['message'] = 'Not enough epochs to determine overfitting'

    # Add training and validation statistics
    analysis['statistics'] = {
        'total_epochs': len(train_losses),
        'best_train_epoch': np.argmin(train_losses) + 1,  # Best epoch for training loss
        'best_train_loss': min(train_losses),
        'final_train_loss': train_losses[-1],
        'train_loss_improvement': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100,
        'best_val_epoch': np.argmin(val_losses) + 1 if val_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'val_loss_improvement': (val_losses[0] - val_losses[-1]) / val_losses[0] * 100 if val_losses else None,
        'best_train_accuracy': max(train_accuracies) if train_accuracies else None,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else None,
        'best_val_accuracy': max(val_accuracies) if val_accuracies else None,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else None
    }

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('loss_plot.png')
    plt.show()

    # Plot training and validation accuracy
    if train_accuracies and val_accuracies:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_accuracies, label='Training (MAE) Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation (MAE) Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (MAE)')
        plt.title('Accuracy (MAE) Over Epochs')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig('accuracy_plot.png')
        plt.show()

    return is_overfitting, analysis