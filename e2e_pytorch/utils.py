import numpy as np
import matplotlib.pyplot as plt

def check_overfitting(train_losses, save_plots=True):
    # Extract loss values
    train_loss = train_losses
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

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('training_loss.png')
    plt.show()

    return is_overfitting, analysis