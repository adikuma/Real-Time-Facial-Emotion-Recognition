import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np

class Visualizer:
    def __init__(self, save_dir='results/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_metrics(self, metrics_tracker, epoch):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(metrics_tracker.train_losses) + 1)
        
        ax1.plot(epochs, metrics_tracker.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, metrics_tracker.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(epochs, metrics_tracker.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, metrics_tracker.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        ax3.plot(epochs, metrics_tracker.val_f1_scores, 'g-', label='F1 Score')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        
        ax4.plot(epochs, metrics_tracker.val_precisions, 'y-', label='Precision')
        ax4.plot(epochs, metrics_tracker.val_recalls, 'p-', label='Recall')
        ax4.set_title('Validation Precision & Recall')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Score')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'metrics.png'))
        plt.close()

    def plot_confusion_matrix(self, targets, predictions, classes, epoch):
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix.png'))
        plt.close()