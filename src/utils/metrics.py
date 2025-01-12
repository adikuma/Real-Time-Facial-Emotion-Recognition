import numpy as np
import json
import os
from sklearn.metrics import classification_report
import torch

class MetricsTracker:
    def __init__(self, save_dir='results/metrics'):
        self.save_dir = save_dir
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []
        
    def update_train_metrics(self, loss, accuracy):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        
    def update_val_metrics(self, loss, accuracy, f1, precision, recall):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        
    def save_metrics(self):
        metrics = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls
        }
        
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def load_metrics(self):
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.train_losses = metrics['train_losses']
                self.train_accuracies = metrics['train_accuracies']
                self.val_losses = metrics['val_losses']
                self.val_accuracies = metrics['val_accuracies']
                self.val_f1_scores = metrics['val_f1_scores']
                self.val_precisions = metrics['val_precisions']
                self.val_recalls = metrics['val_recalls']

def compute_metrics(outputs, targets, criterion, classes):
    with torch.no_grad():
        loss = criterion(outputs, targets).item()
        _, predictions = outputs.max(1)
        correct = predictions.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = 100. * correct / total
        
        targets_np = targets.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        report = classification_report(targets_np, predictions_np,
                                    target_names=classes,
                                    output_dict=True,
                                    zero_division=0)
        
        avg_f1 = np.mean([report[cls]['f1-score'] for cls in classes])
        avg_precision = np.mean([report[cls]['precision'] for cls in classes])
        avg_recall = np.mean([report[cls]['recall'] for cls in classes])
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'report': report
        }