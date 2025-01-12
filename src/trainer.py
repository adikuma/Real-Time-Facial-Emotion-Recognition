import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from .utils.metrics import MetricsTracker, compute_metrics
from .utils.visualization import Visualizer

class Trainer:
    def __init__(self, model, optimizer, criterion, device, config,
                 train_loader, val_loader, test_loader, classes):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes
        
        self.metrics = MetricsTracker()
        self.visualizer = Visualizer()
        
        self.best_acc = 0
        self.start_epoch = 1
        
        self.scheduler_config = self.config.config['training']['scheduler']
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.scheduler_config['factor'],
            patience=self.scheduler_config['patience'],
            min_lr=self.scheduler_config['min_lr'],
            verbose = True
        )
                
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_outputs = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_outputs.append(output)
                all_targets.append(target)
                _, predicted = output.max(1)
                all_predictions.append(predicted)
        
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        predictions = torch.cat(all_predictions)
        
        metrics = compute_metrics(outputs, targets, self.criterion, self.classes)
        self.visualizer.plot_confusion_matrix(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            self.classes,
            epoch
        )
        
        return metrics
    
    def test(self):
        print("\nEvaluating on test set...")
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_outputs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_outputs.append(output)
                all_targets.append(target)
                _, predicted = output.max(1)
                all_predictions.append(predicted)
        
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        predictions = torch.cat(all_predictions)
        
        metrics = compute_metrics(outputs, targets, self.criterion, self.classes)
        
        self.visualizer.plot_confusion_matrix(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            self.classes,
            'final_test'
        )
        
        print("\nTest Set Results:")
        print(f'Loss: {metrics["loss"]:.4f}')
        print(f'Accuracy: {metrics["accuracy"]:.2f}%')
        print(f'F1 Score: {metrics["f1"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        
        return metrics
    
    def train(self):
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            self.metrics.update_train_metrics(train_loss, train_acc)
            val_metrics = self.validate(epoch)
            self.metrics.update_val_metrics(
                val_metrics['loss'],
                val_metrics['accuracy'],
                val_metrics['f1'],
                val_metrics['precision'],
                val_metrics['recall']
            )
            
            self.scheduler.step(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > self.best_acc:
                self.best_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
            
            self.visualizer.plot_metrics(self.metrics, epoch)
            self.metrics.save_metrics()
            
            print(f'\nEpoch {epoch} Summary:')
            print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}, '
                  f'Accuracy: {val_metrics["accuracy"]:.2f}%')
            print(f'F1: {val_metrics["f1"]:.4f}, '
                  f'Precision: {val_metrics["precision"]:.4f}, '
                  f'Recall: {val_metrics["recall"]:.4f}')
        
        print("\nTraining completed. Running final evaluation on test set...")
        test_metrics = self.test()
        
        with open(os.path.join(self.config.config['training']['results_dir'], 'test_results.txt'), 'w') as f:
            f.write("Final Test Results:\n")
            f.write(f"Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.2f}%\n")
            f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['recall']:.4f}\n")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'metrics': {
                'train_losses': self.metrics.train_losses,
                'train_accuracies': self.metrics.train_accuracies,
                'val_losses': self.metrics.val_losses,
                'val_accuracies': self.metrics.val_accuracies,
                'val_f1_scores': self.metrics.val_f1_scores,
                'val_precisions': self.metrics.val_precisions,
                'val_recalls': self.metrics.val_recalls
            }
        }
        
        if is_best:
            path = os.path.join(self.config.config['training']['save_dir'], 'best_model.pth')
        else:
            path = os.path.join(self.config.config['training']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        
        metrics = checkpoint['metrics']
        self.metrics.train_losses = metrics['train_losses']
        self.metrics.train_accuracies = metrics['train_accuracies']
        self.metrics.val_losses = metrics['val_losses']
        self.metrics.val_accuracies = metrics['val_accuracies']
        self.metrics.val_f1_scores = metrics['val_f1_scores']
        self.metrics.val_precisions = metrics['val_precisions']
        self.metrics.val_recalls = metrics['val_recalls']