import torch.nn as nn
import torch 

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def save_checkpoint(self, path, epoch, optimizer, best_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return epoch, best_acc
    
    def get_n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)