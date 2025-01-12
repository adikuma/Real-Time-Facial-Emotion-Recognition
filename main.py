import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.data.dataset import get_dataloaders
from src.models.net import Net
from src.trainer import Trainer
import argparse

def main(args):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Data loaders created with:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    model = Net(config.num_classes).to(device)
    print(f"Model created with {model.get_n_params():,} trainable parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=emotion_classes
    )
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    if args.test_only:
        print("Running evaluation on test set...")
        trainer.test()
    else:
        print("Starting training...")
        trainer.train()
        print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--test-only', action='store_true', help='Only run evaluation on test set')
    args = parser.parse_args()
    main(args)