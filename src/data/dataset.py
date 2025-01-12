import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'sad': 4, 'surprise': 5,
            'neutral': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        self._load_dataset()

    def _load_dataset(self):
        for emotion in os.listdir(self.root_dir):
            if emotion.lower() in self.emotion_to_idx:
                emotion_dir = os.path.join(self.root_dir, emotion)
                for image_file in os.listdir(emotion_dir):
                    image_path = os.path.join(emotion_dir, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(self.emotion_to_idx[emotion.lower()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_emotion_label(self, idx):
        return self.idx_to_emotion[idx]

def get_transforms(config):
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomAffine(
            degrees=config.config['transforms']['train']['rotation_degrees'],
            translate=(config.config['transforms']['train']['translate'],
                      config.config['transforms']['train']['translate']),
            scale=tuple(config.config['transforms']['train']['scale'])
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=config.config['transforms']['train']['random_erase_prob']),
        transforms.Normalize(
            mean=config.config['transforms']['normalize']['mean'],
            std=config.config['transforms']['normalize']['std']
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.config['transforms']['normalize']['mean'],
            std=config.config['transforms']['normalize']['std']
        )
    ])
    
    return train_transform, test_transform

def get_dataloaders(config):
    train_transform, eval_transform = get_transforms(config)
    
    train_dataset = FERDataset(
        config.config['data']['root_dir'],
        mode='train',
        transform=train_transform
    )
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = FERDataset(
        config.config['data']['root_dir'],
        mode='test',
        transform=eval_transform
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader