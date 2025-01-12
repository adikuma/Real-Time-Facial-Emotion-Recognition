# Facial Emotion Recognition with PyTorch

A deep learning model for real-time facial emotion recognition achieving human-level performance on the FER2013 dataset. The model achieves 67.94% accuracy on the test set, which is comparable to human-level accuracy (65-68%) as reported in literature.

## Project Overview

This project implements a facial emotion recognition system using PyTorch, with both training capabilities and real-time inference using OpenCV. The system can recognize seven different emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model Architecture

The project implements two neural network architectures:

1. **Basic CNN (Net)**: A traditional convolutional neural network with the following structure:

   - 4 convolutional stages with increasing channels (64 → 128 → 256 → 512)
   - Each stage contains two conv blocks with batch normalization and ReLU
   - MaxPooling after each stage
   - Final classifier with dropout for regularization
2. **SENet**: A more advanced architecture incorporating Squeeze-and-Excitation blocks for adaptive feature recalibration

   - ResNet-style residual connections
   - SE blocks for channel-wise attention
   - Adaptive feature weighting

## Performance Metrics

The model achieves state-of-the-art performance on par with human accuracy:

- Test Loss: 0.9585
- Accuracy: 67.94%
- F1 Score: 0.6522
- Precision: 0.6779
- Recall: 0.6387

These results are particularly impressive considering that human accuracy on the FER2013 dataset is reported to be between 65-68%.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── net.py           # Basic CNN implementation
│   │   └── se_net.py        # SENet implementation
│   ├── data/
│   │   └── dataset.py       # Data loading and preprocessing
│   ├── trainer.py           # Training loop and validation
│   └── config.py            # Configuration handling
├── realtime_emotion.py      # Real-time emotion detection
├── main.py                  # Training entry point
└── config.yaml             # Configuration file
```

## Configuration

The system is highly configurable through `config.yaml`:

```yaml
data:
  image_size: 48
  batch_size: 64

model:
  num_classes: 7
  learning_rate: 0.001
  num_epochs: 50

training:
  device: 'cuda'
  scheduler:
    type: 'ReduceLROnPlateau'
    factor: 0.75
    patience: 5
    min_lr: 0.00001
```

## Training Process

The training process includes:

1. Data augmentation with rotations, translations, and random erasing
2. Learning rate scheduling with ReduceLROnPlateau
3. Regular checkpointing and best model saving
4. Comprehensive metrics tracking (loss, accuracy, F1, precision, recall)
5. Real-time visualization of training progress

Training metrics are visualized during training:

- Loss curves (training/validation)
- Accuracy progression
- F1 score tracking
- Precision and recall monitoring
- Confusion matrix generation

## Real-time Emotion Detection

The project includes real-time emotion detection capabilities through OpenCV:

1. Face detection using Haar Cascades
2. Real-time frame processing and emotion prediction
3. Visualization of results with bounding boxes and emotion labels
4. Confidence score display

To run real-time detection:

```bash
python realtime_emotion.py
```

## Usage

1. Training:

```bash
python main.py
```

2. Resume training from checkpoint:

```bash
python main.py --checkpoint path/to/checkpoint.pth
```

3. Evaluate on test set:

```bash
python main.py --test-only
```

## Model Training Results

The training curves show:

- Steady decrease in loss
- Convergence of training/validation metrics
- Good generalization (small gap between train/val accuracy)
- Stable F1 score improvement
- Balanced precision and recall

The confusion matrix reveals:

- Strong performance on "Happy" emotion
- Good balance across emotion categories
- Expected confusion between similar emotions (e.g., Fear/Surprise)

## Future Improvements

1. Implement ensemble methods
2. Explore additional architectures (EfficientNet, Vision Transformer)
3. Add cross-validation support
4. Implement data cleaning pipeline
5. Add support for model quantization
