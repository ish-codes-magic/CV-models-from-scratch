"""Inference script for AlexNet - predict on single images."""

import argparse
import yaml
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path

from src import AlexNet
from src.data_loader import get_class_names
from utils.checkpoint import CheckpointManager


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_transform(mean, std):
    """Get inference transform."""
    return transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_image(image_path):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    return image


def predict(model, image, transform, device, class_names):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained AlexNet model
        image: PIL Image
        transform: Image transformation
        device: Device to run inference on
        class_names: List of class names
        
    Returns:
        dict: Prediction results
    """
    model.eval()
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    top5_predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top5_idx[0], top5_prob[0])
    ]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'top5_predictions': top5_predictions
    }


def main(args):
    """Main inference function."""
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = AlexNet(num_classes=config['model']['num_classes'])
    
    # Load model weights
    checkpoint_manager = CheckpointManager(save_dir=config['checkpoint']['save_dir'])
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try to find best model
        best_model_path = Path(config['checkpoint']['save_dir']) / 'model_best.pth'
        if best_model_path.exists():
            checkpoint_path = str(best_model_path)
        else:
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
            if not checkpoint_path:
                raise FileNotFoundError("No checkpoint found! Please provide a checkpoint path.")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("Failed to load checkpoint!")
    
    model = model.to(device)
    
    # Get transform
    transform = get_transform(config['dataset']['mean'], config['dataset']['std'])
    
    # Get class names
    class_names = get_class_names()
    
    # Load image
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image)
    
    # Make prediction
    print("Making prediction...")
    results = predict(model, image, transform, device, class_names)
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"\nPredicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']*100:.2f}%")
    
    print("\nTop 5 Predictions:")
    print("-" * 50)
    for i, (class_name, prob) in enumerate(results['top5_predictions'], 1):
        print(f"{i}. {class_name:12}: {prob*100:6.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlexNet Inference on Single Image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (if not provided, uses best or latest)')
    
    args = parser.parse_args()
    main(args)

